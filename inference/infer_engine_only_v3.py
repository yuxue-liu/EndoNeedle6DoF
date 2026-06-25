"""Process-overlapped source-free stereo needle keypoint + 6-DoF pose inference.

v3 of infer_engine_only.py. SAME pipeline, SAME math, SAME engine, SAME numbers
as v1 — the only difference is that the CPU pose registration runs in WORKER
PROCESSES that overlap the GPU forward. Output is identical to v1 (same engine
masks, same process_frame, same Kalman applied in frame order).

WHY A PROCESS, NOT A THREAD
---------------------------
infer_engine_only_v2.py tried to overlap the GPU forward (thread A) with the CPU
pose (thread B). It was SLOWER: the pose stage has a large pure-Python section
(needle_keypoints `_skeleton_points`) that holds the GIL, and the DINOv2 forward
on a tensor-core-less GPU (GTX 1650 Ti) is KERNEL-LAUNCH-bound — its many small
kernels are launched CPU-side under the GIL. The two stages fought over the GIL
and the forward ballooned from ~90 ms to ~345 ms.

Worker PROCESSES have their own GIL, so the pose runs truly in parallel. Two
extra benefits on this hardware:
  * the GPU is kept continuously busy (forward back-to-back) so it stays clocked
    high instead of down-clocking during the idle pose phase, and
  * with >=2 pose workers the ~140 ms pose latency is fully hidden behind the
    ~90 ms (TorchScript) / ~60 ms (TensorRT) forward, so throughput becomes
    forward-bound.

Measured on the local 1650 Ti, seg-size 320, TensorRT-FP32 engine, 2 workers:
    serial v1   ~4 fps   ->   v3   ~12 fps   (~3x), output bit-identical.

The pose result for frame N is consumed `--depth` frames later (a small fixed
latency), then smoothed by the SAME Kalman in frame order, so the saved video /
JSONL is identical to v1 apart from that startup latency.

Masks are shipped to workers bit-packed (np.packbits: 1080x1920 bool -> ~260 KB)
to keep inter-process transfer cheap.

CLI is a superset of infer_engine_only.py. New:
    --workers N   pose worker processes (default 2)
    --depth   D   pipeline depth / max in-flight frames (default workers+1)
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path as _P

import numpy as np
import cv2
import torch

import infer_accel
import infer_engine_only as E   # reuse source, seg, Kalman, drawing helpers


# --------------------------------------------------------------- worker process
# Globals are initialised once per worker (Windows spawn re-imports this module,
# so keep top-level import-safe; the heavy state lives behind _winit).
_W = {}


def _winit(calib_path, tools_dir, num_kp, needle_class, thread_class, radius):
    sys.path.insert(0, tools_dir)
    import needle_keypoints_v2 as nk
    _W['nk'] = nk
    _W['calib'] = nk.load_calib(calib_path)
    _W['nkp'] = num_kp
    _W['nc'] = needle_class
    _W['tc'] = thread_class
    _W['radius'] = radius


def _pose_task(shape, packL, packR, packTL):
    """Unpack the bit-packed needle/thread masks and run pose registration.
    Returns the `out` dict (picklable) or None. Stateless: the Kalman lives in
    the parent so smoothing stays sequential/identical to v1."""
    n = shape[0] * shape[1]
    nL = np.unpackbits(packL)[:n].astype(bool).reshape(shape)
    nR = np.unpackbits(packR)[:n].astype(bool).reshape(shape)
    tL = np.unpackbits(packTL)[:n].astype(bool).reshape(shape)
    if nL.sum() < 20 or nR.sum() < 20:
        return None
    try:
        out, _ = _W['nk'].process_frame(nL, nR, tL, _W['calib'], _W['nkp'],
                                        model_radius=_W['radius'])
        return out
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', required=True, help='pre-exported TorchScript/TensorRT engine (.ts)')
    p.add_argument('--calib', required=True)
    p.add_argument('--needle-model', default=None)
    p.add_argument('--model-radius', type=float, default=None)
    # sources
    p.add_argument('--root', type=_P); p.add_argument('--dataset'); p.add_argument('--key')
    p.add_argument('--left'); p.add_argument('--right')
    p.add_argument('--capture'); p.add_argument('--layout', choices=['sbs', 'tb'], default='sbs')
    p.add_argument('--limit', type=int, default=0)
    # params
    p.add_argument('--num-keypoints', type=int, default=5)
    p.add_argument('--needle-class', type=int, default=1)
    p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--seg-size', type=int, default=640, help='MUST match the engine export --seg-size')
    p.add_argument('--stride', type=int, default=1,
                   help='run segmentation+pose every Nth frame, reusing the last result on '
                        'skipped frames (throughput knob; >1 changes output). 1=every frame.')
    p.add_argument('--patch', type=int, default=14)
    p.add_argument('--view-height', type=int, default=720, help='downscale L|R canvas (0=full)')
    p.add_argument('--sam2-tools', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'SAM2-Plus', 'tools'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2, help='pose worker processes')
    p.add_argument('--depth', type=int, default=0, help='max in-flight frames (default workers+1)')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-async', action='store_true', help='disable the read prefetch thread')
    p.add_argument('--no-reproject', action='store_true')
    p.add_argument('--show', action='store_true')
    p.add_argument('--save-video', default=None)
    p.add_argument('--save-results', default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    engine = infer_accel.load_seg_engine(args.engine, device)
    print(f'[engine-only-v3] segmentation engine = {args.engine}')
    tools_dir = os.path.abspath(args.sam2_tools)
    nk = E.load_nk(tools_dir)
    calib = nk.load_calib(args.calib)
    if args.needle_model and os.path.isfile(args.needle_model):
        args.model_radius = float(json.loads(
            open(args.needle_model, encoding='utf-8').read())['radius_mm'])
    if args.model_radius:
        print(f'[engine-only-v3] model-based pose: fixed needle radius = {args.model_radius:.2f} mm')

    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel()
    tvecR = np.asarray(calib['t'], float).ravel()
    kp_names = (['tip'] + [f'k{i}' for i in range(1, args.num_keypoints - 1)] + ['tail'])

    src = E.StereoSource(args)
    if not args.no_async:
        src = infer_accel.PrefetchReader(src)
    pk = None if args.no_smooth else E.PoseKalman()
    rjsonl = open(args.save_results, 'w', encoding='utf-8') if args.save_results else None
    writer = None
    WIN = 'engine-only v3 stereo keypoints (q=quit)'

    nworkers = max(1, args.workers)
    depth = args.depth if args.depth > 0 else nworkers + 1
    exe = ProcessPoolExecutor(
        max_workers=nworkers, initializer=_winit,
        initargs=(args.calib, tools_dir, args.num_keypoints,
                  args.needle_class, args.thread_class, args.model_radius))
    # warm the worker pool (spawn + import + first call) so it doesn't stall frame 0
    dummy = np.packbits(np.zeros((4, 4), bool))
    list(exe.map(_pose_task, [(4, 4)] * nworkers,
                 [dummy] * nworkers, [dummy] * nworkers, [dummy] * nworkers))

    pend = deque()             # (fi, stem, L, R, ml, mr, fut|None)
    last_out = None
    fi = 0; n_out = 0
    fps = 0.0
    last_tick = [None]      # perf_counter of the previous consumed frame (output cadence)
    print(f'[engine-only-v3] started ({nworkers} pose workers, depth {depth})'
          + ('' if not args.show else ' — press q to quit'))
    t_wall0 = time.perf_counter()
    stop = False

    def consume(item):
        """Apply Kalman + draw/write/show for one completed frame (frame order)."""
        nonlocal writer, last_out, n_out, fps
        fi_, stem, L, R, ml, mr, fut = item
        if fut is None:                      # --stride skipped frame: reuse last
            out = last_out
        else:
            out = fut.result()
            if pk is not None:
                if out is not None:
                    ts, rs = pk.update(out['pose']['t'], out['pose']['rvec'])
                    out['pose']['t'] = list(map(float, ts))
                    out['pose']['rvec'] = list(map(float, rs))
                    out['pose']['R'] = cv2.Rodrigues(np.asarray(rs, float))[0].tolist()
                else:
                    pk.coast()
            last_out = out
        # Live fps = true output cadence (wall time between consecutive consumed
        # frames), NOT the cost of this consume() — in the pipeline the read/
        # forward/submit work happens in the main loop and the pose future is
        # usually already done, so timing only consume() would wildly overstate
        # throughput. The first frame has no predecessor, so it seeds the clock.
        now = time.perf_counter()
        if last_tick[0] is not None:
            dt = now - last_tick[0]
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)
        last_tick[0] = now

        if rjsonl is not None:
            if out is not None:
                needle = {"keypoints": [
                    {"name": kp_names[i], "x": out['left'][i][0], "y": out['left'][i][1],
                     "x_right": out['right'][i][0], "y_right": out['right'][i][1],
                     "xyz_mm": out['xyz_mm'][i], "visible": int(out['visible'][i])}
                    for i in range(args.num_keypoints)],
                    "pose": out['pose'], "conf": out['conf']}
            else:
                needle = None
            rjsonl.write(json.dumps({"frame": fi_, "stem": stem,
                                     "needle": needle}) + '\n')

        if args.show or args.save_video:
            visL = E.overlay_segmentation(L, ml, args.needle_class, args.thread_class)
            visR = E.overlay_segmentation(R, mr, args.needle_class, args.thread_class)
            if out is not None:
                nk.draw_debug(visL, out['left'], kp_names, out['visible'], tag=None)
                nk.draw_debug(visR, out['right'], kp_names, out['visible'], tag=None)
                if not args.no_reproject:
                    Ro = out['pose']['R']; to = out['pose']['t']
                    E.draw_pose_axes(visL, Ro, to, calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                    E.draw_pose_axes(visR, Ro, to, calib['K2'], calib['D2'], rvecR, tvecR)
                    E.draw_reproj(visL, out['xyz_mm'], calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                    E.draw_reproj(visR, out['xyz_mm'], calib['K2'], calib['D2'], rvecR, tvecR)
            canvas = E.hstack_same_h(visL, visR)
            cv2.putText(canvas, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            if out is not None:
                t = out['pose']['t']; eu = E.rot_to_euler_deg(out['pose']['R'])
                for k, s in enumerate((f"t=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm",
                                       f"rot=({eu[0]:.0f},{eu[1]:.0f},{eu[2]:.0f})deg")):
                    yk = 64 + 28 * k
                    cv2.putText(canvas, s, (10, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(canvas, s, (10, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            if args.view_height and canvas.shape[0] > args.view_height:
                s = args.view_height / canvas.shape[0]
                canvas = cv2.resize(canvas, (int(round(canvas.shape[1] * s)), args.view_height),
                                    interpolation=cv2.INTER_AREA)
            if args.save_video:
                if writer is None:
                    h, w = canvas.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = (cv2.VideoWriter(args.save_video, fourcc, 20.0, (w, h))
                              if args.no_async else
                              infer_accel.AsyncVideoWriter(args.save_video, fourcc, 20.0, (w, h)))
                writer.write(canvas)
            if args.show:
                cv2.imshow(WIN, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    return False
        n_out += 1
        return True

    last_ml = last_mr = None
    while not stop:
        L, R, stem = src.read()
        if L is None:
            break
        run_seg = (last_ml is None) or (fi % max(1, args.stride) == 0)
        if run_seg:
            ml, mr = E.seg_engine_batch(engine, [L, R], args.seg_size, device, args.patch)
            sh = ml.shape
            packL = np.packbits(ml == args.needle_class)
            packR = np.packbits(mr == args.needle_class)
            packTL = np.packbits(ml == args.thread_class)
            fut = exe.submit(_pose_task, sh, packL, packR, packTL)
            last_ml, last_mr = ml, mr
        else:
            ml, mr, fut = last_ml, last_mr, None     # reuse last masks + last_out
        pend.append((fi, stem, L, R, ml, mr, fut))
        fi += 1
        if len(pend) >= depth:
            if not consume(pend.popleft()):
                stop = True
    # drain remaining in-flight frames
    while pend and not stop:
        if not consume(pend.popleft()):
            break

    wall = time.perf_counter() - t_wall0
    src.release()
    exe.shutdown(wait=True, cancel_futures=True)
    if writer is not None:
        writer.release()
    if rjsonl is not None:
        rjsonl.close()
        print(f'[engine-only-v3] results -> {args.save_results}')
    cv2.destroyAllWindows()
    eff = n_out / max(wall, 1e-6)
    print(f'[engine-only-v3] done — {n_out} frames in {wall:.1f}s, '
          f'~{eff:.1f} fps (process-overlapped wall-clock)')


if __name__ == '__main__':
    main()
