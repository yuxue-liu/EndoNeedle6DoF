"""Pipelined source-free stereo needle keypoint + 6-DoF pose inference.

v2 of infer_engine_only.py. SAME pipeline, SAME numbers, SAME CLI — the only
change is HOW the per-frame stages are scheduled.

WHY THIS EXISTS
---------------
Measured on the local GTX 1650 Ti (seg-size 320), the per-frame cost splits into
two near-equal halves that ran *serially* in infer_engine_only.py:

    GPU forward (engine)   ~91 ms     <- pure GPU
    pose registration      ~115 ms    <- pure CPU (numpy / opencv / skimage)
    ----------------------------------
    serial total           ~226 ms -> 4.4 fps

The forward is GPU-bound and the pose is CPU-bound, and the heavy CPU work
(connectedComponents, skeletonize, ellipse fit) releases the GIL, while the CUDA
forward also releases the GIL. So they can run AT THE SAME TIME on different
frames. This module runs a 3-stage software pipeline:

    [read thread] --frames--> [GPU thread: preprocess+forward] --masks--> [main: pose+draw+write]

so end-to-end throughput becomes ~max(read, forward, pose+draw) instead of their
sum. On the 1650 Ti that is ~max(91, 115) -> ~8 fps at seg-size 320, with
BIT-IDENTICAL output to v1 (same engine, same process_frame, same Kalman; only
the order of overlap differs, not the math).

There is NO precision change. To ALSO cut the 91 ms forward, build a TensorRT
FP32 engine with export_seg_engine_fp32.py and pass it as --engine; this script
loads any engine load_seg_engine accepts.

CLI is a superset of infer_engine_only.py. New/changed:
    --queue N        pipeline depth between stages (default 4)
    (everything else is identical; --no-async disables the read prefetch only)

Note: --show must run cv2.imshow on the main thread (it does); the GPU forward is
on the worker thread, which is fine for a single CUDA context.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path as _P

import numpy as np
import cv2
import torch

import infer_accel
import infer_engine_only as E   # reuse source, seg, Kalman, drawing helpers (no duplication)


# --------------------------------------------------------------- GPU stage worker
class _SegWorker(threading.Thread):
    """Stage 2: pull stereo pairs from `src`, run the segmentation engine, and
    push (fi, stem, L, R, ml, mr, run_seg) to `outq`. Runs on its own thread so
    the GPU forward overlaps the main thread's CPU pose/draw of the PREVIOUS frame.

    --stride is honoured here: on skipped frames the forward is not run and the
    last masks are forwarded with run_seg=False so the consumer reuses last pose."""
    def __init__(self, src, engine, args, device, outq, prof_fwd):
        super().__init__(daemon=True)
        self.src, self.engine, self.args = src, engine, args
        self.device, self.outq, self.prof_fwd = device, outq, prof_fwd

    def run(self):
        fi = 0
        last_ml = last_mr = None
        stride = max(1, self.args.stride)
        while True:
            L, R, stem = self.src.read()
            if L is None:
                self.outq.put(None)               # sentinel: end of stream
                break
            run_seg = (last_ml is None) or (fi % stride == 0)
            if run_seg:
                prof = {} if self.prof_fwd is not None else None
                ml, mr = E.seg_engine_batch(self.engine, [L, R], self.args.seg_size,
                                            self.device, self.args.patch, prof)
                if prof is not None:
                    for k, v in prof.items():
                        self.prof_fwd[k] = self.prof_fwd.get(k, 0.0) + v
                last_ml, last_mr = ml, mr
            else:
                ml, mr = last_ml, last_mr
            self.outq.put((fi, stem, L, R, ml, mr, run_seg))
            fi += 1


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
    p.add_argument('--queue', type=int, default=4, help='pipeline depth between GPU and CPU stages')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-async', action='store_true', help='disable the read prefetch thread')
    p.add_argument('--no-reproject', action='store_true')
    p.add_argument('--show', action='store_true')
    p.add_argument('--profile', action='store_true',
                   help='accumulate per-stage timing and print the breakdown at the end; '
                        'reports both the GPU-stage and CPU-stage wall time so you can see overlap')
    p.add_argument('--save-video', default=None)
    p.add_argument('--save-results', default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    engine = infer_accel.load_seg_engine(args.engine, device)
    print(f'[engine-only-v2] segmentation engine = {args.engine}')
    nk = E.load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    if args.needle_model and os.path.isfile(args.needle_model):
        args.model_radius = float(json.loads(
            open(args.needle_model, encoding='utf-8').read())['radius_mm'])
    if args.model_radius:
        print(f'[engine-only-v2] model-based pose: fixed needle radius = {args.model_radius:.2f} mm')

    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel()
    tvecR = np.asarray(calib['t'], float).ravel()
    kp_names = (['tip'] + [f'k{i}' for i in range(1, args.num_keypoints - 1)] + ['tail'])

    src = E.StereoSource(args)
    if not args.no_async:
        src = infer_accel.PrefetchReader(src)
    pk = None if args.no_smooth else E.PoseKalman()
    rjsonl = open(args.save_results, 'w', encoding='utf-8') if args.save_results else None
    writer = None
    WIN = 'engine-only v2 stereo keypoints (q=quit)'
    fps = 0.0
    prof_fwd = {} if args.profile else None     # GPU-stage timing (worker thread)
    prof_cpu = {} if args.profile else None     # CPU-stage timing (main thread)

    # start GPU stage on its own thread; this main thread is the CPU (pose+draw) stage
    outq = queue.Queue(maxsize=max(2, args.queue))
    worker = _SegWorker(src, engine, args, device, outq, prof_fwd)
    worker.start()

    last_out = None
    n_out = 0; n_pose = 0
    print('[engine-only-v2] started' + ('' if not args.show else ' — press q to quit'))
    t_wall0 = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        item = outq.get()
        if item is None:
            break
        fi, stem, L, R, ml, mr, run_seg = item

        if run_seg:
            needleL = ml == args.needle_class; needleR = mr == args.needle_class
            threadL = ml == args.thread_class
            out = None
            _tp = time.perf_counter()
            if needleL.sum() >= 20 and needleR.sum() >= 20:
                try:
                    out, _ = nk.process_frame(needleL, needleR, threadL, calib,
                                              args.num_keypoints, model_radius=args.model_radius)
                except Exception:
                    out = None
            if prof_cpu is not None:
                prof_cpu['pose'] = prof_cpu.get('pose', 0.0) + (time.perf_counter() - _tp)
                n_pose += 1
            if pk is not None:
                if out is not None:
                    ts, rs = pk.update(out['pose']['t'], out['pose']['rvec'])
                    out['pose']['t'] = list(map(float, ts))
                    out['pose']['rvec'] = list(map(float, rs))
                    out['pose']['R'] = cv2.Rodrigues(np.asarray(rs, float))[0].tolist()
                else:
                    pk.coast()
            last_out = out
        else:
            out = last_out

        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)

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
            rjsonl.write(json.dumps({"frame": fi, "stem": stem, "fps": round(fps, 2),
                                     "needle": needle}) + '\n')

        if args.show or args.save_video:
            _td = time.perf_counter()
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
            if prof_cpu is not None:
                prof_cpu['draw'] = prof_cpu.get('draw', 0.0) + (time.perf_counter() - _td)
            if args.show:
                cv2.imshow(WIN, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
        n_out += 1

    wall = time.perf_counter() - t_wall0
    src.release()
    if writer is not None:
        writer.release()
    if rjsonl is not None:
        rjsonl.close()
        print(f'[engine-only-v2] results -> {args.save_results}')
    cv2.destroyAllWindows()
    eff = n_out / max(wall, 1e-6)
    print(f'[engine-only-v2] done — {n_out} frames in {wall:.1f}s, '
          f'~{eff:.1f} fps (pipelined, wall-clock)')
    if args.profile and n_out > 0:
        gpu = sum(prof_fwd.values())
        cpu = sum(prof_cpu.values())
        print('[profile] GPU stage (worker thread, overlaps CPU stage):')
        for k in ('pre', 'fwd', 'post'):
            print(f'    {k:<10} {1000.0*prof_fwd.get(k,0.0)/n_out:7.2f} ms/frame')
        print(f'    GPU SUM   {1000.0*gpu/n_out:7.2f} ms/frame')
        print('[profile] CPU stage (main thread, overlaps GPU stage):')
        print(f'    pose      {1000.0*prof_cpu.get("pose",0.0)/max(n_pose,1):7.2f} ms/pose-frame')
        print(f'    draw      {1000.0*prof_cpu.get("draw",0.0)/n_out:7.2f} ms/frame')
        print(f'    CPU SUM   {1000.0*cpu/n_out:7.2f} ms/frame')
        print(f'[profile] pipelined wall-clock = {1000.0*wall/n_out:7.2f} ms/frame '
              f'(~max of the two stages, not their sum)')


if __name__ == '__main__':
    main()
