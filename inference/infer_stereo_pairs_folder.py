"""Stereo NEEDLE-KEYPOINT inference over EVERY left/right video pair in a folder,
stitched into ONE output video.

This is the stereo-keypoint counterpart of infer_videos_folder.py (which only did
mono segmentation). For each pair it runs the SAME pipeline as
realtime_stereo_keypoints_v3_accel.py -- segment both eyes -> needle keypoints ->
stereo triangulation -> 6-DoF pose -> reprojection overlay -- but:

  * the model/engine is built ONCE and reused across all pairs (no reload),
  * all pairs are written into a SINGLE VideoWriter (no ffmpeg concat needed),
  * left/right are auto-paired by filename (``<grp>_01*`` = left / cam1,
    ``<grp>_02*`` = right / cam2),
  * each eye is strided to a common --target-fps, which both downsamples for
    speed AND re-synchronises a pair whose eyes were encoded at different fps
    (e.g. 60fps left vs 30fps right -> left stride 2, right stride 1).

Run (from EndoNeedle6DoF/inference, with UniMatch-V2_local on PYTHONPATH so
`test`/`util` import):

  set PYTHONPATH=D:/study/code/UniMatch-V2_local
  python infer_stereo_pairs_folder.py \
      --folder "D:/study/code/autonomous_surgery/suture_needle_video/March" \
      --config ../configs/surgical_combined_base.yaml \
      --checkpoint ../weights/best.pth \
      --calib "D:/study/code/SAM2-Plus/tools/needle_calib.json" \
      --sam2-tools "D:/study/code/SAM2-Plus/tools" \
      --seg-engine ../weights/seg_engine_fp32_320.ts --seg-size 320 \
      --output "D:/study/code/autonomous_surgery/suture_needle_video/March_stereo_keypoints.mp4"
"""
import argparse
import glob
import os
import re
import time

import cv2
import numpy as np
import torch
import yaml

# the heavy lifting + every per-frame helper already lives in the v3 driver;
# importing it also pulls `from test import build_inference_model` (needs
# UniMatch-V2_local on PYTHONPATH) and registers infer_accel.
import realtime_stereo_keypoints_v3_accel as D
import infer_accel


def discover_pairs(folder):
    """Group .mp4s into (group, left, right). left=<grp>_01*, right=<grp>_02*."""
    pairs = {}
    for p in sorted(glob.glob(os.path.join(folder, '*.mp4'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r'(.+?)_(\d+)', stem)          # e.g. '1_01', '2_01-1'
        if not m:
            continue
        grp, eye = m.group(1), int(m.group(2))
        pairs.setdefault(grp, {})[eye] = p
    out = []
    for grp in sorted(pairs):
        d = pairs[grp]
        if 1 in d and 2 in d:
            out.append((grp, d[1], d[2]))
        else:
            print(f'[warn] group {grp}: incomplete pair {sorted(d)} -> skipped')
    return out


def fps_of(path):
    c = cv2.VideoCapture(path)
    f = c.get(cv2.CAP_PROP_FPS) or 0.0
    c.release()
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--sam2-tools', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seg-engine', default=None)
    p.add_argument('--seg-size', type=int, default=320)
    p.add_argument('--num-keypoints', type=int, default=5)
    p.add_argument('--needle-class', type=int, default=1)
    p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--target-fps', type=float, default=30.0,
                   help='each eye is strided to ~this fps (also re-syncs mismatched pairs)')
    p.add_argument('--groups', default=None,
                   help='comma-separated group ids to run (default: all), e.g. "4"')
    p.add_argument('--out-height', type=int, default=720, help='output canvas height (L|R)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--model-radius', type=float, default=None)
    args = p.parse_args()

    cfg = yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda') and not args.no_amp
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('[stereo] building model ...', flush=True)
    bundle = D.build_inference_model(cfg, args.checkpoint, device, visual_adapter=False)
    if args.seg_engine:
        infer_accel.attach_engine_to_bundle(bundle, args.seg_engine, device)
    nk = D.load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    rvecL = np.zeros(3); tvecL = np.zeros(3)
    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel()
    tvecR = np.asarray(calib['t'], float).ravel()
    kp_names = (["tip"] + [f"k{i}" for i in range(1, args.num_keypoints - 1)] + ["tail"])

    pairs = discover_pairs(args.folder)
    if args.groups:
        want = {g.strip() for g in args.groups.split(',')}
        pairs = [pr for pr in pairs if pr[0] in want]
    if not pairs:
        raise SystemExit(f'no complete L/R pairs found in {args.folder}')
    print(f'[stereo] {len(pairs)} pairs:', [g for g, _, _ in pairs], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    writer = None
    ow = oh = None
    total = 0
    t_start = time.perf_counter()

    for pi, (grp, lpath, rpath) in enumerate(pairs):
        lf, rf = fps_of(lpath) or args.target_fps, fps_of(rpath) or args.target_fps
        ls = max(1, int(round(lf / args.target_fps)))
        rs = max(1, int(round(rf / args.target_fps)))
        capL = cv2.VideoCapture(lpath); capR = cv2.VideoCapture(rpath)
        print(f'[pair {pi + 1}/{len(pairs)}] grp={grp}  L={os.path.basename(lpath)}({lf:.0f}fps/str{ls}) '
              f'R={os.path.basename(rpath)}({rf:.0f}fps/str{rs})', flush=True)

        pk = D.PoseKalman()                       # reset pose smoother per clip
        fps_ema = 0.0
        kept = 0
        clip_t0 = time.perf_counter()
        while True:
            L = R = None
            for _ in range(ls):
                okL, L = capL.read()
                if not okL:
                    L = None; break
            for _ in range(rs):
                okR, R = capR.read()
                if not okR:
                    R = None; break
            if L is None or R is None:
                break

            t0 = time.perf_counter()
            ml, mr = D.seg_masks_batch(bundle, [L, R], args.seg_size, device, use_amp)
            needleL = ml == args.needle_class; needleR = mr == args.needle_class
            threadL = ml == args.thread_class
            out = None
            if needleL.sum() >= 20 and needleR.sum() >= 20:
                try:
                    out, _ = nk.process_frame(needleL, needleR, threadL, calib,
                                              args.num_keypoints, model_radius=args.model_radius)
                except Exception:
                    out = None
            if out is not None:
                ts, rs_ = pk.update(out['pose']['t'], out['pose']['rvec'])
                out['pose']['t'] = list(map(float, ts))
                out['pose']['rvec'] = list(map(float, rs_))
                out['pose']['R'] = cv2.Rodrigues(np.asarray(rs_, float))[0].tolist()
            else:
                pk.coast()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = max(time.perf_counter() - t0, 1e-9)
            fps_ema = (1.0 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1.0 / dt)

            # ---- draw exactly like the driver ----
            visL = D.overlay_segmentation(L, ml, args.needle_class, args.thread_class)
            visR = D.overlay_segmentation(R, mr, args.needle_class, args.thread_class)
            rep_err = D.reprojection_error(out, calib, rvecR, tvecR)
            if out is not None:
                # one bad frame (non-finite reprojection / degenerate pose) must not
                # kill the whole run -> draw best-effort, skip overlays on failure.
                try:
                    nk.draw_debug(visL, out['left'], kp_names, out['visible'], tag=None)
                    nk.draw_debug(visR, out['right'], kp_names, out['visible'], tag=None)
                    Ro = out['pose']['R']; to = out['pose']['t']
                    D.draw_pose_axes(visL, Ro, to, calib['K1'], calib['D1'], rvecL, tvecL)
                    D.draw_pose_axes(visR, Ro, to, calib['K2'], calib['D2'], rvecR, tvecR)
                    D.draw_reproj(visL, out['xyz_mm'], calib['K1'], calib['D1'], rvecL, tvecL)
                    D.draw_reproj(visR, out['xyz_mm'], calib['K2'], calib['D2'], rvecR, tvecR)
                except (cv2.error, OverflowError, ValueError):
                    pass
            canvas = D.hstack_same_h(visL, visR)

            lines = [f'pair {pi + 1}/{len(pairs)}  grp {grp}', f'FPS {fps_ema:5.1f}']
            if out is not None:
                t = out['pose']['t']; eu = D.rot_to_euler_deg(out['pose']['R'])
                lines += [f't=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm',
                          f'rot=({eu[0]:.0f},{eu[1]:.0f},{eu[2]:.0f})deg',
                          'reproj=N/A' if rep_err is None else f"reproj={rep_err['mean']:.2f}px"]
            for k, s in enumerate(lines):
                yk = 30 + 30 * k
                cv2.putText(canvas, s, (12, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, s, (12, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

            if oh is None:
                oh = args.out_height
                ow = int(round(canvas.shape[1] * oh / canvas.shape[0]))
                ow -= ow % 2
            if canvas.shape[0] != oh:
                canvas = cv2.resize(canvas, (ow, oh))
            if writer is None:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.target_fps, (ow, oh))
                if not writer.isOpened():
                    raise SystemExit(f'cannot open writer: {args.output}')
            writer.write(canvas)
            kept += 1; total += 1
            if kept % 100 == 0:
                el = time.perf_counter() - clip_t0
                print(f'    {kept} frames  ({kept / max(el, 1e-9):.1f} fps)', flush=True)
        capL.release(); capR.release()
        el = time.perf_counter() - clip_t0
        print(f'[pair done] grp={grp}: {kept} frames in {el / 60:.1f} min', flush=True)

    if writer is not None:
        writer.release()
    dt = time.perf_counter() - t_start
    print(f'[DONE] {total} frames from {len(pairs)} pairs in {dt / 60:.1f} min -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
