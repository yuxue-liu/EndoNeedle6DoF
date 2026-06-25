"""Run segmentation inference on EVERY .mp4 in a folder and stitch the overlays
into ONE output video.

Monocular demo driver (no stereo / no pose): for each video it reads frames,
runs the SAME seg engine forward as infer_engine_only.py (seg_engine_batch),
overlays the predicted needle/thread mask, burns a small HUD (clip name, frame#,
FPS), and writes every frame into a single shared VideoWriter so all clips end up
in one mp4.

To keep playback real-time and the run tractable on the local GTX 1650 Ti, each
clip is strided to a common target fps (default 30) and the canvas is downscaled
to --out-width.

Example (from EndoNeedle6DoF/inference):
    python infer_videos_folder.py \
        --folder "D:/study/code/autonomous_surgery/suture_needle_video/March" \
        --engine ../weights/seg_engine_trtfp32_320.ts --seg-size 320 \
        --output "D:/study/code/autonomous_surgery/suture_needle_video/March_seg_inference.mp4"
"""
import argparse
import glob
import os
import time

import cv2
import torch

from infer_accel import load_seg_engine
from infer_engine_only import seg_engine_batch, overlay_segmentation


def draw_hud(bgr, lines):
    pad, lh = 8, 26
    w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for s in lines)
    cv2.rectangle(bgr, (0, 0), (w + 2 * pad, lh * len(lines) + pad), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(bgr, s, (pad, pad + 20 + i * lh), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', required=True, help='folder of .mp4 videos')
    p.add_argument('--engine', default='../weights/seg_engine_trtfp32_320.ts')
    p.add_argument('--output', required=True, help='single stitched output mp4')
    p.add_argument('--seg-size', type=int, default=320)
    p.add_argument('--patch', type=int, default=14)
    p.add_argument('--target-fps', type=float, default=30.0,
                   help='clips are strided to roughly this fps and the output plays at it')
    p.add_argument('--out-width', type=int, default=960, help='output canvas width (height keeps AR)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--needle-class', type=int, default=1)
    p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.4)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f'[run] device={device}  engine={args.engine}', flush=True)
    engine = load_seg_engine(args.engine, device)

    vids = sorted(glob.glob(os.path.join(args.folder, '*.mp4')))
    if not vids:
        raise SystemExit(f'no .mp4 found in {args.folder}')
    print(f'[run] {len(vids)} videos:', [os.path.basename(v) for v in vids], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    writer = None
    ow = oh = None
    total = 0
    t_start = time.perf_counter()

    for vi, path in enumerate(vids):
        name = os.path.basename(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f'[skip] cannot open {name}'); continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or args.target_fps
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(round(src_fps / args.target_fps)))
        print(f'[clip {vi + 1}/{len(vids)}] {name}  src_fps={src_fps:.0f} '
              f'frames={nframes} stride={stride}', flush=True)

        idx = 0
        kept = 0
        clip_t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            t0 = time.perf_counter()
            mask = seg_engine_batch(engine, [frame], args.seg_size, device, args.patch)[0]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fps = 1.0 / max(time.perf_counter() - t0, 1e-9)

            vis = overlay_segmentation(frame, mask, args.needle_class,
                                       args.thread_class, args.alpha)
            if ow is None:
                ow = args.out_width
                oh = int(round(frame.shape[0] * ow / frame.shape[1]))
                oh -= oh % 2
            vis = cv2.resize(vis, (ow, oh))
            draw_hud(vis, [f'{name}  [{vi + 1}/{len(vids)}]',
                           f'frame {kept}   seg {fps:4.1f} FPS'])
            if writer is None:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.target_fps, (ow, oh))
                if not writer.isOpened():
                    raise SystemExit(f'cannot open writer: {args.output}')
            writer.write(vis)
            idx += 1
            kept += 1
            total += 1
            if kept % 100 == 0:
                print(f'    {kept} frames written ...', flush=True)
        cap.release()
        dt = time.perf_counter() - clip_t0
        print(f'[clip done] {name}: {kept} frames in {dt:.1f}s '
              f'({kept / max(dt, 1e-9):.1f} fps)', flush=True)

    if writer is not None:
        writer.release()
    dt = time.perf_counter() - t_start
    print(f'[DONE] {total} frames from {len(vids)} clips in {dt / 60:.1f} min '
          f'-> {args.output}', flush=True)


if __name__ == '__main__':
    main()
