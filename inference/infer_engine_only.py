"""Source-free stereo needle keypoint + 6-DoF pose inference (engine-only).

This is the ENCAPSULATED deployment entry point: it runs the SAME pipeline as
realtime_stereo_keypoints_v3_accel.py but loads a pre-exported segmentation
ENGINE (TorchScript / TensorRT, from export_seg_engine.py) instead of building
the network from source. It therefore does NOT import `test.py` or the `model/`
package, so a public release can ship just:

    infer_engine_only.py  +  infer_accel.py  +  needle_keypoints_v2.py
    +  <engine>.ts  +  needle_calib.json  +  needle_model.json

…and never expose the model architecture / training source or the raw best.pth.

The engine is FIXED-SHAPE: run with the SAME --seg-size used at export, and a
source whose per-eye resolution matches the export (--src-h/--src-w). Mismatch
errors out — re-export, or use the full-source v3 script with --compile instead.

Input sources (same as the v3 script):
  --root R --dataset D --key K   replay a stored stereo sequence
  --left A --right B             two video files or camera indices (e.g. 0 1)
  --capture 0 --layout sbs|tb    one device carrying side-by-side / top-bottom stereo

Example:
  python infer_engine_only.py --engine exp/combined_r100_base/seg_trt_s640.ts \
    --calib  /root/autodl-tmp/code/SAM2-Plus/tools/needle_calib.json \
    --needle-model /root/autodl-tmp/code/SAM2-Plus/tools/needle_model.json \
    --left left.mp4 --right right.mp4 --seg-size 640 --num-keypoints 5 \
    --save-video out.mp4 --save-results out.jsonl
"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import infer_accel

_NORM = T.Compose([T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_nk(path):
    sys.path.insert(0, path)
    import needle_keypoints_v2 as nk     # model-based (fixed-radius) pose registration
    return nk


# ----------------------------------------------------------------- source
class StereoSource:
    def __init__(self, args):
        self.mode = None
        if args.dataset:
            self.ds = args.root.resolve() / args.dataset
            self.key = args.key
            meta = json.loads((self.ds / 'meta.json').read_text(encoding='utf-8'))
            self.recs = sorted(meta['videos'][args.key], key=lambda r: r['ordinal'])
            if args.limit:
                self.recs = self.recs[:args.limit]
            self.i = 0
            self.mode = 'dataset'
        elif args.capture is not None:
            self.cap = cv2.VideoCapture(int(args.capture))
            self.layout = args.layout
            self.mode = 'split'
        else:
            def _open(s):
                return cv2.VideoCapture(int(s) if str(s).isdigit() else s)
            self.capL = _open(args.left)
            self.capR = _open(args.right)
            self.mode = 'pair'

    def read(self):
        if self.mode == 'dataset':
            if self.i >= len(self.recs):
                return None, None, None
            r = self.recs[self.i]; self.i += 1
            stem = os.path.splitext(os.path.basename(r['image']))[0]
            L = cv2.imread(str(self.ds / r['image']))
            R = cv2.imread(str(self.ds / 'stereo_right' / self.key / f'{stem}.jpg'))
            if L is None or R is None:
                return self.read()
            return L, R, stem
        if self.mode == 'split':
            ok, fr = self.cap.read()
            if not ok:
                return None, None, None
            h, w = fr.shape[:2]
            if self.layout == 'sbs':
                return fr[:, :w // 2], fr[:, w // 2:], None
            return fr[:h // 2], fr[h // 2:], None
        okL, L = self.capL.read(); okR, R = self.capR.read()
        if not okL or not okR:
            return None, None, None
        return L, R, None

    def release(self):
        for c in ('cap', 'capL', 'capR'):
            if hasattr(self, c):
                getattr(self, c).release()


# ----------------------------------------------------------------- seg via engine
def seg_engine_batch(engine, bgrs, seg_size, device, patch=14, prof=None):
    """Preprocess BOTH eyes exactly like the full pipeline's seg_masks_batch, run
    one engine forward, return per-eye uint8 label maps at the source resolution.
    The engine forward shape must equal what export_seg_engine.py built.

    If `prof` (a dict) is given, accumulates wall-clock seconds for the 'pre',
    'fwd' and 'post' sub-stages (CUDA-synchronized so the timings are real)."""
    def _sync():
        if prof is not None and device.type == 'cuda':
            torch.cuda.synchronize()

    H, W = bgrs[0].shape[:2]
    if seg_size and max(H, W) > seg_size:
        s = seg_size / float(max(H, W)); sh, sw = int(round(H * s)), int(round(W * s))
    else:
        sh, sw = H, W
    new_h = max(patch, int(round(sh / patch)) * patch)
    new_w = max(patch, int(round(sw / patch)) * patch)
    _sync(); _t = time.perf_counter()
    ts = [_NORM(Image.fromarray(cv2.cvtColor(cv2.resize(b, (sw, sh)), cv2.COLOR_BGR2RGB)))
          for b in bgrs]
    x = torch.stack(ts).to(device)
    if (new_h, new_w) != (sh, sw):
        x = F.interpolate(x, (new_h, new_w), mode='bilinear', align_corners=True)
    _sync()
    if prof is not None:
        prof['pre'] = prof.get('pre', 0.0) + (time.perf_counter() - _t); _t = time.perf_counter()
    with torch.inference_mode():
        logits = engine(x)                          # _HalfInputEngine casts to fp16
        lab = logits.argmax(1)
        _sync()
        if prof is not None:
            prof['fwd'] = prof.get('fwd', 0.0) + (time.perf_counter() - _t); _t = time.perf_counter()
        if lab.shape[-2:] != (H, W):
            lab = F.interpolate(lab[:, None].float(), (H, W), mode='nearest')[:, 0]
        preds = lab.to(torch.uint8).cpu().numpy()
    _sync()
    if prof is not None:
        prof['post'] = prof.get('post', 0.0) + (time.perf_counter() - _t)
    return [preds[i] for i in range(len(bgrs))]


# ----------------------------------------------------------------- pose smoothing
class PoseKalman:
    """Constant-velocity Kalman on [tx,ty,tz, rx,ry,rz]; coasts on missing frames."""
    def __init__(self, meas_noise=1.0, proc_noise=1e-2):
        kf = cv2.KalmanFilter(12, 6)
        Ft = np.eye(12, dtype=np.float32)
        for i in range(6):
            Ft[i, i + 6] = 1.0
        kf.transitionMatrix = Ft
        Hm = np.zeros((6, 12), np.float32); Hm[:6, :6] = np.eye(6)
        kf.measurementMatrix = Hm
        kf.processNoiseCov = np.eye(12, dtype=np.float32) * proc_noise
        kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * meas_noise
        self.kf = kf; self.inited = False; self.prev_rvec = None

    def _align(self, rvec):
        rvec = np.asarray(rvec, float)
        if self.prev_rvec is not None and np.dot(rvec, self.prev_rvec) < 0:
            rvec = -rvec
        self.prev_rvec = rvec
        return rvec

    def update(self, t, rvec):
        rvec = self._align(rvec)
        m = np.asarray(list(t) + list(rvec), np.float32).reshape(6, 1)
        if not self.inited:
            self.kf.statePost = np.vstack([m, np.zeros((6, 1), np.float32)])
            self.inited = True
            return np.asarray(t, float), np.asarray(rvec, float)
        self.kf.predict()
        est = self.kf.correct(m)[:6, 0]
        return est[:3], est[3:6]

    def coast(self):
        if not self.inited:
            return None
        est = self.kf.predict()[:6, 0]
        return est[:3], est[3:6]


# ----------------------------------------------------------------- drawing
def rot_to_euler_deg(R):
    R = np.asarray(R, float)
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2]); y = np.arctan2(-R[2, 0], sy); z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1]); y = np.arctan2(-R[2, 0], sy); z = 0.0
    return np.degrees([x, y, z])


def overlay_segmentation(bgr, mask, needle_class=1, thread_class=2, alpha=0.38):
    color = bgr.copy()
    color[mask == needle_class] = (0, 0, 255)
    color[mask == thread_class] = (0, 220, 0)
    color[mask == 3] = (255, 120, 0)
    return cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0)


def hstack_same_h(a, b):
    if a.shape[0] != b.shape[0]:
        s = a.shape[0] / b.shape[0]
        b = cv2.resize(b, (int(b.shape[1] * s), a.shape[0]))
    return cv2.hconcat([a, b])


def draw_pose_axes(img, R, t, K, D, rvec_cam, tvec_cam, axis_mm=20.0):
    R = np.asarray(R, float); t = np.asarray(t, float)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        return
    pts = np.stack([t, t + axis_mm * R[:, 0], t + axis_mm * R[:, 1], t + axis_mm * R[:, 2]])
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec_cam, float),
                              np.asarray(tvec_cam, float), np.asarray(K, float), np.asarray(D, float))
    pr = pr.reshape(-1, 2)
    if not np.all(np.isfinite(pr)):
        return
    o, ax, ay, az = [tuple(np.int32(np.round(q))) for q in pr]
    cv2.line(img, o, ax, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, o, ay, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, o, az, (255, 0, 0), 2, cv2.LINE_AA)


def draw_reproj(img, xyz, K, D, rvec_cam, tvec_cam):
    pts = np.asarray([q for q in xyz if q is not None and np.all(np.isfinite(q))], float)
    if len(pts) == 0:
        return
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec_cam, float),
                              np.asarray(tvec_cam, float), np.asarray(K, float), np.asarray(D, float))
    h, w = img.shape[:2]
    for (x, y) in pr.reshape(-1, 2):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if abs(x) > 1e6 or abs(y) > 1e6:
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < -32 or xi >= w + 32 or yi < -32 or yi >= h + 32:
            continue
        cv2.circle(img, (xi, yi), 9, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', required=True, help='pre-exported TorchScript/TensorRT engine (.ts)')
    p.add_argument('--calib', required=True)
    p.add_argument('--needle-model', default=None,
                   help='needle_model.json {"radius_mm": ...} -> fixed-radius registration (recommended)')
    p.add_argument('--model-radius', type=float, default=None, help='alt to --needle-model')
    # sources
    from pathlib import Path as _P
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
                        'skipped frames (throughput knob for weak GPUs; >1 changes output). 1=every frame.')
    p.add_argument('--patch', type=int, default=14, help='backbone patch (DINOv2=14)')
    p.add_argument('--view-height', type=int, default=720, help='downscale L|R canvas (0=full)')
    p.add_argument('--sam2-tools', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'SAM2-Plus', 'tools'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-async', action='store_true')
    p.add_argument('--no-reproject', action='store_true')
    p.add_argument('--show', action='store_true')
    p.add_argument('--profile', action='store_true',
                   help='accumulate per-stage timing (read/pre/fwd/post/pose/draw) and '
                        'print the breakdown at the end; adds CUDA syncs so absolute fps drops')
    p.add_argument('--save-video', default=None)
    p.add_argument('--save-results', default=None, help='JSONL: one record per frame')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    engine = infer_accel.load_seg_engine(args.engine, device)
    print(f'[engine-only] segmentation engine = {args.engine}')
    nk = load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    if args.needle_model and os.path.isfile(args.needle_model):
        args.model_radius = float(json.loads(
            open(args.needle_model, encoding='utf-8').read())['radius_mm'])
    if args.model_radius:
        print(f'[engine-only] model-based pose: fixed needle radius = {args.model_radius:.2f} mm')

    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel()
    tvecR = np.asarray(calib['t'], float).ravel()
    kp_names = (['tip'] + [f'k{i}' for i in range(1, args.num_keypoints - 1)] + ['tail'])

    src = StereoSource(args)
    if not args.no_async:
        src = infer_accel.PrefetchReader(src)
    pk = None if args.no_smooth else PoseKalman()
    rjsonl = open(args.save_results, 'w', encoding='utf-8') if args.save_results else None
    writer = None
    WIN = 'engine-only stereo keypoints (q=quit)'
    fps = 0.0; fi = 0
    prof = {} if args.profile else None
    n_pose = 0
    last_ml = last_mr = last_out = None       # --stride: reuse on skipped frames
    print('[engine-only] started' + ('' if not args.show else ' — press q to quit'))

    while True:
        t0 = time.perf_counter()
        L, R, stem = src.read()
        if L is None:
            break
        if prof is not None:
            prof['read'] = prof.get('read', 0.0) + (time.perf_counter() - t0)
        run_seg = (last_ml is None) or (fi % max(1, args.stride) == 0)
        if run_seg:
            ml, mr = seg_engine_batch(engine, [L, R], args.seg_size, device, args.patch, prof)
            needleL = ml == args.needle_class; needleR = mr == args.needle_class
            threadL = ml == args.thread_class
            out = None
            _tp = time.perf_counter()
            if needleL.sum() >= 20 and needleR.sum() >= 20:
                try:
                    out, _ = nk.process_frame(needleL, needleR, threadL, calib, args.num_keypoints,
                                              model_radius=args.model_radius)
                except Exception:
                    out = None
            if prof is not None:
                prof['pose'] = prof.get('pose', 0.0) + (time.perf_counter() - _tp)
                n_pose += 1
            if pk is not None:
                if out is not None:
                    ts, rs = pk.update(out['pose']['t'], out['pose']['rvec'])
                    out['pose']['t'] = list(map(float, ts))
                    out['pose']['rvec'] = list(map(float, rs))
                    out['pose']['R'] = cv2.Rodrigues(np.asarray(rs, float))[0].tolist()
                else:
                    pk.coast()
            last_ml, last_mr, last_out = ml, mr, out
        else:                                  # skipped frame: reuse last seg+pose (skips the GPU forward)
            ml, mr, out = last_ml, last_mr, last_out
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
            visL = overlay_segmentation(L, ml, args.needle_class, args.thread_class)
            visR = overlay_segmentation(R, mr, args.needle_class, args.thread_class)
            if out is not None:
                nk.draw_debug(visL, out['left'], kp_names, out['visible'], tag=None)
                nk.draw_debug(visR, out['right'], kp_names, out['visible'], tag=None)
                if not args.no_reproject:
                    Ro = out['pose']['R']; to = out['pose']['t']
                    draw_pose_axes(visL, Ro, to, calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                    draw_pose_axes(visR, Ro, to, calib['K2'], calib['D2'], rvecR, tvecR)
                    draw_reproj(visL, out['xyz_mm'], calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                    draw_reproj(visR, out['xyz_mm'], calib['K2'], calib['D2'], rvecR, tvecR)
            canvas = hstack_same_h(visL, visR)   # no right info panel (text too small to read)
            cv2.putText(canvas, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            if out is not None:
                t = out['pose']['t']; eu = rot_to_euler_deg(out['pose']['R'])
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
                if prof is not None:
                    prof['draw'] = prof.get('draw', 0.0) + (time.perf_counter() - _td)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            elif prof is not None:
                prof['draw'] = prof.get('draw', 0.0) + (time.perf_counter() - _td)
        fi += 1

    src.release()
    if writer is not None:
        writer.release()
    if rjsonl is not None:
        rjsonl.close()
        print(f'[engine-only] results -> {args.save_results}')
    cv2.destroyAllWindows()
    print(f'[engine-only] done — {fi} frames, ~{fps:.1f} fps')
    if prof is not None and fi > 0:
        order = [('read', 'frame read (async q)'), ('pre', 'preprocess (PIL/resize/H2D)'),
                 ('fwd', 'engine forward'), ('post', 'argmax/upscale/D2H'),
                 ('pose', 'pose registration (CPU)'), ('draw', 'overlay/draw/imshow')]
        total = sum(prof.get(k, 0.0) for k, _ in order)
        print('[profile] per-frame breakdown (avg over '
              f'{fi} frames; pose avg over {max(n_pose,1)} pose-frames):')
        for k, label in order:
            tot = prof.get(k, 0.0)
            n = n_pose if k == 'pose' else fi
            ms = 1000.0 * tot / max(n, 1)
            pct = 100.0 * tot / total if total else 0.0
            print(f'    {label:<28} {ms:7.2f} ms/frame   {pct:5.1f}%')
        print(f'    {"SUM (serial, excl. overlap)":<28} {1000.0*total/fi:7.2f} ms/frame')


if __name__ == '__main__':
    main()
