"""Stereo needle keypoint + 6-DoF pose inference using a RELEASED TorchScript
segmentation model. Loads `seg.ts.pt` via torch.jit.load — the model architecture
source is NOT required, so this tool can be shipped publicly without exposing the
model internals. Only the (publishable) geometric keypoint code is imported.

Inputs (one of):
  --left A --right B          two video files / camera indices
  --capture 0 --layout sbs    one capture-card device (side-by-side / top-bottom)
  --root R --dataset D --key K  replay a stored dataset sequence (+ optional metrics)

Outputs: live/recorded LEFT|RIGHT overlay (keypoints + pose axes + reprojection),
FPS, and per-frame results JSONL (keypoint 3D coords + pose {R,t,rvec}).

Run:
  python tools/infer_ts_stereo_keypoints.py --ts-model seg.ts.pt --calib tools/needle_calib.json \
      --left left.mp4 --right right.mp4 --num-keypoints 5 \
      --save-video out.mp4 --save-results result.jsonl
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needle_keypoints as nk            # geometric keypoint code (publishable)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_chw(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def seg_pair(ts, L, R, S, device, use_amp):
    """Segment both eyes (batched) with the TorchScript model -> two id masks."""
    same = L.shape[:2] == R.shape[:2]
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()

    def run(imgs):
        x = torch.stack([to_chw(cv2.resize(im, (S, S))) for im in imgs]).to(device)
        with torch.inference_mode(), ctx:
            logits = ts(x)
        out = []
        for i, im in enumerate(imgs):
            h, w = im.shape[:2]
            lg = torch.nn.functional.interpolate(logits[i:i + 1], (h, w),
                                                 mode='bilinear', align_corners=True)
            out.append(lg.argmax(1)[0].to(torch.uint8).cpu().numpy())
        return out

    if same:
        a, b = run([L, R]); return a, b
    return run([L])[0], run([R])[0]


# --------------------------------------------------------- pose smoothing / draw
class PoseKalman:
    def __init__(self):
        kf = cv2.KalmanFilter(12, 6)
        Ft = np.eye(12, dtype=np.float32)
        for i in range(6):
            Ft[i, i + 6] = 1.0
        kf.transitionMatrix = Ft
        Hm = np.zeros((6, 12), np.float32); Hm[:6, :6] = np.eye(6)
        kf.measurementMatrix = Hm
        kf.processNoiseCov = np.eye(12, np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(6, np.float32)
        self.kf = kf; self.inited = False; self.prev = None

    def update(self, t, rvec):
        rvec = np.asarray(rvec, float)
        if self.prev is not None and np.dot(rvec, self.prev) < 0:
            rvec = -rvec
        self.prev = rvec
        m = np.asarray(list(t) + list(rvec), np.float32).reshape(6, 1)
        if not self.inited:
            self.kf.statePost = np.vstack([m, np.zeros((6, 1), np.float32)]); self.inited = True
            return np.asarray(t, float), rvec
        self.kf.predict(); e = self.kf.correct(m)[:6, 0]
        return e[:3], e[3:6]


def euler_deg(R):
    R = np.asarray(R, float); sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    if sy > 1e-6:
        a = [np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])]
    else:
        a = [np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0]
    return np.degrees(a)


def draw_axes(img, R, t, K, D, rvec, tvec, L=20.0):
    R = np.asarray(R, float); t = np.asarray(t, float)
    pts = np.stack([t, t + L * R[:, 0], t + L * R[:, 1], t + L * R[:, 2]])
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec, float), np.asarray(tvec, float),
                              np.asarray(K, float), np.asarray(D, float))
    o, x, y, z = [tuple(np.int32(np.round(q))) for q in pr.reshape(-1, 2)]
    cv2.line(img, o, x, (0, 0, 255), 2); cv2.line(img, o, y, (0, 255, 0), 2); cv2.line(img, o, z, (255, 0, 0), 2)


def draw_reproj(img, xyz, K, D, rvec, tvec):
    pts = np.asarray([q for q in xyz if q is not None], float)
    if len(pts) == 0:
        return
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec, float), np.asarray(tvec, float),
                              np.asarray(K, float), np.asarray(D, float))
    for (x, y) in pr.reshape(-1, 2):
        cv2.circle(img, (int(round(x)), int(round(y))), 9, (255, 255, 255), 1)


def hstack(a, b):
    if a.shape[0] != b.shape[0]:
        s = a.shape[0] / b.shape[0]; b = cv2.resize(b, (int(b.shape[1] * s), a.shape[0]))
    return cv2.hconcat([a, b])


# ------------------------------------------------------------------- source
class Source:
    def __init__(self, a):
        if a.dataset:
            self.ds = a.root / a.dataset; self.key = a.key
            meta = json.loads((self.ds / 'meta.json').read_text(encoding='utf-8'))
            self.recs = sorted(meta['videos'][a.key], key=lambda r: r['ordinal'])
            self.i = 0; self.mode = 'ds'
        elif a.capture is not None:
            self.cap = cv2.VideoCapture(int(a.capture)); self.layout = a.layout; self.mode = 'split'
        else:
            op = lambda s: cv2.VideoCapture(int(s) if str(s).isdigit() else s)
            self.cl = op(a.left); self.cr = op(a.right); self.mode = 'pair'

    def read(self):
        if self.mode == 'ds':
            if self.i >= len(self.recs):
                return None, None, None
            r = self.recs[self.i]; self.i += 1
            st = os.path.splitext(os.path.basename(r['image']))[0]
            L = cv2.imread(str(self.ds / r['image']))
            R = cv2.imread(str(self.ds / 'stereo_right' / self.key / f'{st}.jpg'))
            return (L, R, st) if (L is not None and R is not None) else self.read()
        if self.mode == 'split':
            ok, fr = self.cap.read()
            if not ok:
                return None, None, None
            h, w = fr.shape[:2]
            return (fr[:, :w // 2], fr[:, w // 2:], None) if self.layout == 'sbs' else (fr[:h // 2], fr[h // 2:], None)
        okL, L = self.cl.read(); okR, R = self.cr.read()
        return (L, R, None) if (okL and okR) else (None, None, None)


def main():
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ts-model', required=True, help='released TorchScript seg model (.ts.pt)')
    p.add_argument('--calib', required=True)
    p.add_argument('--left'); p.add_argument('--right')
    p.add_argument('--capture'); p.add_argument('--layout', choices=['sbs', 'tb'], default='sbs')
    p.add_argument('--root', type=Path); p.add_argument('--dataset'); p.add_argument('--key')
    p.add_argument('--num-keypoints', type=int, default=5)
    p.add_argument('--needle-class', type=int, default=1); p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--seg-size', type=int, default=0, help='override (default: from .ts.pt.json)')
    p.add_argument('--gt-subdir', default='keypoints'); p.add_argument('--pck-thresh', type=float, default=10.0)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-smooth', action='store_true'); p.add_argument('--no-reproject', action='store_true')
    p.add_argument('--show', action='store_true'); p.add_argument('--save-video', default=None)
    p.add_argument('--save-results', default=None); p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not a.no_amp
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    ts = torch.jit.load(a.ts_model, map_location=device).eval()
    meta = {}
    if os.path.exists(a.ts_model + '.json'):
        meta = json.loads(open(a.ts_model + '.json').read())
    S = a.seg_size or int(meta.get('seg_size', 640))
    calib = nk.load_calib(a.calib)
    rvecL = np.zeros(3); tvecL = np.zeros(3)
    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel(); tvecR = np.asarray(calib['t'], float).ravel()
    names = ['tip'] + [f'k{i}' for i in range(1, a.num_keypoints - 1)] + ['tail']
    pk = None if a.no_smooth else PoseKalman()
    src = Source(a)

    gt_on = bool(a.dataset and a.gt_subdir)
    gt_dir = (a.root.resolve() / a.dataset / a.gt_subdir / a.key) if gt_on else None
    gt_idx = {q.stem: q for q in gt_dir.rglob('*.json')} if (gt_on and gt_dir and gt_dir.is_dir()) else {}
    errs, n_eval = [], 0

    rj = open(a.save_results, 'w', encoding='utf-8') if a.save_results else None
    writer = None; fps = 0.0; fi = 0
    print(f'[infer-ts] started  seg-size={S}  amp={"on" if use_amp else "off"}')
    while True:
        t0 = time.perf_counter()
        L, R, stem = src.read()
        if L is None:
            break
        if a.limit and fi >= a.limit:
            break
        ml, mr = seg_pair(ts, L, R, S, device, use_amp)
        nL = ml == a.needle_class; nR = mr == a.needle_class
        out = None
        if nL.sum() >= 20 and nR.sum() >= 20:
            try:
                out, _ = nk.process_frame(nL, nR, ml == a.thread_class, calib, a.num_keypoints)
            except Exception:
                out = None
        if pk is not None and out is not None:
            tt, rr = pk.update(out['pose']['t'], out['pose']['rvec'])
            out['pose']['t'] = list(map(float, tt)); out['pose']['rvec'] = list(map(float, rr))
            out['pose']['R'] = cv2.Rodrigues(np.asarray(rr, float))[0].tolist()
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6) if fps else 1.0 / max(time.perf_counter() - t0, 1e-6)

        if out is not None and gt_on and stem in gt_idx:
            nd = json.loads(gt_idx[stem].read_text(encoding='utf-8')).get('needle')
            if nd and len(nd['keypoints']) >= a.num_keypoints:
                n_eval += 1
                for i in range(a.num_keypoints):
                    g = nd['keypoints'][i]
                    errs.append(((out['left'][i][0] - g['x']) ** 2 + (out['left'][i][1] - g['y']) ** 2) ** 0.5)

        if rj is not None:
            needle = None
            if out is not None:
                needle = {'keypoints': [{'name': names[i], 'x': out['left'][i][0], 'y': out['left'][i][1],
                                         'x_right': out['right'][i][0], 'y_right': out['right'][i][1],
                                         'xyz_mm': out['xyz_mm'][i], 'visible': int(out['visible'][i])}
                                        for i in range(a.num_keypoints)], 'pose': out['pose'], 'conf': out['conf']}
            rj.write(json.dumps({'frame': fi, 'stem': stem, 'fps': round(fps, 2), 'needle': needle}) + '\n')

        if a.show or a.save_video:
            vL, vR = L.copy(), R.copy()
            if out is not None:
                nk.draw_debug(vL, out['left'], names, out['visible']); nk.draw_debug(vR, out['right'], names, out['visible'])
                if not a.no_reproject:
                    Ro, to = out['pose']['R'], out['pose']['t']
                    draw_axes(vL, Ro, to, calib['K1'], calib['D1'], rvecL, tvecL)
                    draw_axes(vR, Ro, to, calib['K2'], calib['D2'], rvecR, tvecR)
                    draw_reproj(vL, out['xyz_mm'], calib['K1'], calib['D1'], rvecL, tvecL)
                    draw_reproj(vR, out['xyz_mm'], calib['K2'], calib['D2'], rvecR, tvecR)
            cv = hstack(vL, vR)
            cv2.putText(cv, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if out is not None:
                t = out['pose']['t']; eu = euler_deg(out['pose']['R'])
                cv2.putText(cv, f't=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm rot=({eu[0]:.0f},{eu[1]:.0f},{eu[2]:.0f})',
                            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            if a.save_video:
                if writer is None:
                    h, w = cv.shape[:2]
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                writer.write(cv)
            if a.show:
                cv2.imshow('keypoints (q=quit)', cv)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
        fi += 1

    if writer is not None:
        writer.release()
    if rj is not None:
        rj.close()
    cv2.destroyAllWindows()
    print(f'[infer-ts] done — {fi} frames, ~{fps:.1f} fps')
    if gt_on and n_eval:
        e = np.asarray(errs)
        print(f'[metrics] frames={n_eval}  mean px err={e.mean():.2f}  '
              f'PCK@{a.pck_thresh:.0f}px={(e <= a.pck_thresh).mean() * 100:.1f}%  median={np.median(e):.2f}')


if __name__ == '__main__':
    main()
