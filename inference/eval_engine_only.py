"""Engine-only evaluation: segmentation + keypoint + 3D + tip/tail + pose-drop metrics.

Source-free counterpart of eval_pose_val_v3_accel.py: it loads a pre-exported
segmentation ENGINE (TorchScript/TensorRT) — so it needs NO `test.py` / network
source — replays the stored stereo sequences, and saves ONE consolidated metrics
report covering everything we report in the paper:

  1. Segmentation   — per-class IoU + mIoU (and foreground-only mIoU), pixel acc,
                      evaluated on the LEFT view against the GT masks.
  2. Keypoints 2D   — per-keypoint and overall pixel error + PCK@<px>, on frames
                      that carry a GT keypoint sidecar.
  3. Keypoints 3D   — per-keypoint and overall metric error (mean/median/RMSE) +
                      PCK@<mm>, plus tip-specific and tail-specific 3D error.
  4. Tip/tail       — tip<->tail CLASSIFICATION error rate (how often the ordered
                      tip/tail ends up reversed vs. GT). 2D/3D keypoint errors are
                      reported BOTH raw and order-aligned so localization accuracy
                      is separated from this classification error.
  5. Pose coverage  — detection/drop rate over the whole video AND pose-success
                      rate restricted to GT (needle-present) frames = "漏帧".
  + reprojection error (3D->2D, both views; needs no GT) and FPS.

GT layout (auto-discovered under <root>/<dataset>/):
    images/<key>/.../<stem>.jpg               left frames    (from meta.json)
    stereo_right/<key>/<stem>.jpg             right frames
    masks/<key>/.../<stem>.png                GT seg (0=bg,1=needle,2=thread,3=holder)
    keypoints/<key>/.../<stem>.json           GT keypoints (needle.keypoints[*].x,y,xyz_mm)

Run (one consolidated dump over the val split):
    python inference/eval_engine_only.py \
      --engine weights/seg_engine_640.ts \
      --calib  calib/needle_calib.json \
      --needle-model calib/needle_model.json \
      --root <DATA_ROOT> \
      --val-split <DATA_ROOT>/combined/splits/r100/val.txt \
      --out-dir results/eval --seg-size 640 --num-keypoints 5 --no-video
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
import torch
from PIL import Image

import infer_accel
from infer_engine_only import (
    StereoSource, seg_engine_batch, PoseKalman, load_nk,
    overlay_segmentation, hstack_same_h, draw_pose_axes, draw_reproj,
)

SEG_CLASS_NAMES = ['background', 'needle', 'thread', 'holder']


# --------------------------------------------------------------------------- #
#  GT discovery                                                                #
# --------------------------------------------------------------------------- #
def parse_val_keys(val_split, root):
    """From a combined val.txt return ordered (dataset, key) pairs that have a
    stereo_right/ dir, plus the set of val stems per key."""
    pairs, val_stems, seen = [], {}, set()
    for line in Path(val_split).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        img = line.split('\t')[0].split(' ')[0]
        parts = img.split('/')
        if len(parts) < 4 or parts[1] != 'images':
            continue
        dataset, key = parts[0], parts[2]
        stem = os.path.splitext(parts[-1])[0]
        val_stems.setdefault((dataset, key), set()).add(stem)
        if (dataset, key) in seen:
            continue
        if (Path(root) / dataset / 'stereo_right').is_dir():
            pairs.append((dataset, key))
            seen.add((dataset, key))
    return pairs, val_stems


def index_sidecars(root, dataset, key, subdir, ext):
    """stem -> path for GT sidecars under <root>/<dataset>/<subdir>/<key>/**.<ext>."""
    d = Path(root) / dataset / subdir / key
    if not d.is_dir():
        return {}
    return {p.stem: p for p in d.rglob(f'*.{ext}')}


# --------------------------------------------------------------------------- #
#  Metric helpers                                                              #
# --------------------------------------------------------------------------- #
def add_confusion(hist, pred, gt, nclass, ignore=255):
    valid = (gt >= 0) & (gt < nclass)
    idx = nclass * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    hist += np.bincount(idx, minlength=nclass ** 2).reshape(nclass, nclass)
    return hist


def reprojection_error(out, K1, D1, K2, D2, rvecR, tvecR):
    """Mean reprojection error (px) of triangulated 3D keypoints back into both
    views vs. the detected 2D keypoints. No GT needed."""
    if out is None:
        return None
    errs = []
    for i, q in enumerate(out['xyz_mm']):
        if q is None or not np.all(np.isfinite(q)):
            continue
        P = np.asarray(q, float).reshape(1, 1, 3)
        prL = cv2.projectPoints(P, np.zeros(3), np.zeros(3), np.asarray(K1, float), np.asarray(D1, float))[0].ravel()
        prR = cv2.projectPoints(P, rvecR, tvecR, np.asarray(K2, float), np.asarray(D2, float))[0].ravel()
        if np.all(np.isfinite(prL)):
            errs.append(float(np.hypot(prL[0] - out['left'][i][0], prL[1] - out['left'][i][1])))
        if np.all(np.isfinite(prR)):
            errs.append(float(np.hypot(prR[0] - out['right'][i][0], prR[1] - out['right'][i][1])))
    return float(np.mean(errs)) if errs else None


def _np(x):
    return np.asarray(x, float)


# --------------------------------------------------------------------------- #
#  Per-key evaluation                                                          #
# --------------------------------------------------------------------------- #
def run_key(dataset, key, engine, nk, calib, rvecR, tvecR, kp_names, args, device, use_amp):
    src = StereoSource(SimpleNamespace(
        dataset=dataset, key=key, root=Path(args.root), limit=args.limit,
        capture=None, left=None, right=None, layout='sbs'))
    if not args.no_async:
        src = infer_accel.PrefetchReader(src)
    mask_idx = index_sidecars(args.root, dataset, key, 'masks', 'png')
    kp_idx = index_sidecars(args.root, dataset, key, args.gt_subdir, 'json') if args.gt_subdir else {}
    val_stems = args._val_stems.get((dataset, key))
    K = args.num_keypoints
    nclass = len(args.seg_names)

    pk = None if args.no_smooth else PoseKalman()
    out_path = os.path.join(args.out_dir, f'{dataset}__{key}.mp4')
    writer = None

    hist = np.zeros((nclass, nclass), dtype=np.int64)
    n_seg = 0
    reproj_all, fps_all = [], []
    # keypoint accumulators (raw = predicted order, aln = tip/tail aligned)
    px_raw = [[] for _ in range(K)]; px_aln = [[] for _ in range(K)]
    mm_raw = [[] for _ in range(K)]; mm_aln = [[] for _ in range(K)]
    n_frames = n_det = n_valid = n_gt = n_gt_pose = 0
    tt_eval = tt_swap = 0
    reproj_valid = []
    fps = 0.0
    gate = args.reproj_gate_px

    def in_val(stem):
        return val_stems is None or stem in val_stems

    while True:
        t0 = time.perf_counter()
        L, R, stem = src.read()
        if L is None:
            break
        n_frames += 1
        ml, mr = seg_engine_batch(engine, [L, R], args.seg_size, device, args.patch)
        needleL = ml == args.needle_class
        needleR = mr == args.needle_class
        threadL = ml == args.thread_class

        # ---- (1) segmentation IoU on LEFT view (val-only frames with GT mask) ----
        if stem in mask_idx and in_val(stem):
            gt = np.array(Image.open(mask_idx[stem]))
            if gt.ndim == 3:
                gt = gt[..., 0]
            if gt.shape == ml.shape:
                add_confusion(hist, ml, gt.astype(np.int64), nclass)
                n_seg += 1

        # ---- pose / keypoints ----
        out = None
        if needleL.sum() >= 20 and needleR.sum() >= 20:
            try:
                out, _ = nk.process_frame(needleL, needleR, threadL, calib, K,
                                          model_radius=args.model_radius)
            except Exception:
                out = None
        if pk is not None and out is not None:
            ts, rs = pk.update(out['pose']['t'], out['pose']['rvec'])
            out['pose']['t'] = list(map(float, ts))
            out['pose']['rvec'] = list(map(float, rs))
            out['pose']['R'] = cv2.Rodrigues(_np(rs))[0].tolist()
        elif pk is not None:
            pk.coast()

        dt = time.perf_counter() - t0
        fps = (0.9 * fps + 0.1 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)
        fps_all.append(1.0 / max(dt, 1e-6))
        if out is not None:
            n_det += 1
        rep = reprojection_error(out, calib['K1'], calib['D1'], calib['K2'], calib['D2'], rvecR, tvecR)
        if rep is not None:
            reproj_all.append(rep)
        # a pose is VALID only if it triangulated and reprojects sanely; otherwise
        # it is a geometrically-invalid output -> treated as a drop (漏帧).
        valid = (out is not None) and (rep is not None) and (gate <= 0 or rep <= gate)
        if valid:
            n_valid += 1
            reproj_valid.append(rep)

        # ---- (2,3,4) keypoint / 3D / tip-tail metrics on GT frames ----
        use_gt = (stem in kp_idx) and in_val(stem)
        if use_gt:
            nd = json.loads(Path(kp_idx[stem]).read_text(encoding='utf-8')).get('needle')
            gt_kps = nd['keypoints'] if nd else None
            if gt_kps and len(gt_kps) >= K:
                n_gt += 1
                if valid:
                    n_gt_pose += 1
                    gxy = _np([[g['x'], g['y']] for g in gt_kps[:K]])
                    pxy = _np(out['left'][:K])
                    # tip/tail classification: does reversing match GT better?
                    direct = np.hypot(*(pxy[0] - gxy[0])) + np.hypot(*(pxy[-1] - gxy[-1]))
                    swap = np.hypot(*(pxy[0] - gxy[-1])) + np.hypot(*(pxy[-1] - gxy[0]))
                    if nd.get('tip_tail_known', True):
                        tt_eval += 1
                        if swap < direct:
                            tt_swap += 1
                    order = list(range(K))[::-1] if swap < direct else list(range(K))
                    for i in range(K):
                        j = order[i]
                        px = float(np.hypot(*(pxy[i] - gxy[i])))
                        px_raw[i].append(px)
                        px_aln[i].append(float(np.hypot(*(pxy[j] - gxy[i]))))
                        gmm = gt_kps[i].get('xyz_mm')
                        if gmm is not None:
                            if out['xyz_mm'][i] is not None and np.all(np.isfinite(out['xyz_mm'][i])):
                                mm_raw[i].append(float(np.linalg.norm(_np(out['xyz_mm'][i]) - _np(gmm))))
                            if out['xyz_mm'][j] is not None and np.all(np.isfinite(out['xyz_mm'][j])):
                                mm_aln[i].append(float(np.linalg.norm(_np(out['xyz_mm'][j]) - _np(gmm))))

        # ---- optional video ----
        if not args.no_video:
            visL = overlay_segmentation(L, ml, args.needle_class, args.thread_class)
            visR = overlay_segmentation(R, mr, args.needle_class, args.thread_class)
            if out is not None:
                nk.draw_debug(visL, out['left'], kp_names, out['visible'], tag=None)
                nk.draw_debug(visR, out['right'], kp_names, out['visible'], tag=None)
                draw_pose_axes(visL, out['pose']['R'], out['pose']['t'], calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                draw_pose_axes(visR, out['pose']['R'], out['pose']['t'], calib['K2'], calib['D2'], rvecR, tvecR)
            canvas = hstack_same_h(visL, visR)
            if writer is None:
                h, w = canvas.shape[:2]
                writer = infer_accel.AsyncVideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                      float(args.fps_out), (w, h))
            writer.write(canvas)

    if writer is not None:
        writer.release()
    src.release()
    return dict(dataset=dataset, key=key, hist=hist, n_seg=n_seg,
                n_frames=n_frames, n_det=n_det, n_valid=n_valid,
                n_gt=n_gt, n_gt_pose=n_gt_pose,
                tt_eval=tt_eval, tt_swap=tt_swap,
                reproj_all=reproj_all, reproj_valid=reproj_valid, fps_all=fps_all,
                px_raw=px_raw, px_aln=px_aln, mm_raw=mm_raw, mm_aln=mm_aln)


# --------------------------------------------------------------------------- #
#  Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def seg_scores(hist, names):
    h = hist.astype(np.float64)
    inter = np.diag(h)
    union = h.sum(1) + h.sum(0) - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter / union
    miou = float(np.nanmean(iou))
    miou_fg = float(np.nanmean(iou[1:])) if len(iou) > 1 else float('nan')
    pix_acc = float(inter.sum() / h.sum()) if h.sum() else float('nan')
    return {
        'per_class_iou': {names[i]: (None if union[i] == 0 else float(iou[i])) for i in range(len(names))},
        'mIoU': miou, 'mIoU_fg': miou_fg, 'pixel_acc': pix_acc,
    }


def cat(list_of_lists):
    flat = [v for sub in list_of_lists for v in sub]
    return np.asarray(flat, float)


def summarize(stats, K, kp_names, args):
    nclass = len(args.seg_names)
    hist = np.sum([s['hist'] for s in stats], axis=0) if stats else np.zeros((nclass, nclass), np.int64)
    px_raw = [cat([s['px_raw'][i] for s in stats]) for i in range(K)]
    px_aln = [cat([s['px_aln'][i] for s in stats]) for i in range(K)]
    mm_raw = [cat([s['mm_raw'][i] for s in stats]) for i in range(K)]
    mm_aln = [cat([s['mm_aln'][i] for s in stats]) for i in range(K)]
    all_px_raw = np.concatenate(px_raw) if any(len(a) for a in px_raw) else np.array([])
    all_px_aln = np.concatenate(px_aln) if any(len(a) for a in px_aln) else np.array([])
    all_mm_raw = np.concatenate(mm_raw) if any(len(a) for a in mm_raw) else np.array([])
    all_mm_aln = np.concatenate(mm_aln) if any(len(a) for a in mm_aln) else np.array([])
    rep = cat([s['reproj_valid'] for s in stats])      # gated: valid poses only
    rep_raw = cat([s['reproj_all'] for s in stats])    # ungated (incl. invalid poses)
    fpsv = cat([s['fps_all'] for s in stats])

    def stat(a):
        return {'mean': float(a.mean()), 'median': float(np.median(a)),
                'rmse': float(np.sqrt((a ** 2).mean()))} if len(a) else {'mean': None, 'median': None, 'rmse': None}

    n_frames = sum(s['n_frames'] for s in stats)
    n_det = sum(s['n_det'] for s in stats)
    n_valid = sum(s['n_valid'] for s in stats)
    n_gt = sum(s['n_gt'] for s in stats)
    n_gt_pose = sum(s['n_gt_pose'] for s in stats)
    tt_eval = sum(s['tt_eval'] for s in stats)
    tt_swap = sum(s['tt_swap'] for s in stats)

    return {
        'segmentation': {**seg_scores(hist, args.seg_names),
                         'n_frames_scored': int(sum(s['n_seg'] for s in stats))},
        'keypoints_2d_px': {
            'overall_raw': stat(all_px_raw), 'overall_aligned': stat(all_px_aln),
            f'PCK@{args.pck_px:g}px_raw': (float((all_px_raw <= args.pck_px).mean() * 100) if len(all_px_raw) else None),
            f'PCK@{args.pck_px:g}px_aligned': (float((all_px_aln <= args.pck_px).mean() * 100) if len(all_px_aln) else None),
            'per_kp_raw': {kp_names[i]: (float(px_raw[i].mean()) if len(px_raw[i]) else None) for i in range(K)},
            'per_kp_aligned': {kp_names[i]: (float(px_aln[i].mean()) if len(px_aln[i]) else None) for i in range(K)},
        },
        'keypoints_3d_mm': {
            'overall_raw': stat(all_mm_raw), 'overall_aligned': stat(all_mm_aln),
            f'PCK@{args.pck_mm:g}mm_aligned': (float((all_mm_aln <= args.pck_mm).mean() * 100) if len(all_mm_aln) else None),
            'tip_mm_aligned': (float(mm_aln[0].mean()) if len(mm_aln[0]) else None),
            'tail_mm_aligned': (float(mm_aln[-1].mean()) if len(mm_aln[-1]) else None),
            'per_kp_aligned': {kp_names[i]: (float(mm_aln[i].mean()) if len(mm_aln[i]) else None) for i in range(K)},
        },
        'tip_tail': {
            'n_eval': int(tt_eval), 'n_swapped': int(tt_swap),
            'swap_error_rate_pct': (float(100.0 * tt_swap / tt_eval) if tt_eval else None),
        },
        'pose_coverage': {
            'n_frames': int(n_frames), 'n_detected': int(n_det), 'n_valid': int(n_valid),
            'reproj_gate_px': args.reproj_gate_px,
            'detect_rate_pct': (float(100.0 * n_det / n_frames) if n_frames else None),
            'valid_rate_pct': (float(100.0 * n_valid / n_frames) if n_frames else None),
            'drop_rate_pct': (float(100.0 * (n_frames - n_valid) / n_frames) if n_frames else None),
            'n_gt_frames': int(n_gt), 'n_gt_pose_ok': int(n_gt_pose),
            'gt_pose_success_pct': (float(100.0 * n_gt_pose / n_gt) if n_gt else None),
            'gt_pose_drop_pct': (float(100.0 * (n_gt - n_gt_pose) / n_gt) if n_gt else None),
        },
        'reprojection_px': stat(rep),
        'reprojection_px_ungated': stat(rep_raw),
        'fps': {'mean': float(fpsv.mean()) if len(fpsv) else 0.0,
                'median': float(np.median(fpsv)) if len(fpsv) else 0.0},
    }


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', required=True, help='pre-exported TorchScript/TensorRT seg engine (.ts)')
    p.add_argument('--calib', required=True)
    p.add_argument('--needle-model', default=None, help='needle_model.json {"radius_mm": ...}')
    p.add_argument('--model-radius', type=float, default=None)
    p.add_argument('--root', required=True)
    p.add_argument('--val-split', default=None, help='combined val.txt (keys auto-selected)')
    p.add_argument('--datasets', nargs='*', default=None, help='explicit dataset list (all keys)')
    p.add_argument('--out-dir', default='results/eval')
    p.add_argument('--num-keypoints', type=int, default=5)
    p.add_argument('--needle-class', type=int, default=1)
    p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--seg-names', nargs='*', default=SEG_CLASS_NAMES, help='class names (index order)')
    p.add_argument('--seg-size', type=int, default=640, help='MUST match the engine export --seg-size')
    p.add_argument('--patch', type=int, default=14)
    p.add_argument('--gt-subdir', default='keypoints', help='GT keypoint sidecar dir ("" to disable)')
    p.add_argument('--pck-px', type=float, default=10.0)
    p.add_argument('--pck-mm', type=float, default=2.0)
    p.add_argument('--reproj-gate-px', type=float, default=50.0,
                   help='a pose whose reprojection error exceeds this is geometrically invalid '
                        '-> counted as a DROP and excluded from kp/3D metrics (0=disable gate)')
    p.add_argument('--val-only', action='store_true',
                   help='restrict GT (seg/kp) metrics to the val stems in --val-split')
    p.add_argument('--limit', type=int, default=0, help='cap frames per key (0=all)')
    p.add_argument('--fps-out', type=float, default=20.0)
    p.add_argument('--sam2-tools', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'tools'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-async', action='store_true')
    p.add_argument('--no-video', action='store_true', help='numbers-only (fastest)')
    args = p.parse_args()

    if args.needle_model and os.path.isfile(args.needle_model):
        args.model_radius = float(json.loads(open(args.needle_model, encoding='utf-8').read())['radius_mm'])
    if args.model_radius:
        print(f'[eval] model-based pose: fixed needle radius = {args.model_radius:.3f} mm')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    engine = infer_accel.load_seg_engine(args.engine, device)
    print(f'[eval] segmentation engine = {args.engine}  (seg-size {args.seg_size})')
    nk = load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    rvecR = cv2.Rodrigues(_np(calib['R']))[0].ravel()
    tvecR = _np(calib['t']).ravel()
    kp_names = (['tip'] + [f'k{i}' for i in range(1, args.num_keypoints - 1)] + ['tail'])

    args._val_stems = {}
    if args.datasets:
        keys = []
        for d in args.datasets:
            meta = json.loads((Path(args.root) / d / 'meta.json').read_text(encoding='utf-8'))
            keys += [(d, k) for k in meta['videos'].keys()]
    else:
        assert args.val_split, 'provide --val-split or --datasets'
        keys, vs = parse_val_keys(args.val_split, args.root)
        if args.val_only:
            args._val_stems = vs
    print(f'[eval] {len(keys)} stereo video(s): ' + ', '.join(f'{d}/{k}' for d, k in keys))

    stats, t0 = [], time.perf_counter()
    for d, k in keys:
        print(f'[eval] >>> {d}/{k} ...', flush=True)
        s = run_key(d, k, engine, nk, calib, rvecR, tvecR, kp_names, args, device, use_amp)
        print(f'    frames={s["n_frames"]} det={s["n_det"]} seg_scored={s["n_seg"]} '
              f'gt_kp={s["n_gt"]} tt_swap={s["tt_swap"]}/{s["tt_eval"]}', flush=True)
        stats.append(s)

    summary = summarize(stats, args.num_keypoints, kp_names, args)
    summary['meta'] = {'engine': args.engine, 'seg_size': args.seg_size,
                       'num_keypoints': args.num_keypoints, 'val_only': bool(args.val_only),
                       'model_radius_mm': args.model_radius, 'wall_seconds': time.perf_counter() - t0,
                       'n_videos': len(stats)}
    summary['per_key'] = [{
        'dataset': s['dataset'], 'key': s['key'], 'frames': s['n_frames'],
        'detected': s['n_det'], 'seg_scored': s['n_seg'], 'gt_kp': s['n_gt'],
        'tt_swap': s['tt_swap'], 'tt_eval': s['tt_eval'],
        'seg': seg_scores(s['hist'], args.seg_names) if s['n_seg'] else None,
    } for s in stats]

    out_json = os.path.join(args.out_dir, 'metrics.json')
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _print_report(summary, args)
    print(f'\n[eval] full metrics -> {out_json}')


def _print_report(s, args):
    seg = s['segmentation']; k2 = s['keypoints_2d_px']; k3 = s['keypoints_3d_mm']
    tt = s['tip_tail']; pc = s['pose_coverage']
    f = lambda x: 'n/a' if x is None else f'{x:.3f}'
    print('\n==================== EVAL SUMMARY ====================')
    print(f'videos={s["meta"]["n_videos"]}  frames={pc["n_frames"]}  fps_med={s["fps"]["median"]:.1f}')
    print('-- Segmentation (left view) --')
    print(f'  mIoU={f(seg["mIoU"])}  mIoU_fg={f(seg["mIoU_fg"])}  pixAcc={f(seg["pixel_acc"])}  (frames={seg["n_frames_scored"]})')
    for n, v in seg['per_class_iou'].items():
        print(f'    {n:<12} IoU={f(v)}')
    print('-- Keypoints 2D (px) --')
    print(f'  overall raw mean={f(k2["overall_raw"]["mean"])}  aligned mean={f(k2["overall_aligned"]["mean"])}  '
          f'PCK@{args.pck_px:g}px(aln)={f(k2[f"PCK@{args.pck_px:g}px_aligned"])}%')
    print('-- Keypoints 3D (mm) --')
    print(f'  aligned mean={f(k3["overall_aligned"]["mean"])}  median={f(k3["overall_aligned"]["median"])}  '
          f'rmse={f(k3["overall_aligned"]["rmse"])}  tip={f(k3["tip_mm_aligned"])}  tail={f(k3["tail_mm_aligned"])}  '
          f'PCK@{args.pck_mm:g}mm={f(k3[f"PCK@{args.pck_mm:g}mm_aligned"])}%')
    print('-- Tip/Tail classification --')
    print(f'  swap_error_rate={f(tt["swap_error_rate_pct"])}%  ({tt["n_swapped"]}/{tt["n_eval"]})')
    print('-- Pose coverage / drop --')
    print(f'  detect(raw)={f(pc["detect_rate_pct"])}%  valid(gated@{pc["reproj_gate_px"]:g}px)={f(pc["valid_rate_pct"])}%  '
          f'drop={f(pc["drop_rate_pct"])}%')
    print(f'  GT-frame pose success={f(pc["gt_pose_success_pct"])}%  drop(漏帧)={f(pc["gt_pose_drop_pct"])}%')
    print(f'  reproj px (gated) mean={f(s["reprojection_px"]["mean"])} median={f(s["reprojection_px"]["median"])}'
          f'  | ungated median={f(s["reprojection_px_ungated"]["median"])}')


if __name__ == '__main__':
    main()
