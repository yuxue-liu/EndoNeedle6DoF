"""Split-stage evaluation, live panels, videos, and parameter sweeps.

This script exports three synchronized outputs for each input source:

  1. segmentation video: left/right segmentation overlays
  2. keypoints video: segmentation + detected 2D keypoints
  3. pose video: segmentation + keypoints + 6-DoF pose axes/reprojection

Inputs supported:

  * dataset validation split: --root --val-split [--val-only]
  * dataset keys:             --root --datasets DATASET [...]
  * frame folders:            --image-dir LEFT_DIR --right-image-dir RIGHT_DIR
  * stereo videos/streams:    --left LEFT --right RIGHT
  * capture card:             --capture 0 --layout sbs|tb

For frame folders and datasets, masks/keypoint JSONs are auto-detected when
available. Videos and capture inputs still save all videos and report FPS,
detection, pose coverage and reprojection metrics, but GT metrics are omitted.
"""
import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import infer_accel
try:
    import torch_tensorrt  # noqa: F401
except Exception:
    torch_tensorrt = None
from infer_engine_only import (
    PoseKalman, load_nk, seg_engine_batch, overlay_segmentation, hstack_same_h,
    draw_pose_axes, draw_reproj,
)
from eval_engine_only import (
    SEG_CLASS_NAMES, parse_val_keys, index_sidecars, add_confusion,
    reprojection_error, seg_scores, summarize,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _np(x):
    return np.asarray(x, float)


def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def ensure_writer(path, frame, fps):
    h, w = frame.shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return infer_accel.AsyncVideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )


def put_text(img, text, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def append_panel(canvas, title, rows, panel_w=540):
    h, w = canvas.shape[:2]
    panel = np.full((h, panel_w, 3), (34, 38, 43), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, h - 1), (80, 88, 96), 1)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, 72), (24, 28, 32), -1)
    put_text(panel, title, (20, 50), scale=1.05, color=(255, 255, 255), thick=3)
    y = 112
    for label, value, color in rows:
        if y > h - 28:
            break
        put_text(panel, label, (20, y), scale=0.70, color=(168, 178, 188), thick=1)
        put_text(panel, value, (228, y), scale=0.86, color=color, thick=2)
        y += 42
    return np.concatenate([canvas, panel], axis=1)


def gpu_mem_row():
    """Real-time device memory usage row: used/total and utilization %."""
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            pct = 100.0 * used / total
            color = (110, 215, 255) if pct < 90 else (110, 130, 235)
            return ("GPU mem", f"{used / 1e9:.1f}/{total / 1e9:.0f}G  {pct:.0f}%", color)
    except Exception:
        pass
    return ("GPU mem", "n/a", (180, 180, 180))


def fmt(value, nd=2, suffix=""):
    if value is None:
        return "n/a"
    return f"{float(value):.{nd}f}{suffix}"


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def hist_miou(hist):
    total = int(hist.sum())
    if total <= 0:
        return None, None
    diag = np.diag(hist).astype(np.float64)
    union = hist.sum(0) + hist.sum(1) - diag
    valid = union > 0
    miou = float(np.mean(diag[valid] / np.maximum(union[valid], 1.0)) * 100.0) if np.any(valid) else None
    pix = float(diag.sum() / total * 100.0)
    return miou, pix


class DatasetSource:
    def __init__(self, root, dataset, key, limit=0):
        self.dataset = dataset
        self.key = key
        self.ds = Path(root) / dataset
        meta = json.loads((self.ds / "meta.json").read_text(encoding="utf-8"))
        self.recs = sorted(meta["videos"][key], key=lambda r: r["ordinal"])
        if limit:
            self.recs = self.recs[:limit]
        self.i = 0

    def read(self):
        while self.i < len(self.recs):
            r = self.recs[self.i]
            self.i += 1
            stem = os.path.splitext(os.path.basename(r["image"]))[0]
            left = cv2.imread(str(self.ds / r["image"]))
            right = cv2.imread(str(self.ds / "stereo_right" / self.key / f"{stem}.jpg"))
            if left is not None and right is not None:
                return left, right, stem
        return None, None, None

    def release(self):
        pass


class FrameDirSource:
    def __init__(self, image_dir, right_image_dir, limit=0):
        self.left_dir = Path(image_dir)
        self.right_dir = Path(right_image_dir)
        lefts = [p for p in sorted(self.left_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if limit:
            lefts = lefts[:limit]
        self.lefts = lefts
        self.right_by_stem = {p.stem: p for p in self.right_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        self.i = 0

    def read(self):
        while self.i < len(self.lefts):
            lp = self.lefts[self.i]
            self.i += 1
            rp = self.right_by_stem.get(lp.stem)
            if rp is None:
                continue
            left = cv2.imread(str(lp))
            right = cv2.imread(str(rp))
            if left is not None and right is not None:
                return left, right, lp.stem
        return None, None, None

    def release(self):
        pass


class VideoPairSource:
    def __init__(self, left, right, limit=0):
        def _open(s):
            return cv2.VideoCapture(int(s) if str(s).isdigit() else s)
        self.capL = _open(left)
        self.capR = _open(right)
        self.i = 0
        self.limit = limit

    def read(self):
        if self.limit and self.i >= self.limit:
            return None, None, None
        okL, left = self.capL.read()
        okR, right = self.capR.read()
        if not okL or not okR:
            return None, None, None
        self.i += 1
        return left, right, f"frame_{self.i:06d}"

    def release(self):
        self.capL.release()
        self.capR.release()


class CaptureSplitSource:
    def __init__(self, capture, layout="sbs", limit=0):
        self.cap = cv2.VideoCapture(int(capture))
        self.layout = layout
        self.i = 0
        self.limit = limit

    def read(self):
        if self.limit and self.i >= self.limit:
            return None, None, None
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None
        h, w = frame.shape[:2]
        self.i += 1
        if self.layout == "tb":
            return frame[:h // 2], frame[h // 2:], f"frame_{self.i:06d}"
        return frame[:, :w // 2], frame[:, w // 2:], f"frame_{self.i:06d}"

    def release(self):
        self.cap.release()


def replace_path_part(path, src, dst):
    parts = list(Path(path).parts)
    if src not in parts:
        return None
    parts[parts.index(src)] = dst
    return Path(*parts)


def collect_sidecars_from_dir(sidecar_dir, ext):
    if not sidecar_dir:
        return {}
    p = Path(sidecar_dir)
    if not p.exists():
        return {}
    return {x.stem: str(x) for x in p.rglob(f"*.{ext}") if x.is_file()}


def auto_sidecar_index(image_dir, explicit_dir, kind, ext):
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    img = Path(image_dir)
    repl = replace_path_part(img, "images", kind)
    if repl is not None:
        candidates.append(repl)
    candidates += [
        img.parent / kind,
        img.parent.parent / kind / img.name,
        img.parent.parent / kind,
    ]
    out = {}
    for c in candidates:
        out.update(collect_sidecars_from_dir(c, ext))
    return out


def load_gt_mask(path):
    gt = np.array(Image.open(path))
    if gt.ndim == 3:
        gt = gt[..., 0]
    return gt.astype(np.int64)


def load_gt_keypoints(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nd = data.get("needle") if isinstance(data, dict) else None
    if not nd:
        return None, None
    return nd, nd.get("keypoints")


def live_rows(stage, source_name, frame_idx, stem, fps, gt_seg, gt_kp, hist,
              px_aln, rep, valid, n_frames, n_det, n_valid,
              thread_present=False, n_needle=(0, 0), pose_status=""):
    miou, pix = hist_miou(hist)
    all_px = [x for arr in px_aln for x in arr]
    valid_rate = 100.0 * n_valid / n_frames if n_frames else None
    det_rate = 100.0 * n_det / n_frames if n_frames else None
    thr_col = (120, 220, 140) if thread_present else (180, 180, 180)
    st_ok = pose_status in ("ok", "ok_fallback")
    st_skip = pose_status == "skipped_no_thread"
    st_col = (120, 220, 140) if st_ok else ((180, 180, 180) if st_skip else (230, 120, 110))
    rows = [
        ("Source", source_name[:24], (230, 230, 230)),
        ("Frame", f"{frame_idx}  {stem}", (230, 230, 230)),
        ("FPS", fmt(fps, 1), (110, 215, 255)),
        gpu_mem_row(),
        ("GT seg", "yes" if gt_seg else "no", (120, 220, 140) if gt_seg else (180, 180, 180)),
        ("GT kps", "yes" if gt_kp else "no", (120, 220, 140) if gt_kp else (180, 180, 180)),
    ]
    if stage == "Segmentation":
        rows += [
            ("mIoU live", fmt(miou, 2, "%"), (140, 235, 170)),
            ("Pixel acc", fmt(pix, 2, "%"), (140, 235, 170)),
            ("Thread", "yes" if thread_present else "no", thr_col),
        ]
    elif stage == "Keypoints":
        rows += [
            ("Thread", "yes" if thread_present else "no", thr_col),
            ("Needle px", f"{n_needle[0]}/{n_needle[1]}", (200, 210, 220)),
            ("KP status", pose_status or "n/a", st_col),
            ("Detect rate", fmt(det_rate, 1, "%"), (140, 235, 170)),
            ("Mean 2D px", fmt(mean_or_none(all_px), 2), (255, 220, 120)),
        ]
    else:
        rows += [
            ("Thread", "yes" if thread_present else "no", thr_col),
            ("Pose status", pose_status or "n/a", st_col),
            ("Valid pose", "yes" if valid else "no", (120, 220, 140) if valid else (230, 120, 110)),
            ("Valid rate", fmt(valid_rate, 1, "%"), (140, 235, 170)),
            ("Reproj", fmt(rep, 2, " px"), (255, 220, 120)),
        ]
    return rows


def run_source(job, engine, nk, calib, rvecR, tvecR, kp_names, args, device):
    K = args.num_keypoints
    nclass = len(args.seg_names)
    pk = None if args.no_smooth else PoseKalman()
    gate = args.reproj_gate_px
    writers = {"seg": None, "kp": None, "pose": None}
    show_enabled = bool(args.show)

    hist = np.zeros((nclass, nclass), dtype=np.int64)
    reproj_all, reproj_valid, fps_all = [], [], []
    px_raw = [[] for _ in range(K)]
    px_aln = [[] for _ in range(K)]
    mm_raw = [[] for _ in range(K)]
    mm_aln = [[] for _ in range(K)]
    n_frames = n_det = n_valid = n_gt = n_gt_pose = n_seg = 0
    tt_eval = tt_swap = 0
    fps_ema = None

    try:
        while True:
            t0 = time.perf_counter()
            L, R, stem = job["source"].read()
            if L is None:
                break
            n_frames += 1
            stem = stem or f"frame_{n_frames:06d}"

            ml, mr = seg_engine_batch(engine, [L, R], args.seg_size, device, args.patch)
            needleL = ml == args.needle_class
            needleR = mr == args.needle_class
            threadL = ml == args.thread_class

            gt_seg = stem in job["mask_idx"] and job["in_val"](stem)
            if gt_seg:
                gt = load_gt_mask(job["mask_idx"][stem])
                if gt.shape == ml.shape:
                    add_confusion(hist, ml, gt, nclass)
                    n_seg += 1

            threadR = mr == args.thread_class
            n_needle = (int(needleL.sum()), int(needleR.sum()))
            thread_present = (int(threadL.sum()) >= args.min_thread_px or
                              int(threadR.sum()) >= args.min_thread_px)
            needle_ok = n_needle[0] >= args.min_needle_px and n_needle[1] >= args.min_needle_px
            out = None
            # Rule: a frame WITHOUT suture thread may stay segmentation-only; a
            # frame WITH thread must attempt keypoints + pose (--pose-requires-thread
            # additionally SKIPS pose on thread-less frames instead of trying).
            if args.pose_requires_thread and not thread_present:
                pose_status = "skipped_no_thread"
            elif not needle_ok:
                pose_status = "no_needle"
            else:
                try:
                    out, pose_status = nk.process_frame(
                        needleL, needleR, threadL, calib, K, model_radius=args.model_radius
                    )
                except Exception as exc:
                    out, pose_status = None, f"error:{str(exc)[:24]}"

            if pk is not None and out is not None:
                ts, rs = pk.update(out["pose"]["t"], out["pose"]["rvec"])
                out["pose"]["t"] = list(map(float, ts))
                out["pose"]["rvec"] = list(map(float, rs))
                out["pose"]["R"] = cv2.Rodrigues(_np(rs))[0].tolist()
            elif pk is not None:
                pk.coast()

            if out is not None:
                n_det += 1
            rep = reprojection_error(out, calib["K1"], calib["D1"], calib["K2"], calib["D2"], rvecR, tvecR)
            if rep is not None:
                reproj_all.append(rep)
            valid = (out is not None) and (rep is not None) and (gate <= 0 or rep <= gate)
            if valid:
                n_valid += 1
                reproj_valid.append(rep)

            gt_kp = stem in job["kp_idx"] and job["in_val"](stem)
            if gt_kp:
                nd, gt_kps = load_gt_keypoints(job["kp_idx"][stem])
                if gt_kps and len(gt_kps) >= K:
                    n_gt += 1
                    if valid:
                        n_gt_pose += 1
                        gxy = _np([[g["x"], g["y"]] for g in gt_kps[:K]])
                        pxy = _np(out["left"][:K])
                        direct = np.hypot(*(pxy[0] - gxy[0])) + np.hypot(*(pxy[-1] - gxy[-1]))
                        swap = np.hypot(*(pxy[0] - gxy[-1])) + np.hypot(*(pxy[-1] - gxy[0]))
                        if nd.get("tip_tail_known", True):
                            tt_eval += 1
                            if swap < direct:
                                tt_swap += 1
                        order = list(range(K))[::-1] if swap < direct else list(range(K))
                        for i in range(K):
                            j = order[i]
                            px_raw[i].append(float(np.hypot(*(pxy[i] - gxy[i]))))
                            px_aln[i].append(float(np.hypot(*(pxy[j] - gxy[i]))))
                            gmm = gt_kps[i].get("xyz_mm")
                            if gmm is not None:
                                if out["xyz_mm"][i] is not None and np.all(np.isfinite(out["xyz_mm"][i])):
                                    mm_raw[i].append(float(np.linalg.norm(_np(out["xyz_mm"][i]) - _np(gmm))))
                                if out["xyz_mm"][j] is not None and np.all(np.isfinite(out["xyz_mm"][j])):
                                    mm_aln[i].append(float(np.linalg.norm(_np(out["xyz_mm"][j]) - _np(gmm))))

            dt = time.perf_counter() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps
            fps_all.append(inst_fps)

            if args.write_videos:
                segL = overlay_segmentation(L, ml, args.needle_class, args.thread_class)
                segR = overlay_segmentation(R, mr, args.needle_class, args.thread_class)
                seg_canvas = hstack_same_h(segL, segR)
                seg_canvas = append_panel(
                    seg_canvas, "Segmentation",
                    live_rows("Segmentation", job["name"], n_frames, stem, fps_ema, gt_seg,
                              gt_kp, hist, px_aln, rep, valid, n_frames, n_det, n_valid,
                              thread_present, n_needle, pose_status),
                )

                kpL, kpR = segL.copy(), segR.copy()
                if out is not None:
                    nk.draw_debug(kpL, out["left"], kp_names, out["visible"], tag=None)
                    nk.draw_debug(kpR, out["right"], kp_names, out["visible"], tag=None)
                kp_canvas = append_panel(
                    hstack_same_h(kpL, kpR), "Keypoints",
                    live_rows("Keypoints", job["name"], n_frames, stem, fps_ema, gt_seg,
                              gt_kp, hist, px_aln, rep, valid, n_frames, n_det, n_valid,
                              thread_present, n_needle, pose_status),
                )

                poseL, poseR = kpL.copy(), kpR.copy()
                if out is not None:
                    draw_pose_axes(poseL, out["pose"]["R"], out["pose"]["t"], calib["K1"], calib["D1"],
                                   np.zeros(3), np.zeros(3))
                    draw_pose_axes(poseR, out["pose"]["R"], out["pose"]["t"], calib["K2"], calib["D2"],
                                   rvecR, tvecR)
                    draw_reproj(poseL, out["xyz_mm"], calib["K1"], calib["D1"], np.zeros(3), np.zeros(3))
                    draw_reproj(poseR, out["xyz_mm"], calib["K2"], calib["D2"], rvecR, tvecR)
                pose_canvas = append_panel(
                    hstack_same_h(poseL, poseR), "Pose 6DoF",
                    live_rows("Pose", job["name"], n_frames, stem, fps_ema, gt_seg,
                              gt_kp, hist, px_aln, rep, valid, n_frames, n_det, n_valid,
                              thread_present, n_needle, pose_status),
                )

                for name, canvas in [("seg", seg_canvas), ("kp", kp_canvas), ("pose", pose_canvas)]:
                    if args.view_height and canvas.shape[0] != args.view_height:
                        scale = args.view_height / canvas.shape[0]
                        canvas = cv2.resize(
                            canvas, (int(round(canvas.shape[1] * scale)), args.view_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    if show_enabled and name == args.show_stage:
                        try:
                            cv2.imshow(f"EndoNeedle6DoF - {name}", canvas)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                show_enabled = False
                        except cv2.error as exc:
                            print(f"[split-eval] live display disabled: {exc}", flush=True)
                            show_enabled = False
                    if writers[name] is None:
                        sub = {"seg": "segmentation", "kp": "keypoints", "pose": "pose"}[name]
                        out_path = Path(args.out_dir) / "videos" / sub / f"{_safe_name(job['name'])}.mp4"
                        writers[name] = ensure_writer(out_path, canvas, args.fps_out)
                    writers[name].write(canvas)
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.release()
        if show_enabled:
            cv2.destroyAllWindows()
        job["source"].release()

    return dict(dataset=job.get("dataset", job["mode"]), key=job["key"], hist=hist, n_seg=n_seg,
                n_frames=n_frames, n_det=n_det, n_valid=n_valid,
                n_gt=n_gt, n_gt_pose=n_gt_pose, tt_eval=tt_eval, tt_swap=tt_swap,
                reproj_all=reproj_all, reproj_valid=reproj_valid, fps_all=fps_all,
                px_raw=px_raw, px_aln=px_aln, mm_raw=mm_raw, mm_aln=mm_aln)


def split_metrics(full):
    return {
        "segmentation_metrics": {
            "segmentation": full["segmentation"],
            "fps": full["fps"],
            "meta": full["meta"],
            "per_key": full["per_key"],
        },
        "keypoint_metrics": {
            "keypoints_2d_px": full["keypoints_2d_px"],
            "keypoints_3d_mm": full["keypoints_3d_mm"],
            "tip_tail": full["tip_tail"],
            "fps": full["fps"],
            "meta": full["meta"],
        },
        "pose_metrics": {
            "pose_coverage": full["pose_coverage"],
            "reprojection_px": full["reprojection_px"],
            "reprojection_px_ungated": full["reprojection_px_ungated"],
            "fps": full["fps"],
            "meta": full["meta"],
            "per_key": full["per_key"],
        },
    }


def build_jobs(args):
    jobs = []
    if args.image_dir:
        jobs.append({
            "mode": "frames",
            "name": Path(args.image_dir).name,
            "key": Path(args.image_dir).name,
            "source": FrameDirSource(args.image_dir, args.right_image_dir, args.limit),
            "mask_idx": auto_sidecar_index(args.image_dir, args.mask_dir, "masks", "png"),
            "kp_idx": auto_sidecar_index(args.image_dir, args.keypoint_dir, "keypoints", "json"),
            "in_val": lambda _stem: True,
        })
        return jobs

    if args.capture is not None:
        jobs.append({
            "mode": "capture",
            "name": f"capture_{args.capture}_{args.layout}",
            "key": f"capture_{args.capture}",
            "source": CaptureSplitSource(args.capture, args.layout, args.limit),
            "mask_idx": {},
            "kp_idx": {},
            "in_val": lambda _stem: True,
        })
        return jobs

    if args.left and args.right:
        name = f"{Path(str(args.left)).stem}__{Path(str(args.right)).stem}"
        jobs.append({
            "mode": "video",
            "name": name,
            "key": name,
            "source": VideoPairSource(args.left, args.right, args.limit),
            "mask_idx": {},
            "kp_idx": {},
            "in_val": lambda _stem: True,
        })
        return jobs

    if args.datasets:
        for dataset in args.datasets:
            meta = json.loads((Path(args.root) / dataset / "meta.json").read_text(encoding="utf-8"))
            for key in meta["videos"].keys():
                jobs.append({
                    "mode": "dataset",
                    "dataset": dataset,
                    "key": key,
                    "name": f"{dataset}__{key}",
                    "source": DatasetSource(args.root, dataset, key, args.limit),
                    "mask_idx": index_sidecars(args.root, dataset, key, "masks", "png"),
                    "kp_idx": index_sidecars(args.root, dataset, key, args.gt_subdir, "json") if args.gt_subdir else {},
                    "in_val": lambda _stem: True,
                })
        return jobs

    if not args.val_split:
        raise SystemExit("provide --val-split, --datasets, --image-dir, --left/--right, or --capture")
    keys, val_stems = parse_val_keys(args.val_split, args.root)
    for dataset, key in keys:
        allowed = val_stems.get((dataset, key)) if args.val_only else None
        jobs.append({
            "mode": "dataset",
            "dataset": dataset,
            "key": key,
            "name": f"{dataset}__{key}",
            "source": DatasetSource(args.root, dataset, key, args.limit),
            "mask_idx": index_sidecars(args.root, dataset, key, "masks", "png"),
            "kp_idx": index_sidecars(args.root, dataset, key, args.gt_subdir, "json") if args.gt_subdir else {},
            "in_val": (lambda stem, allowed=allowed: allowed is None or stem in allowed),
        })
    return jobs


def write_outputs(out_dir, full):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_full.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    for name, payload in split_metrics(full).items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metric_value(payload, path):
    cur = payload
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key) if isinstance(cur, dict) else None
    return cur


def summary_row(run_name, full):
    return {
        "run": run_name,
        "engine": metric_value(full, ["meta", "engine"]),
        "seg_size": metric_value(full, ["meta", "seg_size"]),
        "gate_px": metric_value(full, ["pose_coverage", "reproj_gate_px"]),
        "frames": metric_value(full, ["pose_coverage", "n_frames"]),
        "fps_mean": metric_value(full, ["fps", "mean"]),
        "miou": metric_value(full, ["segmentation", "mIoU"]),
        "miou_fg": metric_value(full, ["segmentation", "mIoU_fg"]),
        "pixel_acc": metric_value(full, ["segmentation", "pixel_acc"]),
        "kp2d_mean_px": metric_value(full, ["keypoints_2d_px", "overall_aligned", "mean"]),
        "kp3d_mean_mm": metric_value(full, ["keypoints_3d_mm", "overall_aligned", "mean"]),
        "pose_valid_pct": metric_value(full, ["pose_coverage", "valid_rate_pct"]),
        "gt_pose_success_pct": metric_value(full, ["pose_coverage", "gt_pose_success_pct"]),
        "reproj_mean_px": metric_value(full, ["reprojection_px", "mean"]),
    }


def write_sweep_summary(base_out, rows):
    if not rows:
        return
    base_out = Path(base_out)
    fields = list(rows[0].keys())
    with (base_out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |",
             "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for k in fields:
            v = row[k]
            vals.append("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    (base_out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


BUILTIN_SWEEP = [
    {"name": "ts640_s640_gate50", "engine": "weights/seg_engine_640.ts", "seg_size": 640, "gate": 50},
    {"name": "ts448_s448_gate50", "engine": "weights/seg_engine_448.ts", "seg_size": 448, "gate": 50},
    {"name": "trtfp32_448_s448_gate50", "engine": "weights/seg_engine_trtfp32_448.ts", "seg_size": 448, "gate": 50},
    {"name": "trtfp32_320_s320_gate50", "engine": "weights/seg_engine_trtfp32_320.ts", "seg_size": 320, "gate": 50},
    {"name": "ts448_s448_gate20", "engine": "weights/seg_engine_448.ts", "seg_size": 448, "gate": 20},
    {"name": "ts448_s448_gate30", "engine": "weights/seg_engine_448.ts", "seg_size": 448, "gate": 30},
    {"name": "ts448_s448_gate0", "engine": "weights/seg_engine_448.ts", "seg_size": 448, "gate": 0},
]


def parse_run_spec(spec):
    out = {}
    for item in spec.split(","):
        if not item:
            continue
        if "=" not in item:
            out["name"] = item
            continue
        k, v = item.split("=", 1)
        if k in {"seg_size", "gate"}:
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    if "name" not in out:
        out["name"] = f"{Path(out.get('engine', 'engine')).stem}_s{out.get('seg_size', 'x')}_g{out.get('gate', 'x')}"
    return out


def run_once(args, nk, calib, rvecR, tvecR, kp_names, device):
    engine = infer_accel.load_seg_engine(args.engine, device)
    jobs = build_jobs(args)
    print(f"[split-eval] {len(jobs)} source(s): " + ", ".join(j["name"] for j in jobs), flush=True)
    stats, t0 = [], time.perf_counter()
    for job in jobs:
        print(f"[split-eval] >>> {job['name']}", flush=True)
        stats.append(run_source(job, engine, nk, calib, rvecR, tvecR, kp_names, args, device))

    full = summarize(stats, args.num_keypoints, kp_names, args)
    full["meta"] = {
        "engine": args.engine,
        "seg_size": args.seg_size,
        "num_keypoints": args.num_keypoints,
        "val_only": bool(args.val_only),
        "model_radius_mm": args.model_radius,
        "wall_seconds": time.perf_counter() - t0,
        "n_sources": len(stats),
        "split_stage_outputs": True,
        "source_modes": sorted(set(s["dataset"] for s in stats)),
    }
    full["per_key"] = [{
        "dataset": s["dataset"], "key": s["key"], "frames": s["n_frames"],
        "detected": s["n_det"], "valid": s["n_valid"],
        "seg_scored": s["n_seg"], "gt_kp": s["n_gt"],
        "tt_swap": s["tt_swap"], "tt_eval": s["tt_eval"],
        "seg": seg_scores(s["hist"], args.seg_names) if s["n_seg"] else None,
    } for s in stats]
    write_outputs(args.out_dir, full)
    print(f"[split-eval] metrics: {Path(args.out_dir) / 'metrics_full.json'}", flush=True)
    if args.write_videos:
        print(f"[split-eval] videos: {Path(args.out_dir) / 'videos'}", flush=True)
    return full


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default="weights/seg_engine_640.ts")
    p.add_argument("--calib", required=True)
    p.add_argument("--needle-model", default=None)
    p.add_argument("--model-radius", type=float, default=None)

    p.add_argument("--root", default=None)
    p.add_argument("--val-split", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--val-only", action="store_true")

    p.add_argument("--image-dir", default=None, help="left frame folder")
    p.add_argument("--right-image-dir", default=None, help="right frame folder")
    p.add_argument("--mask-dir", default=None, help="optional GT mask folder for frame input")
    p.add_argument("--keypoint-dir", default=None, help="optional GT keypoint JSON folder for frame input")
    p.add_argument("--left", default=None, help="left video path, stream URL, or device index")
    p.add_argument("--right", default=None, help="right video path, stream URL, or device index")
    p.add_argument("--capture", default=None, help="single capture card index with stereo layout")
    p.add_argument("--layout", choices=["sbs", "tb"], default="sbs")

    p.add_argument("--out-dir", default="results/split_stage_eval")
    p.add_argument("--num-keypoints", type=int, default=5)
    p.add_argument("--needle-class", type=int, default=1)
    p.add_argument("--thread-class", type=int, default=2)
    p.add_argument("--min-needle-px", type=int, default=20,
                   help="min needle pixels per view to attempt keypoints/pose")
    p.add_argument("--min-thread-px", type=int, default=50,
                   help="min thread pixels (either view) to consider the suture thread present")
    p.add_argument("--pose-requires-thread", action="store_true",
                   help="only run keypoints/pose on frames that contain suture thread; "
                        "thread-less frames stay segmentation-only")
    p.add_argument("--seg-names", nargs="*", default=SEG_CLASS_NAMES)
    p.add_argument("--seg-size", type=int, default=640)
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--gt-subdir", default="keypoints")
    p.add_argument("--pck-px", type=float, default=10.0)
    p.add_argument("--pck-mm", type=float, default=2.0)
    p.add_argument("--reproj-gate-px", type=float, default=50.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fps-out", type=float, default=20.0)
    p.add_argument("--view-height", type=int, default=720)
    p.add_argument("--show", action="store_true", help="display one live visualization window while running")
    p.add_argument("--show-stage", choices=["seg", "kp", "pose"], default="pose",
                   help="which live panel to display when --show is set")
    p.add_argument("--sam2-tools", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--no-videos", dest="write_videos", action="store_false")
    p.set_defaults(write_videos=True)

    p.add_argument("--sweep", choices=["none", "speed_accuracy"], default="none",
                   help="run built-in parameter combinations and write summary.csv/md")
    p.add_argument("--run", action="append", default=[],
                   help="custom sweep spec: name=exp,engine=...,seg_size=448,gate=30")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.image_dir and not args.right_image_dir:
        raise SystemExit("--image-dir requires --right-image-dir")
    if (args.left and not args.right) or (args.right and not args.left):
        raise SystemExit("--left and --right must be provided together")
    if (args.val_split or args.datasets) and not args.root:
        raise SystemExit("--root is required for dataset inputs")

    if args.needle_model and os.path.isfile(args.needle_model):
        args.model_radius = float(json.loads(Path(args.needle_model).read_text(encoding="utf-8"))["radius_mm"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    nk = load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    rvecR = cv2.Rodrigues(_np(calib["R"]))[0].ravel()
    tvecR = _np(calib["t"]).ravel()
    kp_names = ["tip"] + [f"k{i}" for i in range(1, args.num_keypoints - 1)] + ["tail"]

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    run_specs = []
    if args.sweep == "speed_accuracy":
        run_specs.extend(BUILTIN_SWEEP)
    run_specs.extend(parse_run_spec(s) for s in args.run)

    if not run_specs:
        full = run_once(args, nk, calib, rvecR, tvecR, kp_names, device)
        write_sweep_summary(base_out, [summary_row("single", full)])
        return

    rows = []
    for spec in run_specs:
        run_args = copy.deepcopy(args)
        run_args.engine = spec.get("engine", args.engine)
        run_args.seg_size = int(spec.get("seg_size", args.seg_size))
        run_args.reproj_gate_px = float(spec.get("gate", args.reproj_gate_px))
        run_args.out_dir = str(base_out / _safe_name(spec["name"]))
        print(f"[split-eval] === run {spec['name']} ===", flush=True)
        try:
            full = run_once(run_args, nk, calib, rvecR, tvecR, kp_names, device)
        except RuntimeError as exc:
            msg = str(exc)
            if "torch.classes.tensorrt.Engine" not in msg:
                raise
            print(f"[split-eval] SKIP {spec['name']}: Torch-TensorRT runtime is not available.", flush=True)
            full = {
                "meta": {"engine": run_args.engine, "seg_size": run_args.seg_size},
                "pose_coverage": {"reproj_gate_px": run_args.reproj_gate_px, "n_frames": 0},
                "fps": {"mean": None},
                "segmentation": {},
                "keypoints_2d_px": {"overall_aligned": {}},
                "keypoints_3d_mm": {"overall_aligned": {}},
                "reprojection_px": {},
            }
        rows.append(summary_row(spec["name"], full))
    write_sweep_summary(base_out, rows)
    print(f"[split-eval] summary: {base_out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
