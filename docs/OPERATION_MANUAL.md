# Operation Manual

End-to-end workflow for the EndoNeedle6DoF toolkit, in the order it is normally run:

1. **Data annotation** — label needle / thread / holder masks (and keypoints) on stereo frames.
2. **Segmentation model** — obtain / export the segmentation engine and measure its accuracy.
3. **Inference & 6-DoF pose** — run the full stereo pipeline and dump the evaluation metrics.

All commands are issued **from the repository root** and use **relative paths**; angle-bracketed
tokens (`<...>`) are user-supplied. The public release is inference-only — network *training* is
out of scope and not documented here.

---

## 0. Environment & conventions

```bash
conda activate needle-infer          # or the server env that has torch + the deps
cd EndoNeedle6DoF                    # all commands below assume this working directory
```

Required artifacts (all referenced by relative path):

| Path | Purpose |
|---|---|
| `weights/seg_engine_640.ts` | TorchScript segmentation engine (FP16, per-eye 1920×1080, seg long-side 640) |
| `calib/needle_calib.json` | stereo calibration (`K1,D1,K2,D2,R,t`; `t` in mm) |
| `calib/needle_model.json` | canonical needle arc radius for model-based pose |
| `checkpoints/checkpoint_phase123.pt` | SAM2-Plus predictor checkpoint (annotation only) |

**Expected dataset layout** (a "dataset" is one acquisition session; a "key" is one stereo video):

```
<DATA_ROOT>/<dataset>/
  meta.json                                   # {videos: {<key>: [{ordinal, image, mask}, ...]}}
  images/<key>/.../<stem>.jpg                 # left frames
  stereo_right/<key>/<stem>.jpg               # right frames
  masks/<key>/.../<stem>.png                  # GT seg: 0=bg 1=needle 2=thread 3=holder
  keypoints/<key>/.../<stem>.json             # GT keypoints (needle.keypoints[*]: x,y,xyz_mm,...)
```

A `combined/splits/r100/val.txt` lists held-out frames as `"<dataset>/images/<key>/.../<stem>.jpg\t<...>/masks/.../<stem>.png"`.

---

## 1. Data annotation

The interactive annotator (`annotation/`) is built on the SAM2-Plus video predictor in `sam2_plus/`:
multi-class point/brush labeling with forward video propagation and pause-resume correction. It
writes single-channel indexed masks (the class ids above).

```bash
# monocular / left-view annotation
python annotation/app_gui.py \
  --image_dir <frames_dir> \
  --checkpoint checkpoints/checkpoint_phase123.pt
```

Stereo and dual-view correction variants share the same invocation:

```bash
python annotation/app_gui_stereo.py --image_dir <frames_dir> --checkpoint checkpoints/checkpoint_phase123.pt
python annotation/app_gui_dual.py   --image_dir <frames_dir> --checkpoint checkpoints/checkpoint_phase123.pt
```

The annotation GUIs need a display and a SAM-2 environment with `transformers<4.49`. Output masks
follow the dataset layout in §0 and are consumed directly by the evaluation in §3.

---

## 2. Segmentation model

The segmentation network is a DINOv2-base encoder with a DPT decoder predicting four classes
(background, needle, thread, holder). It is shipped as a **self-contained TorchScript engine** so
no network/training source is needed at run time.

### 2.1 Using the provided engine

`weights/seg_engine_640.ts` is ready to use. It is **fixed-shape**: the run-time `--seg-size` and
the per-eye source resolution must match the export (640 and 1920×1080). To target a different GPU
or resolution, re-export (requires the network source, full repo only):

```bash
python inference/export_seg_engine.py --checkpoint <best.pth> --seg-size 640 --out weights/seg_engine_640.ts
```

### 2.2 Measuring segmentation accuracy

Segmentation mIoU is reported as part of the consolidated evaluation in §3 (left view, against the
GT masks). The relevant block in `results/.../metrics.json` is:

```json
"segmentation": { "per_class_iou": {...}, "mIoU": ..., "mIoU_fg": ..., "pixel_acc": ... }
```

---

## 3. Inference & 6-DoF pose

### 3.1 Deployment (single sequence / live)

`inference/infer_engine_only.py` runs the complete engine-only pipeline (segmentation → centerline
→ stereo triangulation → 6-DoF pose) and writes an overlay video and a per-frame JSONL.

```bash
python inference/infer_engine_only.py \
  --engine weights/seg_engine_640.ts \
  --calib calib/needle_calib.json \
  --needle-model calib/needle_model.json \
  --left <left.mp4> --right <right.mp4> \
  --seg-size 640 --num-keypoints 5 \
  --save-video out.mp4 --save-results out.jsonl
```

Input sources (mutually exclusive):

| Source | Arguments |
|---|---|
| Stored stereo sequence | `--root <DATA_ROOT> --dataset <name> --key <key>` |
| Dual videos or cameras | `--left <l> --right <r>` (file paths or camera indices) |
| Single stereo device | `--capture <idx> --layout sbs\|tb` |

### 3.2 Consolidated evaluation (recommended for results)

`inference/eval_engine_only.py` replays the stored stereo sequences and saves **one** metrics file
covering everything reported in the paper. It is engine-only (no network source required).

```bash
python inference/eval_engine_only.py \
  --engine weights/seg_engine_640.ts \
  --calib calib/needle_calib.json \
  --needle-model calib/needle_model.json \
  --root <DATA_ROOT> \
  --val-split <DATA_ROOT>/combined/splits/r100/val.txt --val-only \
  --seg-size 640 --num-keypoints 5 \
  --out-dir results/eval_r100val --no-video
```

- Keys with a `stereo_right/` directory are auto-selected from the split. Use `--datasets <a> <b>`
  to evaluate explicit datasets (all keys) instead of a split.
- `--val-only` restricts the GT (segmentation / keypoint) metrics to the val stems; pose coverage is
  always measured over the complete video.
- Add `--no-video` for numbers-only (fastest); drop it to also write per-key overlay videos.

#### Metrics written to `results/<out-dir>/metrics.json`

| Group | Key fields | Meaning |
|---|---|---|
| `segmentation` | `per_class_iou`, `mIoU`, `mIoU_fg`, `pixel_acc` | left-view seg accuracy vs. GT masks |
| `keypoints_2d_px` | `overall_aligned.mean/median`, `PCK@<px>`, `per_kp_*` | 2D keypoint pixel error |
| `keypoints_3d_mm` | `overall_aligned.mean/median/rmse`, `tip_mm`, `tail_mm`, `PCK@<mm>` | metric 3D keypoint error (triangulated) |
| `tip_tail` | `swap_error_rate_pct`, `n_swapped/n_eval` | tip↔tail **classification** error (ordering reversed vs. GT) |
| `pose_coverage` | `detect_rate_pct`, `drop_rate_pct`, `gt_pose_success_pct`, `gt_pose_drop_pct` | pose **dropped-frame** stats (overall and on GT/needle-present frames = 漏帧) |
| `reprojection_px` | `mean/median` | 3D→2D reprojection error in both views (no GT needed) |
| `fps`, `per_key` | — | speed and per-video breakdown |

2D/3D errors are reported **both raw and tip/tail-aligned**: the aligned figure isolates
localization accuracy, while `tip_tail.swap_error_rate_pct` quantifies the ordering error separately.
Useful knobs: `--pck-px` (default 10), `--pck-mm` (default 2), `--num-keypoints`, `--limit` (cap
frames per key for a quick check), `--seg-names` (override class names/order).

---

## 4. Calibration

### 4.1 Stereo calibration

`calib/needle_calib.json` stores `K1, D1, K2, D2, R, t` and is consumed by every inference and
evaluation entry point. It must match the camera that acquired the input; recompute it whenever the
optics change, otherwise the metric reconstruction and pose are invalid.

### 4.2 Needle radius

The model-based pose fixes the needle arc radius from `calib/needle_model.json` (`{"radius_mm": ...}`),
obtained once per needle type by a robust median over high-confidence per-frame estimates:

```bash
python tools/calibrate_needle_radius.py --root <DATA_ROOT> --datasets <dataset> --out calib/needle_model.json
```

---

## 5. Method summary

- **Segmentation.** DINOv2-base + DPT predicts four classes; both views run in one batched FP16 forward.
- **2D centerline / keypoints.** The largest needle component is skeletonized and fitted to an
  elliptical arc to produce an ordered tip→tail centerline; occlusion gaps are bridged by the fitted
  arc, and the thread endpoint disambiguates tip from tail.
- **Stereo reconstruction.** Left/right centerlines are matched by arc length and triangulated; a 3D
  circle (plane + radius) is fitted and N keypoints are sampled at equal arc length, then reprojected
  for visibility.
- **6-DoF pose.** The pose frame has its origin at the arc center, z along the arc-plane normal, and x
  toward the tip. The model-based variant fixes the radius to the calibrated value, constraining the
  reconstruction under occlusion and left/right asymmetry.
