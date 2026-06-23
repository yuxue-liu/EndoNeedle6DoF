# Operation Manual

This manual describes the inference, calibration, and annotation workflow for the
EndoNeedle6DoF toolkit. It complements the project [`README`](../README.md), which covers
installation and repository structure. The public release is inference-only; network training
is outside its scope and is not documented here.

## 1. Conventions

- Commands are issued from the repository root with the inference environment activated.
- Angle-bracketed tokens (`<...>`) denote user-supplied values.
- Required artifacts: the segmentation engine `weights/seg_engine_640.ts`, the stereo
  calibration `calib/needle_calib.json`, and the needle model `calib/needle_model.json`.

## 2. Inference Variants

The pose estimator is provided in three numerically distinct variants. They share the
segmentation, centerline, and triangulation stages and differ only in the pose formulation
and execution backend.

| Variant | Pose formulation | Primary entry |
|---|---|---|
| v1 | Free-radius circular-arc fit (radius re-estimated per frame) | `inference/realtime_stereo_keypoints.py`, `inference/eval_pose_val.py` |
| v2 | Model-based fixed-radius registration (recommended for accuracy) | `inference/realtime_stereo_keypoints_v2.py`, `inference/eval_pose_val_v2.py` |
| v3 | v2 formulation with accelerated execution (numerically equivalent) | `inference/realtime_stereo_keypoints_v3_accel.py`, `inference/eval_pose_val_v3_accel.py` |

For deployment, `inference/infer_engine_only.py` executes the complete v2/v3 pipeline directly
from the exported engine, without importing the network definition. It is the recommended
entry point for this release; the variant scripts above are provided for reference and require
the network source, which is not included.

## 3. Encapsulated Inference

`infer_engine_only.py` loads the TorchScript engine and runs segmentation, keypoint
localization, triangulation, and 6-DoF pose estimation. The engine is fixed-shape: the
run-time `--seg-size` and the per-eye source resolution must match those used at export
(640 and 1920×1080 for the provided engine).

```bash
python inference/infer_engine_only.py \
  --engine weights/seg_engine_640.ts \
  --calib calib/needle_calib.json \
  --needle-model calib/needle_model.json \
  --sam2-tools tools \
  --left <left.mp4> --right <right.mp4> \
  --seg-size 640 --num-keypoints 5 \
  --save-video out.mp4 --save-results out.jsonl
```

### 3.1 Input Sources

The input is selected by one of three mutually exclusive argument groups.

| Source | Selecting arguments | Notes |
|---|---|---|
| Stored stereo sequence | `--root <root> --dataset <name> --key <key>` | Replay of an archived sequence; ground-truth metrics are reported where annotations exist. |
| Dual video streams or cameras | `--left <left> --right <right>` | File paths or camera indices (e.g. `--left 0 --right 1`). |
| Single stereo capture device | `--capture <index> --layout sbs\|tb` | One device carrying side-by-side (`sbs`) or top-bottom (`tb`) stereo, split automatically. |

When the camera or capture device is changed, the stereo calibration must be recomputed for
the new optics; otherwise the metric reconstruction and pose are invalid.

### 3.2 Outputs

| Flag | File | Contents |
|---|---|---|
| `--save-video` | `*.mp4` | Left/right overlay with keypoints, pose axes, and reprojection. |
| `--save-results` | `*.jsonl` | Per-frame record: left/right 2D keypoints, metric 3D coordinates, visibility, pose `(R, t, rvec)`, and confidence. |

## 4. Inference Acceleration

The provided engine already encapsulates the accelerated segmentation backend (FP16,
both views in a single batched forward). A background prefetch and asynchronous encoding
pipeline overlaps disk decoding and video writing with GPU computation. These affect
throughput only and do not change the numerical results.

Re-exporting the engine for a different GPU or input resolution is performed with
`inference/export_seg_engine.py`; this step requires the network source and is therefore not
available in the inference-only release.

## 5. Calibration

### 5.1 Stereo Calibration

`calib/needle_calib.json` stores the intrinsic and extrinsic stereo parameters
(`K1, D1, K2, D2, R, t`). It is consumed by all inference entry points and must correspond to
the camera used to acquire the input.

### 5.2 Needle Radius

The model-based pose (v2/v3) requires the canonical needle arc radius, stored in
`calib/needle_model.json` as `{"radius_mm": ...}`. It is obtained once per needle type with
`tools/calibrate_needle_radius.py`, which aggregates high-confidence per-frame radius
estimates by a robust median:

```bash
python tools/calibrate_needle_radius.py \
  --root <root> --datasets <dataset> \
  --out calib/needle_model.json
```

## 6. Method Summary

- **Segmentation.** A DINOv2-base encoder with a DPT decoder predicts four classes
  (background, needle, thread, needle holder). Inference is performed in FP16 with both views
  processed in a single batched forward pass.
- **2D centerline and keypoints.** The largest needle connected component is skeletonized and
  fitted to an elliptical arc to produce an ordered tip-to-tail centerline; occlusion gaps are
  bridged by the fitted arc, and the thread endpoint disambiguates tip from tail.
- **Stereo reconstruction.** Left and right centerlines are matched by arc length and
  triangulated; a 3D circle (plane and radius) is fitted, and N keypoints are sampled at equal
  arc length and reprojected to assess visibility.
- **6-DoF pose.** A pose frame is defined with the origin at the arc center, the z-axis along
  the arc-plane normal, and the x-axis toward the needle tip. In the model-based variant the
  radius is fixed to the calibrated value, constraining the reconstruction under occlusion and
  left/right asymmetry.

## 7. Annotation Tool

The `annotation/` directory provides interactive stereo annotation GUIs built on SAM-2:
point/brush multi-class labeling with forward video propagation and pause-resume correction,
saving single-channel indexed masks. Stereo and dual-view correction variants are included.
These tools require a SAM-2 environment and checkpoint; refer to the SAM-2 documentation for
setup.
