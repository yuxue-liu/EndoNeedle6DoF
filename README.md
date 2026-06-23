# Stereo-Endoscopic Surgical Needle Segmentation, Keypoint Localization, and 6-DoF Pose Estimation

This repository provides an inference and annotation toolkit for surgical suturing-needle
perception under a stereo endoscope. Given a calibrated stereo pair, the system performs
semantic segmentation of the needle, suture thread, and needle holder; localizes an ordered
set of needle keypoints; and recovers the full 6-DoF needle pose through stereo
triangulation and geometric registration. Segmentation is built on a DINOv2-base backbone
with a DPT decoder. The release is **inference-only**: model training code and raw network
weights are intentionally excluded; the segmentation network is distributed as a
self-contained TorchScript engine.

## Highlights

- **Stereo needle perception pipeline.** Segmentation → 2D centerline extraction →
  stereo triangulation → 6-DoF pose, producing per-frame keypoints (left/right 2D, metric
  3D), pose `(R, t, rvec)`, and reprojection-based confidence.
- **Two pose formulations.** A free-radius circular-arc fit (v1) and a model-based
  fixed-radius registration (v2) that constrains the reconstruction with the known needle
  geometry, improving robustness under occlusion and left/right asymmetry.
- **Accelerated inference (v3).** Numerically equivalent to v2, with a pre-exported
  TorchScript/TensorRT FP16 segmentation engine, `torch.compile` / channels-last execution,
  and an asynchronous I/O pipeline.
- **Encapsulated deployment.** `infer_engine_only.py` runs the complete pipeline from the
  exported engine alone, without importing the network definition or training code.
- **Multiple input sources.** Stored stereo-sequence replay, dual video streams or cameras,
  and a single side-by-side / top-bottom stereo capture device.

## Method Overview

1. **Segmentation.** A DINOv2-base encoder with a DPT decoder predicts four classes
   (background, needle, thread, needle holder). Both views are processed in a single batched
   forward pass under FP16.
2. **2D centerline and keypoints.** The largest needle connected component is skeletonized
   and fitted to an elliptical arc to yield an ordered tip-to-tail centerline; the thread
   endpoint disambiguates tip from tail.
3. **Stereo reconstruction.** Left/right centerlines are matched by arc-length and
   triangulated; a 3D circle (plane and radius) is fitted, and N keypoints are sampled at
   equal arc length and reprojected for visibility testing.
4. **6-DoF pose.** A pose frame is defined with the origin at the arc center, the z-axis
   along the arc-plane normal, and the x-axis toward the needle tip. In the model-based
   variant the radius is fixed to the calibrated needle model, reducing the per-frame
   estimation to the pose parameters alone.

A complete description of usage, input configuration, and the calibration procedure is
provided in [`docs/OPERATION_MANUAL.md`](docs/OPERATION_MANUAL.md).

## Repository Structure

```
.
├── inference/
│   ├── infer_engine_only.py            # encapsulated entry point (engine-only; no network source)
│   ├── infer_accel.py                  # acceleration helpers (engine load, compile, async I/O)
│   ├── export_seg_engine.py            # TorchScript/TensorRT engine export
│   ├── realtime_stereo_keypoints*.py   # single-sequence / streaming inference (v1, v2, v3)
│   └── eval_pose_val*.py               # validation-set evaluation (v1, v2, v3)
├── tools/
│   ├── needle_keypoints.py             # keypoint + pose module (v1, free radius)
│   ├── needle_keypoints_v2.py          # keypoint + pose module (v2, model-based registration)
│   └── calibrate_needle_radius.py      # needle-radius calibration -> needle_model.json
├── annotation/                         # interactive stereo annotation GUIs
├── calib/
│   ├── needle_calib.json               # stereo calibration
│   └── needle_model.json               # canonical needle radius (v2/v3)
├── configs/
│   └── surgical_combined_base.yaml     # segmentation configuration
├── weights/
│   └── seg_engine_640.ts               # TorchScript segmentation engine (FP16)
└── docs/
    └── OPERATION_MANUAL.md             # detailed operation manual
```

## Installation

A conda environment with Python 3.10, CUDA 12.1, and PyTorch is recommended.

```bash
conda create -n needle-infer python=3.10 -y
conda activate needle-infer

# PyTorch (select the build matching your CUDA toolkit; cu121 shown)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Inference dependencies
pip install -r requirements-inference.txt

# Optional: TensorRT backend for maximum throughput.
# If torch-tensorrt is unavailable, engine export falls back to TorchScript.
pip install torch-tensorrt
```

The interactive annotation GUIs additionally require a SAM-2 environment and its checkpoint,
together with `transformers<4.49`.

## Pretrained Engine

The segmentation network is released as a fixed-shape TorchScript engine,
`weights/seg_engine_640.ts`, built for a per-eye resolution of 1920×1080 at a segmentation
long-side of 640. The engine is hardware- and shape-specific; to target a different GPU or
input size, re-export with `inference/export_seg_engine.py`.

## Model Weights

Large checkpoints are distributed as release assets on the
[Releases](https://github.com/yuxue-liu/EndoNeedle6DoF/releases) page rather than tracked in
the repository.

| Asset | Description | Approx. size |
|---|---|---|
| `weights/seg_engine_640.ts` | DINOv2-base + DPT segmentation engine (included in the repository) | 190 MB |
| `best.pth` | DINOv2-base + DPT segmentation checkpoint (full precision) | 370 MB |
| `checkpoint_phase123.pt` | SAM2-Plus annotation predictor checkpoint | 760 MB |

```bash
mkdir -p checkpoints
curl -L -o checkpoints/checkpoint_phase123.pt \
  https://github.com/yuxue-liu/EndoNeedle6DoF/releases/download/v1.0/checkpoint_phase123.pt
curl -L -o weights/best.pth \
  https://github.com/yuxue-liu/EndoNeedle6DoF/releases/download/v1.0/best.pth
```

## Annotation

The interactive annotator (`annotation/`) is built on the SAM2-Plus video predictor packaged
in `sam2_plus/`: point/brush multi-class labeling with forward video propagation and
pause-resume correction. After downloading `checkpoint_phase123.pt`:

```bash
python annotation/app_gui.py \
  --image_dir <frames_dir> \
  --checkpoint checkpoints/checkpoint_phase123.pt
```

Stereo and dual-view correction variants (`app_gui_stereo.py`, `app_gui_dual.py`) follow the
same invocation.

## Quick Start

### Encapsulated inference (recommended)

Runs the full pipeline from the engine alone:

```bash
python inference/infer_engine_only.py \
  --engine weights/seg_engine_640.ts \
  --calib calib/needle_calib.json \
  --needle-model calib/needle_model.json \
  --sam2-tools tools \
  --left left.mp4 --right right.mp4 \
  --seg-size 640 --num-keypoints 5 \
  --save-video out.mp4 --save-results out.jsonl
```

### Accelerated validation-set evaluation

```bash
python inference/eval_pose_val_v3_accel.py \
  --config configs/surgical_combined_base.yaml \
  --calib calib/needle_calib.json \
  --needle-model calib/needle_model.json \
  --root <DATA_ROOT> \
  --val-split <DATA_ROOT>/combined/splits/r100/val.txt \
  --seg-engine weights/seg_engine_640.ts \
  --out-dir results/pose_val --seg-size 640 --num-keypoints 5 --no-video
```

### Supported inputs

| Input source | Selecting arguments |
|---|---|
| Stored stereo sequence | `--root <root> --dataset <name> --key <key>` |
| Dual video streams or cameras | `--left <left> --right <right>` |
| Single stereo capture device | `--capture <index> --layout sbs\|tb` |

Outputs comprise an overlay visualization video, a per-frame JSONL record (keypoints, pose,
confidence), and an optional flattened CSV of poses. Full parameter documentation is given in
[`docs/OPERATION_MANUAL.md`](docs/OPERATION_MANUAL.md).

## Release Scope

This repository provides the segmentation engine and full-precision checkpoint, the
stereo-needle inference pipeline, and the SAM2-Plus annotation toolkit with its predictor
package and checkpoint. The UniMatch-V2 training code and the auxiliary segmentation modules
used in our experiments are not part of this release.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@misc{liu2026endoneedle6dof,
  title        = {Stereo-Endoscopic Surgical Needle Segmentation, Keypoint
                  Localization, and 6-DoF Pose Estimation},
  author       = {Liu, Yuxue},
  year         = {2026},
  howpublished = {\url{https://github.com/yuxue-liu/EndoNeedle6DoF}},
  note         = {Inference and annotation toolkit}
}
```

A corresponding peer-reviewed publication will be linked here once available.

## License

See [`LICENSE`](LICENSE) for terms. The DINOv2 backbone and SAM-2 annotation components are
subject to their respective upstream licenses.
