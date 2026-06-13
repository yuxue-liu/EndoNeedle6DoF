# Stereo Needle Keypoint & 6-DoF Pose — Inference

Real-time stereo needle keypoint and 6-DoF pose estimation in surgical video.
Inference runs from a released **TorchScript** segmentation model (`seg.ts.pt`);
no model/training source is required. Inputs: capture card, video files, or a
stored dataset sequence.

## Installation

```bash
git clone https://github.com/yuxue-liu/SAM2-Plus.git
cd SAM2-Plus
conda create -n needle python=3.10 -y && conda activate needle
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy scipy scikit-image
```

## Download the model

Download the released weights from the repository **Releases** page and place
them anywhere (e.g. `weights/`):

| File | Description |
|------|-------------|
| `seg.ts.pt` | TorchScript segmentation model (self-contained: graph + weights) |
| `seg.ts.pt.json` | metadata (inference size, class count) |

## Run

Three input modes (choose one):

```bash
# (1) video files
python tools/infer_ts_stereo_keypoints.py --ts-model weights/seg.ts.pt \
  --calib tools/needle_calib.json --left left.mp4 --right right.mp4 \
  --num-keypoints 5 --save-video out.mp4 --save-results result.jsonl

# (2) capture card (one device, side-by-side stereo)
python tools/infer_ts_stereo_keypoints.py --ts-model weights/seg.ts.pt \
  --calib tools/needle_calib.json --capture 0 --layout sbs \
  --num-keypoints 5 --save-video out.mp4 --save-results result.jsonl

# (3) stored dataset sequence (with optional accuracy metrics)
python tools/infer_ts_stereo_keypoints.py --ts-model weights/seg.ts.pt \
  --calib tools/needle_calib.json --root <DATA_ROOT> --dataset <seq> --key <key> \
  --num-keypoints 5 --gt-subdir keypoints --pck-thresh 10 --save-results result.jsonl
```

Add `--show` only on a machine with a display (a headless server raises a Qt
error — use `--save-video` instead).

## Output

- LEFT|RIGHT overlay with the N keypoints (tip green / tail red), the object pose
  axes (X red, Y green, Z blue), reprojected keypoints (white rings), and live FPS.
- `--save-results result.jsonl`: per frame, the N keypoint 3D coordinates (mm) and
  the 6-DoF pose `{R, t, rvec}`.
- Dataset mode prints mean pixel error / PCK@τ against the reference keypoints.

## Calibration

`tools/needle_calib.json` holds the stereo calibration (`K1, K2, D1, D2, R, t`;
`t` in mm). Replace it with your own rig's values to use a different camera.

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `--num-keypoints` | number of equally spaced keypoints (tip→tail) |
| `--seg-size` | inference resolution (default from `.json`; smaller = faster / less VRAM) |
| `--no-amp` | disable fp16 (default fp16 on CUDA) |
| `--no-smooth` | disable Kalman pose smoothing |
| `--no-reproject` | hide the pose-axes / reprojection overlay |

## Files

| File | Purpose |
|------|---------|
| `tools/infer_ts_stereo_keypoints.py` | inference interface (loads TorchScript) |
| `tools/needle_keypoints.py` | geometric keypoint + pose computation |
| `tools/needle_calib.json` | stereo calibration |
| `docs/METHOD_keypoints.html` | method description |
| `docs/USAGE_keypoints.html` | usage (Chinese) |

## Citation

```bibtex
@misc{surgical_needle_keypoints,
  title  = {Stereo Needle Keypoint and 6-DoF Pose Estimation},
  author = {Yuxue Liu},
  year   = {2026},
  howpublished = {\url{https://github.com/yuxue-liu/SAM2-Plus}}
}
```
