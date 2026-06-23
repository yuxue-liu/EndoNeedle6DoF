import os
import numpy as np
import torch
import cv2


# ============================================================
# PALETTE (background + up to a handful of classes)
# index 0 = background (no overlay)
# ============================================================
PALETTE = {                  # BGR, softened for a clean translucent overlay
    0: (0, 0, 0),            # background
    1: (120, 205, 110),      # class 1 - soft green   (needle)
    2: (110, 120, 235),      # class 2 - soft coral    (thread)
    3: (225, 170, 95),       # class 3 - soft blue     (clamps)
    4: (90, 215, 235),       # class 4 - soft amber
    5: (210, 120, 210),      # class 5 - soft violet
}


# ============================================================
# MASK DECODING
# ============================================================
def logits_to_bool(video_res_masks):
    """
    video_res_masks: tensor (num_objs, 1, H, W) of mask logits.
    Returns list of (H,W) bool numpy arrays, one per object (in obj order).
    """
    out = []
    m = video_res_masks
    if m.dim() == 4:
        for i in range(m.shape[0]):
            out.append((m[i, 0] > 0.0).cpu().numpy())
    else:  # (1,H,W) or (H,W)
        out.append((m.squeeze() > 0.0).cpu().numpy())
    return out


def compose_index_mask(per_obj_bool, obj_ids, H, W):
    """
    Combine per-object binary masks into a single uint8 index map.
    per_obj_bool: list of (H,W) bool, aligned with obj_ids.
    obj_ids: list of class ids (== obj_id). Higher id wins on overlap.
    """
    idx = np.zeros((H, W), dtype=np.uint8)
    order = sorted(range(len(obj_ids)), key=lambda i: obj_ids[i])  # ascending
    for i in order:
        cls = int(obj_ids[i])
        m = per_obj_bool[i]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        idx[m] = cls
    return idx


# ============================================================
# SAVE / LOAD INDEX MASK
# ============================================================
def image_to_mask_path(image_path):
    """Mirror images/ -> masks/ and .jpg -> .png (UniMatch convention)."""
    p = image_path.replace("\\", "/")
    p = p.replace("/images/", "/masks/")
    root, _ = os.path.splitext(p)
    return root + ".png"


def save_index_mask(mask_path, idx):
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, idx)  # single-channel uint8 PNG


def load_index_mask(mask_path):
    if not os.path.exists(mask_path):
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return m


# ============================================================
# VISUALIZATION
# ============================================================
def overlay_index_mask(img, idx, alpha=0.38, sat=0.82, palette=PALETTE):
    """
    Color-overlay an index map (0=bg) onto a BGR image.
    alpha : overlay opacity (lower = more transparent / see-through).
    sat   : color saturation (lower = paler; 1.0 = pure palette color).
            Colors are mixed toward mid-gray so the underlying tissue and the
            occlusion boundaries between classes stay visible.
    """
    if idx is None:
        return img
    if idx.shape[:2] != img.shape[:2]:
        idx = cv2.resize(idx, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(img)
    present = []
    for cls, bgr in palette.items():
        if cls == 0:
            continue
        m = idx == cls
        if not m.any():
            continue
        desat = tuple(int(c * sat + 128 * (1 - sat)) for c in bgr)
        color[m] = desat
        present.append((bgr, m))
    out = img.copy()
    fg = idx > 0
    if fg.any():
        blended = cv2.addWeighted(color, alpha, img, 1 - alpha, 0)
        out[fg] = blended[fg]
    # crisp colored outline per class -> clean, distinguishable, still translucent fill
    for bgr, m in present:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(c) for c in bgr), 1, cv2.LINE_AA)
    return out


def overlay_mask(img, mask, alpha=0.5):
    """Backward-compatible single binary-mask overlay (green)."""
    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
