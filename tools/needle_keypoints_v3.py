"""Needle keypoints + 6-DoF pose — v3 (robust geometry).

Improves the model-based v2 (needle_keypoints_v2.py) on the three failure modes
seen in evaluation, WITHOUT retraining and WITHOUT changing the 2D stage:

  1. ARC-CONSTRAINED 3D KEYPOINTS. v2 reported `xyz_mm` = the raw per-keypoint
     stereo triangulation, so a middle keypoint with near-zero disparity could
     blow up to hundreds of mm (poisoning the 3D mean and the pose). v3 fits the
     3D circle from the DENSE matched samples and then places every keypoint ON
     that circle (equal angle tip->tail). All 3D points are therefore bounded by
     the needle model -> no depth blow-ups; the 2D keypoints are unchanged.

  2. RANSAC FIXED-RADIUS CIRCLE FIT. The plane+centre are fit on RANSAC inliers
     of the dense triangulation, so a few bad left/right correspondences no
     longer drag the arc -> fewer geometrically-invalid poses (lower drop rate).

  3. TEMPORAL TIP/TAIL (TipTailTracker). The per-frame tip/tail cue (nearest
     thread) is ambiguous on some frames -> tip<->tail flips. The tracker keeps
     temporal continuity of the tip and corrects ambiguous frames, cutting the
     tip/tail classification error.

All 2D/centerline/utility functions are reused from v2; only the 3D stage and
the tracker are new. process_frame keeps the SAME signature and return dict as
v2, so eval_engine_only.py / infer_engine_only.py can use it interchangeably.
"""
import numpy as np

try:
    import cv2
except Exception as e:  # noqa
    raise SystemExit("needle_keypoints_v3.py needs OpenCV (cv2)") from e

# reuse the whole 2D + helper stack from v2 (centerline, tip/tail cues, calib,
# triangulation, fixed-radius fit, pose, drawing)
from needle_keypoints_v2 import (
    load_calib, fit_arc_2d, order_skeleton, extend_to_mask,
    tail_is_at_start, tail_by_width, undistort_poly, resample_arclen,
    triangulate, fit_circle_3d, fit_circle_3d_fixed_r, pose_from_arc,
    reproject, draw_debug, draw_diag, _HAVE_SCIPY,
)

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# --------------------------------------------------------------------------- #
# robust 3D circle
# --------------------------------------------------------------------------- #
def _circle_residual(X, center, u, v, radius):
    """Distance of each point to the circle (center, plane=span(u,v), radius):
    combine in-plane radial error and out-of-plane offset."""
    d = np.asarray(X, float) - center
    a = d @ u; b = d @ v
    w = d @ np.cross(u, v)
    radial = np.hypot(a, b) - radius
    return np.hypot(radial, w)


def ransac_circle_fixed_r(X, radius, iters=60, tol_mm=2.0, seed=0):
    """RANSAC fixed-radius 3D circle fit. Returns (center, u, v, r, inlier_mask).
    Falls back to a plain fit when there are too few points."""
    X = np.asarray(X, float)
    n = len(X)
    if n < 5:
        c, u, v, r = fit_circle_3d_fixed_r(X, radius)
        return c, u, v, r, np.ones(n, bool)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            c, u, v, r = fit_circle_3d_fixed_r(X[idx], radius)
        except Exception:
            continue
        res = _circle_residual(X, c, u, v, radius)
        inl = res < tol_mm
        s = int(inl.sum())
        if best is None or s > best[0]:
            best = (s, inl)
    inl = best[1] if best is not None else np.ones(n, bool)
    if inl.sum() < 3:
        inl = np.ones(n, bool)
    c, u, v, r = fit_circle_3d_fixed_r(X[inl], radius)
    return c, u, v, r, inl


def _arc_angles(X, center, u, v):
    d = np.asarray(X, float) - center
    return np.arctan2(d @ v, d @ u)


def sample_on_arc(center, u, v, radius, th_tip, th_tail, n):
    ths = np.linspace(th_tip, th_tail, n)
    return center + radius * (np.cos(ths)[:, None] * u + np.sin(ths)[:, None] * v)


# --------------------------------------------------------------------------- #
# temporal tip/tail
# --------------------------------------------------------------------------- #
def reverse_result(out, calib):
    """Reverse tip<->tail order in a result dict (2D L/R, 3D, visibility) and
    recompute the pose so the x-axis points to the new tip."""
    for k in ("left", "right", "xyz_mm", "visible"):
        out[k] = out[k][::-1]
    X3 = np.asarray(out["xyz_mm"], float)
    center = np.asarray(out["circle"]["center"], float)
    normal = np.asarray(out["circle"]["normal"], float)
    R, t, rvec = pose_from_arc(X3, center, normal)
    out["pose"] = dict(R=R.tolist(), t=t.tolist(), rvec=rvec.tolist())
    return out


class TipTailTracker:
    """Keeps temporal continuity of the needle tip to fix ambiguous tip/tail.

    Policy: when the per-frame cue is reliable (`tip_tail_known`) trust it and
    re-seed the tracker; when it is ambiguous, orient so the tip stays closest to
    the previous tip. A large unreliable jump also triggers a temporal override.
    """

    def __init__(self, override_ratio=0.5):
        self.prev_tip = None
        self.override_ratio = override_ratio

    def update(self, out, calib):
        if out is None:
            return out
        tip = np.asarray(out["xyz_mm"][0], float)
        tail = np.asarray(out["xyz_mm"][-1], float)
        ok = np.all(np.isfinite(tip)) and np.all(np.isfinite(tail))
        if ok and self.prev_tip is not None:
            d_keep = np.linalg.norm(tip - self.prev_tip)
            d_flip = np.linalg.norm(tail - self.prev_tip)
            known = bool(out.get("tip_tail_known", True))
            if (not known and d_flip < d_keep) or (known and d_flip < self.override_ratio * d_keep):
                out = reverse_result(out, calib)
                tip = np.asarray(out["xyz_mm"][0], float)
        if np.all(np.isfinite(tip)):
            self.prev_tip = tip
        return out


# --------------------------------------------------------------------------- #
# per-frame
# --------------------------------------------------------------------------- #
def process_frame(maskL, maskR, threadL, calib, n_kp, n_fit=40, arc_fit=True,
                  model_radius=None, ransac_tol_mm=2.0):
    """Same interface/return as v2.process_frame, with arc-constrained 3D
    keypoints and a RANSAC fixed-radius circle fit."""
    polyL = fit_arc_2d(maskL) if arc_fit else None
    if polyL is None:
        polyL = order_skeleton(maskL)
    polyR = fit_arc_2d(maskR) if arc_fit else None
    if polyR is None:
        polyR = order_skeleton(maskR)
    if polyL is None or polyR is None:
        return None, "no_centerline"
    polyL = extend_to_mask(polyL, maskL)
    polyR = extend_to_mask(polyR, maskR)

    # tip(0)->tail(-1) on the LEFT view (nearest-thread, then swage-thickness)
    flip = tail_is_at_start(polyL, threadL)
    tip_tail_known = flip is not None
    if flip is None:
        flip = tail_by_width(polyL, maskL)
    if flip is True:
        polyL = polyL[::-1]

    uL = undistort_poly(polyL, calib["K1"], calib["D1"])
    rL = resample_arclen(uL, n_fit)
    if rL is None:
        return None, "degenerate_poly"

    # orient the right arc consistently (smaller reprojection error)
    best = None
    for rev in (False, True):
        pR_try = polyR[::-1] if rev else polyR
        rR = resample_arclen(undistort_poly(pR_try, calib["K2"], calib["D2"]), n_fit)
        if rR is None:
            continue
        X = triangulate(calib["P1"], calib["P2"], rL, rR)
        err = np.median(np.linalg.norm(reproject(calib["P1"], X) - rL, axis=1))
        if best is None or err < best[0]:
            best = (err, rev, rR, X)
    if best is None:
        return None, "degenerate_poly"
    if best[1]:
        polyR = polyR[::-1]
    rR = best[2]
    Xdense = best[3]                       # dense matched triangulation, tip->tail order

    # ---- 2D keypoints: detected centerline (unchanged from v2) ----
    kpL = resample_arclen(polyL, n_kp)
    kpR = resample_arclen(polyR, n_kp)
    if kpL is None or kpR is None:
        return None, "degenerate_poly"

    # ---- robust 3D circle on the dense triangulation ----
    radius_fixed = bool(model_radius and model_radius > 0)
    try:
        if radius_fixed:
            center, u, v, r, inl = ransac_circle_fixed_r(
                Xdense, float(model_radius), tol_mm=ransac_tol_mm)
        else:
            center, u, v, r = fit_circle_3d(Xdense)
            inl = np.ones(len(Xdense), bool)
    except Exception:
        # fall back to raw triangulation of the keypoints (v2 behaviour)
        uKL = undistort_poly(kpL, calib["K1"], calib["D1"])
        uKR = undistort_poly(kpR, calib["K2"], calib["D2"])
        X3 = triangulate(calib["P1"], calib["P2"], uKL, uKR)
        center, u, v, r = np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]), 0.0
        normal = np.cross(u, v)
        R6, t6, rvec6 = pose_from_arc(X3, center, normal)
        return _pack(X3, kpL, kpR, maskL, maskR, tip_tail_known,
                     center, normal, r, False, R6, t6, rvec6, 0.0), "ok_fallback"

    # ---- ARC-CONSTRAINED 3D keypoints: place tip..tail equally by angle ----
    th = np.unwrap(_arc_angles(Xdense[inl], center, u, v))   # inliers, tip->tail order
    k = max(1, len(th) // 10)
    th_tip = float(np.median(th[:k]))
    th_tail = float(np.median(th[-k:]))
    X3 = sample_on_arc(center, u, v, r, th_tip, th_tail, n_kp)   # (n_kp,3) on circle
    normal = np.cross(u, v)
    R6, t6, rvec6 = pose_from_arc(X3, center, normal)

    # confidence: stereo reprojection consistency of the dense samples
    conf = 1.0
    if _HAVE_SCIPY and cKDTree is not None:
        treeL = cKDTree(uL)
        treeR = cKDTree(undistort_poly(polyR, calib["K2"], calib["D2"]))
        dL, _ = treeL.query(reproject(calib["P1"], Xdense))
        dR, _ = treeR.query(reproject(calib["P2"], Xdense))
        conf = float(np.exp(-np.median(np.concatenate([dL, dR])) / 5.0))

    return _pack(X3, kpL, kpR, maskL, maskR, tip_tail_known,
                 center, normal, r, radius_fixed, R6, t6, rvec6, conf,
                 polyL, polyR), "ok"


def _pack(X3, kpL, kpR, maskL, maskR, tip_tail_known, center, normal, r,
          radius_fixed, R6, t6, rvec6, conf, polyL=None, polyR=None):
    def inside(mask, xy, rad=4):
        h, w = mask.shape
        out = []
        for (x, y) in xy:
            xi, yi = int(round(x)), int(round(y))
            ok = False
            if 0 <= xi < w and 0 <= yi < h:
                y0, y1 = max(0, yi - rad), min(h, yi + rad + 1)
                x0, x1 = max(0, xi - rad), min(w, xi + rad + 1)
                ok = bool(mask[y0:y1, x0:x1].any())
            out.append(ok)
        return np.array(out)

    visL = inside(maskL.astype(bool), kpL)
    visR = inside(maskR.astype(bool), kpR)
    visible = (visL | visR).astype(int)
    d = dict(
        xyz_mm=np.asarray(X3, float).tolist(),
        left=np.asarray(kpL, float).tolist(), right=np.asarray(kpR, float).tolist(),
        visible=visible.tolist(), tip_tail_known=bool(tip_tail_known),
        circle=dict(center=np.asarray(center, float).tolist(),
                    normal=np.asarray(normal, float).tolist(),
                    radius_mm=float(r), radius_fixed=bool(radius_fixed)),
        pose=dict(R=R6.tolist(), t=t6.tolist(), rvec=rvec6.tolist()),
        conf=float(conf),
    )
    if polyL is not None:
        d["polyL"] = np.asarray(polyL, float).tolist()
        d["polyR"] = np.asarray(polyR, float).tolist()
    return d
