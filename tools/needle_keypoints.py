"""Needle centerline keypoints from a calibrated stereo (bimodal) surgical set.

Method (see design notes):
  1. Per view, take the needle segmentation mask, skeletonize -> ordered 2D
     centerline polyline (tip..tail). Left mask = GT (masks/ mirror); right mask
     = full-supervised model prediction (--right-pred-dir).
  2. Undistort the centerline points (K,D) and put them in linear-K pixel space.
  3. RIGID-ARC STEREO FIT: match the two polylines by arc-length fraction (init
     correspondence only), triangulate -> 3D points, fit a 3D circle (plane +
     circle-in-plane). The needle is a planar circular arc, so this 3D model is
     the rigid-body constraint and lets us restore the occluded part by simply
     extrapolating along the fitted arc.
  4. TIP/TAIL: the end nearest the `thread` mask is the tail (swaged to suture);
     the other is the tip. Orient the model tip->tail.
  5. Sample N (=5) points EQUALLY SPACED BY ARC LENGTH (equal angle on the
     circle) from tip to tail; reproject to BOTH views. A keypoint is visible=1
     if its reprojection lands inside the needle mask in >=1 view, else 0
     (restored / occluded).
  6. Export a per-frame sidecar JSON mirroring masks/ layout:
        <dataset>/keypoints/<key>/part_xxx/<stem>.json

Run AFTER you have right-eye predictions (see tools/build_stereo_id_path.py +
test.py --save-preds with the full-sup weights).

Example:
    python tools/needle_keypoints.py \
        --root /root/autodl-tmp/data/surgical_seg --dataset march_1 --key 1_01 \
        --right-pred-dir /root/autodl-tmp/exp/right_pred/march_1 \
        --calib tools/needle_calib.json \
        --num-keypoints 5 --debug-dir /root/autodl-tmp/exp/kp_debug/march_1
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as e:  # noqa
    raise SystemExit("needle_keypoints.py needs OpenCV (cv2)") from e

try:
    from skimage.morphology import skeletonize
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False

try:
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def load_calib(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    K1 = np.asarray(d["K1"], float); K2 = np.asarray(d["K2"], float)
    D1 = np.asarray(d["D1"], float); D2 = np.asarray(d["D2"], float)
    R = np.asarray(d["R"], float); t = np.asarray(d["t"], float).reshape(3, 1)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t])
    return dict(K1=K1, K2=K2, D1=D1, D2=D2, R=R, t=t, P1=P1, P2=P2)


# --------------------------------------------------------------------------- #
# 2D centerline
# --------------------------------------------------------------------------- #
def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == biggest


def order_skeleton(mask_bool, min_pixels=20):
    """Return ordered (M,2) (x,y) centerline along the skeleton's longest path."""
    if mask_bool.sum() < min_pixels:
        return None
    mask_bool = _largest_component(mask_bool)
    if _HAVE_SKIMAGE:
        sk = skeletonize(mask_bool)
    else:
        sk = cv2.ximgproc.thinning(mask_bool.astype(np.uint8) * 255).astype(bool) \
            if hasattr(cv2, "ximgproc") else mask_bool
    ys, xs = np.where(sk)
    if len(xs) < 5:
        return None
    pts = set(zip(xs.tolist(), ys.tolist()))
    nbrs = {}
    for (x, y) in pts:
        adj = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (dx or dy) and (x + dx, y + dy) in pts]
        nbrs[(x, y)] = adj

    def bfs_far(src):
        seen = {src: None}
        order = [src]
        qi = 0
        while qi < len(order):
            u = order[qi]; qi += 1
            for v in nbrs[u]:
                if v not in seen:
                    seen[v] = u
                    order.append(v)
        return order[-1], seen

    start = next(iter(pts))
    a, _ = bfs_far(start)
    b, parent = bfs_far(a)               # a..b = graph diameter (tree approx)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    if len(path) < 5:
        return None
    return np.asarray(path, float)       # (M,2) x,y


def _ellipse_param_angle(pts, e):
    (cx, cy), (MA, ma), ang = e
    a = max(MA / 2.0, 1e-6); b = max(ma / 2.0, 1e-6); phi = np.deg2rad(ang)
    dx = pts[:, 0] - cx; dy = pts[:, 1] - cy
    xr = dx * np.cos(phi) + dy * np.sin(phi)
    yr = -dx * np.sin(phi) + dy * np.cos(phi)
    return np.arctan2(yr / b, xr / a)


def _ellipse_point(t, e):
    (cx, cy), (MA, ma), ang = e
    a = MA / 2.0; b = ma / 2.0; phi = np.deg2rad(ang)
    xr = a * np.cos(t); yr = b * np.sin(t)
    x = cx + xr * np.cos(phi) - yr * np.sin(phi)
    y = cy + xr * np.sin(phi) + yr * np.cos(phi)
    return np.stack([x, y], -1)


def _ellipse_resid(pts, e):
    return np.linalg.norm(pts - _ellipse_point(_ellipse_param_angle(pts, e), e), axis=1)


def _skeleton_points(mask_bool):
    if _HAVE_SKIMAGE:
        sk = skeletonize(mask_bool)
    elif hasattr(cv2, "ximgproc"):
        sk = cv2.ximgproc.thinning(mask_bool.astype(np.uint8) * 255).astype(bool)
    else:
        sk = mask_bool
    ys, xs = np.where(sk)
    return np.stack([xs, ys], 1).astype(np.float32) if len(xs) else None


def fit_arc_2d(mask_bool, n=60, min_area=12, resid_px=2.5):
    """Reconstruct the COMPLETE needle centerline from possibly MULTIPLE mask
    segments (occlusion splits the needle) by robustly fitting ONE ellipse arc
    to the skeleton points of all components, rejecting spurious blobs, and
    sampling the arc across the occluded gap. Returns an ordered (n,2) polyline
    or None (caller falls back to order_skeleton)."""
    m = mask_bool.astype(np.uint8)
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    pts = []
    for c in range(1, ncomp):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        sp = _skeleton_points(lab == c)
        if sp is not None:
            pts.append(sp)
    if not pts:
        return None
    P = np.concatenate(pts, 0)
    if len(P) < 6:
        return None
    # robust ellipse fit: trim worst residuals a few times (drops outlier blobs)
    e = None; Q = P
    for _ in range(4):
        if len(Q) < 5:
            break
        e = cv2.fitEllipse(Q)
        d = _ellipse_resid(Q, e)
        thr = max(np.quantile(d, 0.8), resid_px)
        Qn = Q[d <= thr]
        if len(Qn) < 5:
            break
        Q = Qn
    if e is None:
        return None
    inl = P[_ellipse_resid(P, e) <= max(resid_px, np.quantile(_ellipse_resid(P, e), 0.8))]
    if len(inl) < 5:
        return None
    # order inlier angles avoiding the largest angular gap; that gap is the part
    # of the ellipse the needle does NOT occupy, so the arc spans the rest.
    a = np.sort(_ellipse_param_angle(inl, e))
    k = len(a)
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    gi = int(np.argmax(gaps))
    order = [(gi + 1 + j) % k for j in range(k)]
    ao = np.unwrap(a[order])
    ts = np.linspace(ao[0], ao[-1], n)
    return _ellipse_point(ts, e)            # complete, gap-filled, ordered


def undistort_poly(poly_xy, K, D):
    pts = poly_xy.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, D, P=K)
    return und.reshape(-1, 2)


def resample_arclen(poly, n):
    """Resample an ordered polyline to n points at equal chord-length fractions."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return None
    u = np.linspace(0, s[-1], n)
    x = np.interp(u, s, poly[:, 0]); y = np.interp(u, s, poly[:, 1])
    return np.stack([x, y], 1)


# --------------------------------------------------------------------------- #
# tip / tail
# --------------------------------------------------------------------------- #
def tail_is_at_start(poly_xy, thread_mask, tie_px=6.0):
    """RULE: the needle end nearest the suture thread is the TAIL.
    Returns True if the START endpoint is the tail, False if the END is, or
    None if undecidable (no thread / near tie)."""
    if thread_mask is None or int(thread_mask.sum()) == 0:
        return None
    ys, xs = np.where(thread_mask)
    tp = np.stack([xs, ys], 1).astype(float)
    d0 = float(np.min(np.linalg.norm(tp - poly_xy[0], axis=1)))
    d1 = float(np.min(np.linalg.norm(tp - poly_xy[-1], axis=1)))
    if abs(d0 - d1) < tie_px:
        return None                      # both ends ~equally close -> can't tell
    return d0 < d1                       # start nearer thread => start is tail


def tail_by_width(poly_xy, mask_bool, tie=0.6):
    """Fallback tip/tail when no thread is visible: the TAIL (swage) end is
    thicker than the sharp TIP. Returns True if the START end is thicker (=tail)."""
    dt = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    h, w = dt.shape

    def thick(p):
        x, y = int(round(p[0])), int(round(p[1]))
        x0, x1 = max(0, x - 6), min(w, x + 7)
        y0, y1 = max(0, y - 6), min(h, y + 7)
        sub = dt[y0:y1, x0:x1]
        return float(sub.max()) if sub.size else 0.0

    w0, w1 = thick(poly_xy[0]), thick(poly_xy[-1])
    if abs(w0 - w1) < tie:
        return None
    return w0 > w1


# --------------------------------------------------------------------------- #
# 3D circle
# --------------------------------------------------------------------------- #
def triangulate(P1, P2, xy1, xy2):
    X = cv2.triangulatePoints(P1, P2, xy1.T, xy2.T)
    return (X[:3] / X[3]).T              # (N,3)


def fit_circle_3d(X):
    """Fit plane + circle to 3D points. Returns (center, u, v, radius)."""
    c0 = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c0)
    u, v = Vt[0], Vt[1]                  # in-plane basis
    a = (X - c0) @ u; b = (X - c0) @ v
    A = np.stack([a, b, np.ones_like(a)], 1)
    sol, *_ = np.linalg.lstsq(A, -(a**2 + b**2), rcond=None)
    uc, vc = -sol[0] / 2, -sol[1] / 2
    r = float(np.sqrt(max(uc**2 + vc**2 - sol[2], 1e-9)))
    center = c0 + uc * u + vc * v
    return center, u, v, r


def pose_from_arc(X3, center, normal):
    """6-DoF pose of the rigid needle in the camera (cam1) frame, from the fitted
    arc: origin = arc center; z = plane normal; x = in-plane direction to the tip;
    y = z x x. Returns (R 3x3 [cols=axes], t 3, rvec 3)."""
    center = np.asarray(center, float)
    n = np.asarray(normal, float)
    nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-9 else np.array([0., 0., 1.])
    if n[2] > 0:                              # fix sign ambiguity: face the camera
        n = -n
    tip = np.asarray(X3[0], float)
    x = tip - center
    x = x - x.dot(n) * n                      # project to the arc plane
    xn = np.linalg.norm(x)
    if xn < 1e-6:                             # degenerate fallback
        x = np.array([1., 0., 0.]) - n[0] * n
        xn = np.linalg.norm(x)
    x = x / max(xn, 1e-9)
    z = n
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)           # columns = object axes in cam frame
    rvec = cv2.Rodrigues(R)[0].ravel()
    return R, center, rvec


def reproject(P, X3):
    """Linear reprojection (no distortion) — used INTERNALLY for the fit, which
    works in undistorted pixel space."""
    Xh = np.hstack([X3, np.ones((len(X3), 1))])
    x = (P @ Xh.T).T
    return x[:, :2] / x[:, 2:3]


# --------------------------------------------------------------------------- #
# per-frame
# --------------------------------------------------------------------------- #
def process_frame(maskL, maskR, threadL, calib, n_kp, n_fit=40, arc_fit=True):
    # Reconstruct the COMPLETE needle from (possibly multiple) occluded segments
    # via a robust ellipse-arc fit; fall back to longest-path skeleton.
    polyL = fit_arc_2d(maskL) if arc_fit else None
    if polyL is None:
        polyL = order_skeleton(maskL)
    polyR = fit_arc_2d(maskR) if arc_fit else None
    if polyR is None:
        polyR = order_skeleton(maskR)
    if polyL is None or polyR is None:
        return None, "no_centerline"

    # Orient tip(0)->tail(-1) on the LEFT view. RULE: the end nearest the suture
    # thread is the TAIL. Fall back to the swage-is-thicker cue when no thread.
    flip = tail_is_at_start(polyL, threadL)
    tip_tail_known = flip is not None
    if flip is None:
        flip = tail_by_width(polyL, maskL)
    if flip is True:
        polyL = polyL[::-1]

    # Orient the RIGHT arc consistently with the left (tip->tail). The needle is
    # one rigid arc, so pick the right orientation whose arc-length-matched
    # triangulation reprojects with the smaller error.
    uL = undistort_poly(polyL, calib["K1"], calib["D1"])
    rL = resample_arclen(uL, n_fit)
    if rL is None:
        return None, "degenerate_poly"
    best = None
    for rev in (False, True):
        pR_try = polyR[::-1] if rev else polyR
        rR = resample_arclen(undistort_poly(pR_try, calib["K2"], calib["D2"]), n_fit)
        if rR is None:
            continue
        X = triangulate(calib["P1"], calib["P2"], rL, rR)
        err = np.median(np.linalg.norm(reproject(calib["P1"], X) - rL, axis=1))
        if best is None or err < best[0]:
            best = (err, rev)
    if best is not None and best[1]:
        polyR = polyR[::-1]

    # ---- KEYPOINTS: sample EACH view's reconstructed arc at equal arc length
    # tip->tail, so both views span their OWN full needle (incl. occluded gaps),
    # and the middle points are evenly spaced along the centerline. ----
    kpL = resample_arclen(polyL, n_kp)
    kpR = resample_arclen(polyR, n_kp)
    if kpL is None or kpR is None:
        return None, "degenerate_poly"

    # 3D (best-effort) by triangulating the matched samples; radius for reporting.
    uKL = undistort_poly(kpL, calib["K1"], calib["D1"])
    uKR = undistort_poly(kpR, calib["K2"], calib["D2"])
    X3 = triangulate(calib["P1"], calib["P2"], uKL, uKR)
    try:
        center, u, v, r = fit_circle_3d(X3)
    except Exception:
        center, u, v, r = np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]), 0.0

    # visibility: is the keypoint inside the ACTUAL needle mask (not a filled gap)?
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

    visL = inside(maskL.astype(bool), kpL); visR = inside(maskR.astype(bool), kpR)
    visible = (visL | visR).astype(int)

    # confidence: stereo reprojection consistency of the matched samples
    conf = 1.0
    if _HAVE_SCIPY:
        treeL = cKDTree(uL); treeR = cKDTree(undistort_poly(polyR, calib["K2"], calib["D2"]))
        dL, _ = treeL.query(reproject(calib["P1"], X3))
        dR, _ = treeR.query(reproject(calib["P2"], X3))
        conf = float(np.exp(-np.median(np.concatenate([dL, dR])) / 5.0))

    normal = np.cross(u, v)
    R6, t6, rvec6 = pose_from_arc(X3, center, normal)
    return dict(
        xyz_mm=X3.tolist(), left=kpL.tolist(), right=kpR.tolist(),
        visible=visible.tolist(), tip_tail_known=bool(tip_tail_known),
        circle=dict(center=center.tolist(), normal=normal.tolist(), radius_mm=float(r)),
        pose=dict(R=R6.tolist(), t=t6.tolist(), rvec=rvec6.tolist()),
        conf=conf,
        polyL=polyL.tolist(), polyR=polyR.tolist(),   # reconstructed arcs (diag)
    ), "ok"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def kp_path(root, dataset, img_rel, subdir):
    rel = img_rel.replace("images/", f"{subdir}/", 1)
    return (root / dataset / rel).with_suffix(".json")


def _label(img, xy, text, color, big=False):
    """Readable label: black outline + colored text."""
    x, y = int(xy[0]), int(xy[1])
    fs = 0.9 if big else 0.5
    th = 2 if big else 1
    cv2.putText(img, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                color, th, cv2.LINE_AA)


def draw_debug(img, kps, names, visible=None, tag=None):
    n = len(kps)
    pts = np.round(np.asarray(kps)).astype(int)
    # centerline through the keypoints
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 80, 0), 2, cv2.LINE_AA)
    TIP = (0, 255, 0); TAIL = (0, 0, 255); MID = (0, 235, 255)
    for i, (x, y) in enumerate(kps):
        xi, yi = int(round(x)), int(round(y))
        vis = True if visible is None else bool(visible[i])
        if i == 0:                                   # TIP — big green ring + crosshair
            cv2.circle(img, (xi, yi), 11, (0, 0, 0), 4)
            cv2.circle(img, (xi, yi), 11, TIP, 2)
            cv2.drawMarker(img, (xi, yi), TIP, cv2.MARKER_CROSS, 16, 2)
            _label(img, (xi, yi), "TIP", TIP, big=True)
        elif i == n - 1:                             # TAIL — big red square + crosshair
            cv2.rectangle(img, (xi - 10, yi - 10), (xi + 10, yi + 10), (0, 0, 0), 4)
            cv2.rectangle(img, (xi - 10, yi - 10), (xi + 10, yi + 10), TAIL, 2)
            cv2.drawMarker(img, (xi, yi), TAIL, cv2.MARKER_TILTED_CROSS, 16, 2)
            _label(img, (xi, yi), "TAIL", TAIL, big=True)
        else:                                        # middle — numbered yellow dots
            if vis:
                cv2.circle(img, (xi, yi), 6, (0, 0, 0), -1)
                cv2.circle(img, (xi, yi), 5, MID, -1)
            else:
                cv2.circle(img, (xi, yi), 7, (0, 0, 0), 3)
                cv2.circle(img, (xi, yi), 7, MID, 2)   # ring = occluded
            _label(img, (xi, yi), str(i), MID)
    if tag:
        cv2.putText(img, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_diag(img, mask_bool, poly, tag=None):
    """Overlay the needle mask outline + the extracted skeleton (start=green tip,
    end=red tail). Shows exactly what is fed to the 3D fit."""
    cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 165, 255), 1)            # orange mask outline
    pts = np.round(np.asarray(poly)).astype(int)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 6, (0, 255, 0), -1)           # start = tip
    cv2.circle(img, tuple(pts[-1]), 6, (0, 0, 255), -1)          # end = tail
    if tag:
        cv2.putText(img, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--right-pred-dir", required=True, type=Path,
                    help="dir of right-eye predicted masks (<stem>.png) from test.py --save-preds")
    ap.add_argument("--calib", type=Path, default=Path(__file__).with_name("needle_calib.json"))
    ap.add_argument("--needle-class", type=int, default=1)
    ap.add_argument("--thread-class", type=int, default=2)
    ap.add_argument("--num-keypoints", type=int, default=5)
    ap.add_argument("--out-subdir", default="keypoints")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--diag-dir", type=Path, default=None,
                    help="save per-view needle-mask + extracted skeleton overlays "
                         "(to diagnose whether the mask/skeleton or the 3D fit is wrong)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N frames (0=all)")
    ap.add_argument("--no-arc-fit", action="store_true",
                    help="disable multi-segment ellipse-arc reconstruction (use longest-path skeleton)")
    args = ap.parse_args()

    from PIL import Image
    root = args.root.resolve()
    ds = root / args.dataset
    calib = load_calib(args.calib)
    meta = json.loads((ds / "meta.json").read_text(encoding="utf-8"))
    recs = sorted(meta["videos"][args.key], key=lambda r: r["ordinal"])
    if args.limit:
        recs = recs[:args.limit]

    # index right-eye masks by stem (works for a flat --save-preds dir OR a
    # nested annotator workspace like right_annot/masks/<key>/part_xxx/)
    right_index = {p.stem: p for p in sorted(args.right_pred_dir.rglob("*.png"))}

    kp_names = ["tip"] + [f"k{i}" for i in range(1, args.num_keypoints - 1)] + ["tail"]
    ok = skipped = 0
    for r in recs:
        stem = Path(r["image"]).stem
        lmask_p = ds / r["mask"]
        rmask_p = right_index.get(stem)
        if not lmask_p.is_file() or rmask_p is None or not rmask_p.is_file():
            skipped += 1
            continue
        ml = np.asarray(Image.open(lmask_p)); mr = np.asarray(Image.open(rmask_p))
        needleL = ml == args.needle_class
        needleR = mr == args.needle_class
        threadL = ml == args.thread_class
        if needleL.sum() < 20 or needleR.sum() < 20:
            skipped += 1
            continue
        try:
            out, status = process_frame(needleL, needleR, threadL, calib,
                                        args.num_keypoints, arc_fit=not args.no_arc_fit)
        except Exception as e:  # noqa
            out, status = None, f"error:{e}"
        if out is None:
            skipped += 1
            continue

        rec = {
            "image": r["image"], "key": args.key, "ordinal": r["ordinal"],
            "view": "left", "units": "mm",
            "needle": {
                "model": out["circle"],
                "spacing": "arclength_full",
                "tip_tail_known": out["tip_tail_known"],
                "conf": out["conf"],
                "source": "auto",
                "keypoints": [
                    {"id": i, "name": kp_names[i],
                     "x": out["left"][i][0], "y": out["left"][i][1],
                     "x_right": out["right"][i][0], "y_right": out["right"][i][1],
                     "xyz_mm": out["xyz_mm"][i], "visible": out["visible"][i]}
                    for i in range(args.num_keypoints)
                ],
            },
        }
        outp = kp_path(root, args.dataset, r["image"], args.out_subdir)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        ok += 1

        if args.debug_dir:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            imgL = cv2.imread(str(ds / r["image"]))
            imgR = cv2.imread(str(ds / "stereo_right" / args.key / f"{stem}.jpg"))
            if imgL is not None:
                draw_debug(imgL, out["left"], kp_names, out["visible"],
                           tag=f"L conf={out['conf']:.2f}")
                if imgR is not None:
                    draw_debug(imgR, out["right"], kp_names, out["visible"], tag="R")
                    if imgR.shape[0] != imgL.shape[0]:
                        sc = imgL.shape[0] / imgR.shape[0]
                        imgR = cv2.resize(imgR, (int(imgR.shape[1] * sc), imgL.shape[0]))
                    vis = cv2.hconcat([imgL, imgR])
                else:
                    vis = imgL
                cv2.imwrite(str(args.debug_dir / f"{stem}.jpg"), vis)

        if args.diag_dir:
            args.diag_dir.mkdir(parents=True, exist_ok=True)
            dL = cv2.imread(str(ds / r["image"]))
            dR = cv2.imread(str(ds / "stereo_right" / args.key / f"{stem}.jpg"))
            if dL is not None and dR is not None:
                draw_diag(dL, needleL, out["polyL"], tag="L needle mask + skeleton")
                draw_diag(dR, needleR, out["polyR"], tag="R needle mask + skeleton")
                if dR.shape[0] != dL.shape[0]:
                    sc = dL.shape[0] / dR.shape[0]
                    dR = cv2.resize(dR, (int(dR.shape[1] * sc), dL.shape[0]))
                cv2.imwrite(str(args.diag_dir / f"{stem}.jpg"), cv2.hconcat([dL, dR]))

    print(f"[needle_keypoints] {args.dataset}/{args.key}: wrote {ok}, skipped {skipped} "
          f"-> {ds / args.out_subdir / args.key}")
    if not _HAVE_SCIPY:
        print("  (scipy not found: confidence disabled — install scipy for the conf score)")
    if not _HAVE_SKIMAGE:
        print("  (skimage not found: using fallback skeletonization — install scikit-image)")


if __name__ == "__main__":
    main()
