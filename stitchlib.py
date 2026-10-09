"""
stitchlib.py -- core building blocks for Homework 1 (panoramic stitching).

CONVENTIONS (state these in the report):
  * Points are (x, y) = (column, row).  Images are OpenCV BGR arrays.
  * A homography H maps points of a SOURCE image into the REFERENCE frame:
        p_ref ~ H @ p_src        (homogeneous coordinates)
  * Warping is INVERSE warping: for every canvas pixel q we compute
        p_src = inv(H_src_to_canvas) @ q   and sample the source there.

Libraries are used ONLY for: image I/O, resizing, SIFT detection/description,
drawing.  Matching, ratio test, DLT, RANSAC, canvas bounds, inverse mapping,
sampling and compositing are all implemented here.
"""
import math
import cv2
import numpy as np


# ----------------------------------------------------------------- I/O -----
def read_bgr(path):
    """Robust image reader.
    * np.fromfile + cv2.imdecode works with non-ASCII Windows paths (e.g. Chinese folder names),
      where cv2.imread can fail.
    * Pillow is used as a fallback for files OpenCV cannot decode (EXIF rotation applied)."""
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path!r} not found. Current folder = {os.getcwd()!r}. "
                                f"Files here: {sorted(os.listdir('.'))[:20]}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image, ImageOps
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            head = data[:16].tobytes()
            raise ValueError(f"Cannot decode {path!r} (first bytes {head!r}). A real JPEG starts with "
                             f"b'\\xff\\xd8\\xff'. If it is HEIC/WebP renamed to .jpg, re-export it as JPG.") from e
    return img


def load_image(path, long_edge=800):
    """Load and downscale so the long edge <= long_edge."""
    img = read_bgr(path)
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------ features -----
def detect_sift(img, nfeatures=4000, **sift_kwargs):
    """SIFT keypoints + descriptors (library allowed). Returns (kps, pts Nx2, desc Nx128)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures, **sift_kwargs)
    kps, desc = sift.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kps], dtype=np.float64)
    return kps, pts, desc


# ------------------------------------------------------------ matching -----
def knn2(descA, descB, chunk=512):
    """For each row of descA return indices and Euclidean distances of its two
    nearest rows in descB (brute force, own implementation)."""
    A = descA.astype(np.float32)
    B = descB.astype(np.float32)
    if len(B) < 2:
        raise ValueError("Need at least 2 descriptors in B")
    bb = (B ** 2).sum(1)
    idx = np.empty((len(A), 2), dtype=np.int64)
    dist = np.empty((len(A), 2), dtype=np.float64)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        d2 = (a ** 2).sum(1)[:, None] + bb[None, :] - 2.0 * (a @ B.T)
        np.maximum(d2, 0, out=d2)
        part = np.argpartition(d2, 1, axis=1)[:, :2]
        pd = np.take_along_axis(d2, part, 1)
        o = np.argsort(pd, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, o, 1)
        dist[s:s + chunk] = np.sqrt(np.take_along_axis(pd, o, 1))
    return idx, dist


def match_descriptors(descA, descB, ratio=0.75, cross_check=False):
    """Nearest-neighbour matching A->B with Lowe's ratio test.

    Returns dict:
      nn      (NA,)   nearest neighbour in B for every descriptor of A  ("candidates")
      ratios  (NA,)   d1/d2 for every descriptor of A
      matches (M,2)   [index in A, index in B] surviving the ratio test
                      (and, if cross_check, mutual nearest neighbour)
    """
    idx, dist = knn2(descA, descB)
    ratios = dist[:, 0] / np.maximum(dist[:, 1], 1e-12)
    keep = ratios < ratio
    if cross_check:
        idxB, _ = knn2(descB, descA)
        keep &= (idxB[idx[:, 0], 0] == np.arange(len(descA)))
    ia = np.nonzero(keep)[0]
    return {"nn": idx[:, 0], "ratios": ratios,
            "matches": np.stack([ia, idx[ia, 0]], axis=1)}


# ---------------------------------------------------------- homography -----
def apply_h(H, pts):
    """Apply homography to Nx2 points."""
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def reproj_error(H, src, dst):
    """Forward transfer error in the destination frame (pixels). inf for invalid."""
    e = np.linalg.norm(apply_h(H, src) - dst, axis=1)
    e[~np.isfinite(e)] = np.inf
    return e


def normalize_points(p):
    """Hartley normalisation: centroid at origin, mean distance sqrt(2)."""
    c = p.mean(0)
    d = np.sqrt(((p - c) ** 2).sum(1)).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return (p - c) * s, T


def dlt_homography(src, dst):
    """Normalised DLT. Works for exactly 4 points (exact) or >4 (algebraic LS)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 4:
        raise ValueError("DLT needs >= 4 correspondences")
    sn, Ts = normalize_points(src)
    dn, Td = normalize_points(dst)
    x, y = sn[:, 0], sn[:, 1]
    u, v = dn[:, 0], dn[:, 1]
    A = np.zeros((2 * n, 9))
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = -x, -y, -1
    A[0::2, 6], A[0::2, 7], A[0::2, 8] = u * x, u * y, u
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = -x, -y, -1
    A[1::2, 6], A[1::2, 7], A[1::2, 8] = v * x, v * y, v
    _, _, Vt = np.linalg.svd(A)          # full SVD -> null vector also for n == 4
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H / np.linalg.norm(H)


def sample_ok(p, min_sin=0.05):
    """Reject 4-point samples with coincident or (nearly) collinear triples.
    min_sin = sine of the smallest allowed triangle angle at the first vertex (~3 deg)."""
    for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a, b = p[j] - p[i], p[k] - p[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return False
        if abs(a[0] * b[1] - a[1] * b[0]) / (na * nb) < min_sin:
            return False
    return True


def required_iters(w, confidence=0.99, s=4):
    """N = log(1-p) / log(1-w^s)  (w = best inlier fraction so far)."""
    ws = w ** s
    if ws <= 1e-12:
        return float("inf")
    if ws >= 1 - 1e-12:
        return 1
    return int(math.ceil(math.log(1 - confidence) / math.log(1 - ws)))


def ransac_homography(src, dst, thresh=3.0, max_iters=2000, adaptive=False,
                      confidence=0.99, seed=0):
    """RANSAC homography.  src/dst are Nx2; H maps src -> dst.

    Pipeline: draw 4 random correspondences -> reject degenerate samples ->
    DLT -> reject singular/ill-conditioned H -> count inliers (error < thresh)
    -> keep the hypothesis with the most inliers (ties: smaller summed inlier
    error) -> final DLT refit on ALL inliers of the best hypothesis.
    fixed budget:  loop exactly `max_iters` iterations.
    adaptive:      N is updated with required_iters() (capped by max_iters).
    """
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 4:
        raise ValueError("Need >= 4 matches")
    best = {"cnt": -1, "sum": np.inf, "H": None, "inl": None}
    N, it, degenerate = max_iters, 0, 0
    while it < N:
        it += 1
        idx = rng.choice(n, 4, replace=False)
        if not sample_ok(src[idx]) or not sample_ok(dst[idx]):
            degenerate += 1
            continue
        try:
            H = dlt_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            degenerate += 1
            continue
        if not np.all(np.isfinite(H)) or np.linalg.cond(H) > 1e9:
            degenerate += 1
            continue
        err = reproj_error(H, src, dst)
        inl = err < thresh
        cnt = int(inl.sum())
        s = float(err[inl].sum())
        if cnt > best["cnt"] or (cnt == best["cnt"] and s < best["sum"]):
            best.update(cnt=cnt, sum=s, H=H, inl=inl)
            if adaptive and cnt > 0:
                N = min(max_iters, required_iters(cnt / n, confidence))
    if best["H"] is None:
        raise RuntimeError("RANSAC found no valid hypothesis")
    inl = best["inl"]
    H_ref = dlt_homography(src[inl], dst[inl])
    return {
        "H_best": best["H"], "H": H_ref, "inliers": inl,
        "n_inliers": int(inl.sum()), "n_matches": n,
        "iters_used": it, "degenerate": degenerate,
        "err_before": float(reproj_error(best["H"], src[inl], dst[inl]).mean()),
        "err_after": float(reproj_error(H_ref, src[inl], dst[inl]).mean()),
    }


# -------------------------------------------------------------- warping -----
def image_corners(shape):
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)


def compute_canvas(shapes, Hs, max_pixels=60_000_000):
    """Canvas bounds from transformed corners.  Hs[i] maps image i -> reference.
    Returns T (offset translation, so canvas = T @ H) and (W, H) of the canvas."""
    allc = np.vstack([apply_h(H, image_corners(s)) for s, H in zip(shapes, Hs)])
    if not np.all(np.isfinite(allc)):
        raise ValueError("Non-finite canvas corner (homography maps a corner to infinity)")
    mn, mx = np.floor(allc.min(0)), np.ceil(allc.max(0))
    W, Hh = int(mx[0] - mn[0] + 1), int(mx[1] - mn[1] + 1)
    if W * Hh > max_pixels:
        raise ValueError(f"Canvas {W}x{Hh} too large -- homography likely diverges")
    T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1.0]])
    return T, (W, Hh)


def sample_image(img, sx, sy, interp="bilinear"):
    """Sample img at float coordinates (sx, sy) (own implementation).
    Returns (float32 HxWxC, bool valid mask)."""
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    sx = np.where(np.isfinite(sx), sx, -1e9)
    sy = np.where(np.isfinite(sy), sy, -1e9)
    if interp == "nearest":
        xi, yi = np.rint(sx), np.rint(sy)
        valid = (xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)
        xi = np.clip(xi, 0, w - 1).astype(np.int64)
        yi = np.clip(yi, 0, h - 1).astype(np.int64)
        out = img[yi, xi].astype(np.float32)
    elif interp == "bilinear":
        valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
        x0 = np.clip(np.floor(sx), 0, w - 2).astype(np.int64)
        y0 = np.clip(np.floor(sy), 0, h - 2).astype(np.int64)
        fx = np.clip(sx - x0, 0, 1)[..., None].astype(np.float32)
        fy = np.clip(sy - y0, 0, 1)[..., None].astype(np.float32)
        I = img.astype(np.float32)
        out = ((1 - fy) * ((1 - fx) * I[y0, x0] + fx * I[y0, x0 + 1]) +
               fy * ((1 - fx) * I[y0 + 1, x0] + fx * I[y0 + 1, x0 + 1]))
    else:
        raise ValueError(interp)
    out[~valid] = 0
    return out, valid


def warp_inverse(img, H_to_canvas, size, interp="bilinear"):
    """Inverse warp.  For every canvas pixel q: p = inv(H) q, then sample img at p.
    Returns (warped float32, valid mask)."""
    W, Hh = size
    Hinv = np.linalg.inv(H_to_canvas)
    xs, ys = np.meshgrid(np.arange(W), np.arange(Hh))
    P = np.stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64),
                  np.ones(xs.size)])
    S = Hinv @ P
    # third component of inv(H) q equals 1 / w_forward -> points "behind" the
    # image plane have the opposite sign of the image centre; reject them.
    h, w = img.shape[:2]
    wc = (H_to_canvas @ np.array([w / 2, h / 2, 1.0]))[2]
    front = S[2] * np.sign(wc) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        sx, sy = S[0] / S[2], S[1] / S[2]
    sx[~front] = -1e9
    out, valid = sample_image(img, sx.reshape(Hh, W), sy.reshape(Hh, W), interp)
    return out, valid


def composite_overwrite(warped, masks):
    """Simple overwrite compositing: later images overwrite earlier ones."""
    out = np.zeros_like(warped[0])
    occ = np.zeros(masks[0].shape, bool)
    for w, m in zip(warped, masks):
        out[m] = w[m]
        occ |= m
    return out, occ


def to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# -------------------------------------------------------------- drawing -----
PALETTE = [(0, 0, 255), (0, 200, 0), (255, 80, 0), (0, 200, 255),
           (255, 0, 255), (0, 128, 255), (200, 200, 0), (128, 0, 128)]


def draw_keypoints(img, pts, color=(0, 255, 0), r=3):
    out = img.copy()
    for x, y in pts:
        cv2.circle(out, (int(round(x)), int(round(y))), r, color, 1, cv2.LINE_AA)
    return out


def draw_matches(imgA, imgB, ptsA, ptsB, inlier_mask=None, max_lines=None, seed=0):
    """Side-by-side match lines. With a mask: inliers green, outliers red;
    without: all yellow."""
    h = max(imgA.shape[0], imgB.shape[0])
    wA = imgA.shape[1]
    canvas = np.zeros((h, wA + imgB.shape[1], 3), np.uint8)
    canvas[:imgA.shape[0], :wA] = imgA
    canvas[:imgB.shape[0], wA:] = imgB
    idx = np.arange(len(ptsA))
    if max_lines and len(idx) > max_lines:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_lines, replace=False))
    for i in idx:
        if inlier_mask is None:
            col = (0, 255, 255)
        else:
            col = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        p = (int(round(ptsA[i][0])), int(round(ptsA[i][1])))
        q = (int(round(ptsB[i][0] + wA)), int(round(ptsB[i][1])))
        cv2.line(canvas, p, q, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 2, col, -1)
        cv2.circle(canvas, q, 2, col, -1)
    return canvas


def draw_boundaries(canvas, shapes, H_to_canvas_list, labels=None, thickness=3):
    """Draw each input image's boundary (transformed corners, same H used for
    compositing) in a different colour."""
    out = canvas.copy()
    for k, (s, H) in enumerate(zip(shapes, H_to_canvas_list)):
        c = apply_h(H, image_corners(s))
        poly = np.round(c).astype(np.int32).reshape(-1, 1, 2)
        col = PALETTE[k % len(PALETTE)]
        cv2.polylines(out, [poly], True, col, thickness, cv2.LINE_AA)
        if labels:
            cv2.putText(out, labels[k], tuple(poly[0, 0] + [8, 28]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    return out


def hstack_pad(imgs, pad=10):
    h = max(i.shape[0] for i in imgs)
    cols = []
    for i in imgs:
        c = np.full((h, i.shape[1] + pad, 3), 255, np.uint8)
        c[:i.shape[0], :i.shape[1]] = i
        cols.append(c)
    return np.hstack(cols)