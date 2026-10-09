"""
extras.py -- scripts for the additional features.  One file, six sub-commands:

  python extras.py ls     --pair A.jpg B.jpg                 Least Squares vs RANSAC        (5 pts)
  python extras.py interp --pair A.jpg B.jpg                 Interpolation comparison       (5 pts)
  python extras.py sweep  --pair set1 A B --pair set2 A B    RANSAC threshold analysis      (10 pts)
  python extras.py focal  --images i1 i2 i3 i4 i5            Focal length from pivoted views(10 pts)
  python extras.py cyl    --images i1 i2 i3 i4 i5            Cylindrical vs flat panorama   (10 pts)
  python extras.py gain   --pair A.jpg B.jpg                 Exposure / gain compensation   (10 pts)

Common options: --out out --ratio 0.75 --thresh 3 --iters 2000 --long_edge 800 --nfeatures 4000
Results go to  <out>/extras/<cmd>_<name>/  (PNG figures + result.json); make_dashboard.py shows them.
Library use: image I/O, SIFT, drawing, matplotlib for plots.  Matching, homography, RANSAC, warping,
cylindrical mapping, focal-length fit and gain estimation are all implemented here / in stitchlib.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

from stitchlib import (load_image, detect_sift, match_descriptors, ransac_homography,
                       dlt_homography, compute_canvas, warp_inverse, composite_overwrite,
                       sample_image, reproj_error, apply_h, image_corners, to_u8,
                       draw_matches, draw_boundaries)


# ============================================================ helpers ====
def add_common(ap):
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default=None, help="output sub-folder name")
    ap.add_argument("--long_edge", type=int, default=800)
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)


def outdir(a, cmd, default):
    d = os.path.join(a.out, "extras", f"{cmd}_{a.name or default}")
    os.makedirs(d, exist_ok=True)
    return d


def _np(o):
    return o.item() if hasattr(o, "item") else str(o)


def _dumps(o, **kw):
    return json.dumps(o, default=_np, **kw)


def save_json(d, obj):
    json.dump(obj, open(os.path.join(d, "result.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


def banner(im, text, h=40, scale=0.7):
    bar = np.full((h, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def hcat(imgs, pad=12):
    h = max(i.shape[0] for i in imgs)
    cols = []
    for i in imgs:
        c = np.full((h, i.shape[1] + pad, 3), 255, np.uint8)
        c[:i.shape[0], :i.shape[1]] = i
        cols.append(c)
    return np.hstack(cols)


def vcat(imgs, pad=12):
    w = max(i.shape[1] for i in imgs)
    rows = []
    for i in imgs:
        r = np.full((i.shape[0] + pad, w, 3), 255, np.uint8)
        r[:i.shape[0], :i.shape[1]] = i
        rows.append(r)
    return np.vstack(rows)


def fit_h(im, h):
    return cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)


def load_features(paths, a):
    imgs = [load_image(p, a.long_edge) for p in paths]
    feats = [detect_sift(im, a.nfeatures) for im in imgs]
    return imgs, feats


def pair_matches(feats, i, j, a):
    """Ratio-test matches between image i (reference) and j. Returns src (in j), dst (in i)."""
    m = match_descriptors(feats[i][2], feats[j][2], a.ratio)["matches"]
    return feats[j][1][m[:, 1]], feats[i][1][m[:, 0]]


def pano_overwrite(A, B, H, T=None, size=None, interp="bilinear", scale=1.0):
    """Two-image overwrite panorama. H maps B -> A. Returns pano(float32), dict."""
    if T is None:
        T, size = compute_canvas([A.shape, B.shape], [np.eye(3), H])
    S = np.diag([scale, scale, 1.0])
    T2 = S @ T
    size2 = (int(math.ceil(size[0] * scale)), int(math.ceil(size[1] * scale)))
    Hc = [T2, T2 @ H]
    ws, ms = zip(*[warp_inverse(im, h, size2, interp) for im, h in zip((A, B), Hc)])
    pano, _ = composite_overwrite(list(ws), list(ms))
    return pano, {"T": T, "size": size, "Hc": Hc, "masks": ms, "size2": size2}


def best_crop(diff, win=110):
    """Top-left corner of the win x win window with the largest mean difference."""
    d = cv2.boxFilter(diff.astype(np.float32), -1, (win, win), normalize=True)
    h, w = d.shape
    d[:win // 2, :] = 0
    d[:, :win // 2] = 0
    d[h - win // 2:, :] = 0
    d[:, w - win // 2:] = 0
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return max(0, x - win // 2), max(0, y - win // 2)


def crop(im, x0, y0, win, up=1):
    c = im[y0:y0 + win, x0:x0 + win]
    return cv2.resize(c, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST) if up > 1 else c


def feather_weight(shape):
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    return np.minimum.reduce([xs, w - 1 - xs, ys, h - 1 - ys]).astype(np.float32) + 1e-3


# ===================================================== 1. LS vs RANSAC ====
def cmd_ls(a):
    d = outdir(a, "ls", os.path.splitext(os.path.basename(a.pair[0]))[0])
    A, B = load_image(a.pair[0], a.long_edge), load_image(a.pair[1], a.long_edge)
    _, pA, dA = detect_sift(A, a.nfeatures)
    _, pB, dB = detect_sift(B, a.nfeatures)
    m = match_descriptors(dA, dB, a.ratio)["matches"]
    src, dst = pB[m[:, 1]], pA[m[:, 0]]
    r = ransac_homography(src, dst, a.thresh, a.iters, seed=a.seed)
    H_r, H_l = r["H"], dlt_homography(src, dst)          # LS = normalised DLT on ALL matches
    e_r, e_l = reproj_error(H_r, src, dst), reproj_error(H_l, src, dst)
    inl = r["inliers"]

    T, size = compute_canvas([A.shape, B.shape], [np.eye(3), H_r])
    pano_r, _ = pano_overwrite(A, B, H_r, T, size)
    pano_l, info_l = pano_overwrite(A, B, H_l, T, size)      # same canvas for a fair comparison
    diff = np.abs(pano_r - pano_l).mean(2)
    # crop on the SEAM of the least-squares panorama (the border of image B inside image A), at its most
    # textured place: this is where a wrong homography shows up as a discontinuity. The RANSAC panorama
    # is cropped at exactly the same place.
    mA, mB = (m.astype(np.uint8) for m in info_l["masks"])
    edge = mB - cv2.erode(mB, np.ones((3, 3), np.uint8))
    seam = (edge > 0) & (cv2.dilate(mA, np.ones((9, 9), np.uint8)) > 0)
    g = cv2.cvtColor(to_u8(pano_l), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    score = cv2.boxFilter(grad, -1, (61, 61), normalize=True) * seam
    for sl in (np.s_[:60, :], np.s_[:, :60], np.s_[-60:, :], np.s_[:, -60:]):
        score[sl] = 0
    if score.max() > 0:
        yy, xx = np.unravel_index(int(np.argmax(score)), score.shape)
        x0, y0 = max(0, xx - 60), max(0, yy - 60)
    else:
        x0, y0 = best_crop(diff, 120)

    def stats(e, mask=None):
        e = e if mask is None else e[mask]
        return {"mean": float(np.mean(e)), "median": float(np.median(e)),
                "within_thr": int((e < a.thresh).sum()), "max": float(np.max(e))}
    res = {"n_matches": len(src), "ransac_inliers": int(inl.sum()), "thresh": a.thresh,
           "ransac_H": H_r.tolist(), "ls_H": H_l.tolist(),
           "all_matches": {"ransac": stats(e_r), "ls": stats(e_l)},
           "on_ransac_inliers": {"ransac": stats(e_r, inl), "ls": stats(e_l, inl)},
           "on_ransac_outliers": {"ransac": stats(e_r, ~inl) if (~inl).any() else None,
                                  "ls": stats(e_l, ~inl) if (~inl).any() else None},
           "mean_abs_pixel_diff_between_panoramas": float(diff[diff > 0].mean()) if (diff > 0).any() else 0.0}
    save_json(d, res)

    cv2.imwrite(f"{d}/inliers.png", draw_matches(A, B, dst, src, inl, 400))
    pr, pl = to_u8(pano_r), to_u8(pano_l)
    # footprint of image B on the canvas (area relative to its original size): shows how strongly LS magnifies B
    def foot(H):
        p = apply_h(H, image_corners(B.shape))
        return float(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
                     / (B.shape[0] * B.shape[1]))
    res["footprint_area_of_B_relative_to_original"] = {"ransac": foot(H_r), "ls": foot(H_l)}
    save_json(d, res)
    pr, pl = pr.copy(), pl.copy()
    cv2.imwrite(f"{d}/panorama_ransac.png", pr)
    cv2.imwrite(f"{d}/panorama_ls.png", pl)
    sh = 600
    prb, plb = pr.copy(), pl.copy()
    for im_ in (prb, plb):
        cv2.rectangle(im_, (x0, y0), (x0 + 120, y0 + 120), (0, 0, 255), 4)
    pr_, pl_ = prb, plb
    cv2.imwrite(f"{d}/panoramas_side_by_side.png", hcat([
        banner(fit_h(pr_, sh), f"RANSAC ({int(inl.sum())}/{len(src)} inliers)"),
        banner(fit_h(pl_, sh), f"Least squares (all {len(src)} matches); red box = crop")]))
    big = lambda im: cv2.resize(crop(im, x0, y0, 120), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    dd = np.clip(crop(diff, x0, y0, 120) * 4, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{d}/crop_compare.png", hcat([
        banner(big(pr), "RANSAC crop"), banner(big(pl), "LS crop"),
        banner(cv2.resize(cv2.cvtColor(dd, cv2.COLOR_GRAY2BGR), None, fx=4, fy=4,
                          interpolation=cv2.INTER_NEAREST), "|difference| x4")]))
    print(_dumps({k: res[k] for k in ("n_matches", "ransac_inliers", "all_matches",
                                          "on_ransac_inliers")}, indent=1))
    print("saved to", d)


# ================================================ 2. interpolation ====
def cmd_interp(a):
    d = outdir(a, "interp", os.path.splitext(os.path.basename(a.pair[0]))[0])
    A, B = load_image(a.pair[0], a.long_edge), load_image(a.pair[1], a.long_edge)
    _, pA, dA = detect_sift(A, a.nfeatures)
    _, pB, dB = detect_sift(B, a.nfeatures)
    m = match_descriptors(dA, dB, a.ratio)["matches"]
    r = ransac_homography(pB[m[:, 1]], pA[m[:, 0]], a.thresh, a.iters, seed=a.seed)
    pn, info = pano_overwrite(A, B, r["H"], interp="nearest", scale=a.zoom)
    pb, _ = pano_overwrite(A, B, r["H"], interp="bilinear", scale=a.zoom)
    diff = np.abs(pn - pb).mean(2)
    valid = diff >= 0
    x0, y0 = best_crop(diff, a.win)
    up = max(1, 480 // a.win)
    un, ub = to_u8(pn), to_u8(pb)
    dd = np.clip(crop(diff, x0, y0, a.win) * 6, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{d}/crop_compare.png", hcat([
        banner(crop(un, x0, y0, a.win, up), "nearest-neighbour"),
        banner(crop(ub, x0, y0, a.win, up), "bilinear"),
        banner(cv2.resize(cv2.cvtColor(dd, cv2.COLOR_GRAY2BGR), None, fx=up, fy=up,
                          interpolation=cv2.INTER_NEAREST), "|difference| x6")]))
    full = ub.copy()
    cv2.rectangle(full, (x0, y0), (x0 + a.win, y0 + a.win), (0, 0, 255), 3)
    cv2.imwrite(f"{d}/panorama_with_crop_box.png", full)
    cv2.imwrite(f"{d}/panorama_nearest.png", un)
    cv2.imwrite(f"{d}/panorama_bilinear.png", ub)
    cv2.imwrite(f"{d}/difference_x6.png", np.clip(diff * 6, 0, 255).astype(np.uint8))
    covered = (info["masks"][0] | info["masks"][1])
    dv = diff[covered]
    res = {"zoom": a.zoom, "canvas": list(info["size2"]), "crop_xy": [int(x0), int(y0)], "crop_win": a.win,
           "mean_abs_diff_all_covered": float(dv.mean()), "p99_abs_diff": float(np.percentile(dv, 99)),
           "max_abs_diff": float(dv.max()), "frac_pixels_diff_gt_5": float((dv > 5).mean()),
           "mean_abs_diff_in_crop": float(crop(diff, x0, y0, a.win).mean())}
    save_json(d, res)
    print(_dumps(res, indent=1))
    print("saved to", d)


# ========================================== 3. RANSAC parameter sweep ====
def cmd_sweep(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = outdir(a, "sweep", "_".join(p[0] for p in a.pair))
    thr_list = [float(t) for t in a.thresholds.split(",")]
    results, panels, crop_pos = {}, [], {}
    for name, pa, pb in a.pair:
        A, B = load_image(pa, a.long_edge), load_image(pb, a.long_edge)
        _, pA, dA = detect_sift(A, a.nfeatures)
        _, pB, dB = detect_sift(B, a.nfeatures)
        m = match_descriptors(dA, dB, a.ratio)["matches"]
        src, dst = pB[m[:, 1]], pA[m[:, 0]]
        base = ransac_homography(src, dst, a.thresh, a.iters, seed=a.seed)
        ref = base["inliers"]                       # fixed reference set = inliers of the baseline thr
        rows, Hs = [], {}
        for t in thr_list:
            r = ransac_homography(src, dst, t, a.iters, seed=a.seed)
            e_ref = reproj_error(r["H"], src[ref], dst[ref])
            rows.append({"thr": t, "inliers": r["n_inliers"], "inlier_ratio": r["n_inliers"] / len(src),
                         "mean_err_inliers": r["err_after"], "mean_err_on_reference_set": float(e_ref.mean()),
                         "max_err_on_reference_set": float(e_ref.max())})
            Hs[t] = r["H"]
        results[name] = {"n_matches": len(src), "reference_thr": a.thresh,
                         "reference_set_size": int(ref.sum()), "rows": rows}
        # representative panorama crops at 3 thresholds
        sel = [thr_list[0], a.thresh if a.thresh in Hs else thr_list[len(thr_list) // 2], thr_list[-1]]
        T, size = compute_canvas([A.shape, B.shape], [np.eye(3), Hs[a.thresh] if a.thresh in Hs else base["H"]])
        panos = [pano_overwrite(A, B, Hs[t], T, size)[0] for t in sel]
        if a.crop_mode == "seam":
            # crop on the SEAM (border of image B inside image A), where a misalignment becomes visible,
            # at the most textured place of the seam
            _, inf = pano_overwrite(A, B, Hs[a.thresh] if a.thresh in Hs else base["H"], T, size)
            mA, mB = (m.astype(np.uint8) for m in inf["masks"])
            edge = mB - cv2.erode(mB, np.ones((3, 3), np.uint8))
            seam = (edge > 0) & (cv2.dilate(mA, np.ones((9, 9), np.uint8)) > 0)
            g = cv2.cvtColor(to_u8(panos[1]), cv2.COLOR_BGR2GRAY).astype(np.float32)
            grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
            score = cv2.boxFilter(grad, -1, (45, 45), normalize=True) * seam
            score[:45, :] = 0
            score[:, :45] = 0
            score[-45:, :] = 0
            score[:, -45:] = 0
            if score.max() > 0:
                yy, xx = np.unravel_index(int(np.argmax(score)), score.shape)
                x0, y0 = max(0, xx - 45), max(0, yy - 45)
            else:
                diff = np.abs(panos[0] - panos[2]).mean(2)
                x0, y0 = best_crop(diff, 90)
        else:
            diff = np.maximum(np.abs(panos[0] - panos[1]).mean(2), np.abs(panos[2] - panos[1]).mean(2))
            x0, y0 = best_crop(diff, 90)
        crop_pos[name] = [int(x0), int(y0)]
        panels.append(banner(hcat([banner(cv2.resize(crop(to_u8(p), x0, y0, 90), None, fx=4, fy=4,
                                                       interpolation=cv2.INTER_NEAREST), f"thr = {t:g} px")
                                  for p, t in zip(panos, sel)]), f"{name}: same crop, three thresholds"))
    cv2.imwrite(f"{d}/crops.png", vcat(panels))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, rs in results.items():
        t = [r["thr"] for r in rs["rows"]]
        ax[0].plot(t, [r["inliers"] for r in rs["rows"]], "o-", label=name)
        ax[1].plot(t, [r["mean_err_inliers"] for r in rs["rows"]], "o-", label=name)
        ax[2].plot(t, [r["mean_err_on_reference_set"] for r in rs["rows"]], "o-", label=name)
    for x, ttl, yl in zip(ax, ["Inlier count", "Mean error of its own inliers (after refit)",
                               "Mean error on a FIXED reference set"], ["inliers", "px", "px"]):
        x.set_xscale("log")
        x.set_xlabel("inlier threshold (px)")
        x.set_ylabel(yl)
        x.set_title(ttl)
        x.grid(alpha=.3)
        x.legend()
    plt.tight_layout()
    plt.savefig(f"{d}/sweep_plot.png", dpi=130)
    results["_crop_positions_xy"] = crop_pos
    save_json(d, results)
    results.pop("_crop_positions_xy")
    for name, rs in results.items():
        print(name, f"(matches={rs['n_matches']}, reference set={rs['reference_set_size']})")
        for r in rs["rows"]:
            print(f"  thr={r['thr']:5g}  inliers={r['inliers']:4d}  err_own={r['mean_err_inliers']:.3f}  "
                  f"err_ref={r['mean_err_on_reference_set']:.3f}  max_ref={r['max_err_on_reference_set']:.2f}")
    print("saved to", d)


# ================================================= 4. focal length ====
def ortho_error(H, f, cx, cy):
    """For a pure rotation H ~ K R K^-1.  Returns || M^T M - I ||_F with M = K^-1 H K scaled to det 1."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ H @ K
    dt = np.linalg.det(M)
    if abs(dt) < 1e-12:
        return float("inf")
    M = M / np.cbrt(dt)
    return float(np.linalg.norm(M.T @ M - np.eye(3)))


def _golden(fun, lo, hi, it=70):
    g = (math.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = fun(c), fun(d)
    for _ in range(it):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = fun(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = fun(d)
    return (lo + hi) / 2


def minimise_f(fun, lo=100.0, hi=6000.0, n=500):
    fs = np.logspace(math.log10(lo), math.log10(hi), n)
    errs = np.array([fun(f) for f in fs])
    k = int(np.argmin(errs))
    a_, b_ = math.log(fs[max(k - 1, 0)]), math.log(fs[min(k + 1, n - 1)])
    lf = _golden(lambda x: fun(math.exp(x)), a_, b_)
    return math.exp(lf), fs, errs


def estimate_focal(H, cx, cy):
    f, fs, errs = minimise_f(lambda f: ortho_error(H, f, cx, cy))
    return f, ortho_error(H, f, cx, cy), fs, errs


def rotation_angle_deg(H, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ H @ K
    M = M / np.cbrt(np.linalg.det(M))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def read_exif_f35(path):
    """35mm-equivalent focal length from EXIF (None if the file has no EXIF)."""
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        v = ex.get_ifd(0x8769).get(41989)
        return float(v) if v else None
    except Exception:
        return None


def cmd_focal(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = outdir(a, "focal", "pivot")
    imgs, feats = load_features(a.images, a)
    h, w = imgs[0].shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    diag = math.hypot(w, h)
    exif = [read_exif_f35(p) for p in a.images]
    f35 = next((v for v in exif if v), a.f35)
    f_expected = f35 / 43.2666 * diag
    rng = np.random.default_rng(a.seed)
    rows, curves = [], {}
    n = len(imgs)
    for i in range(n):
        for j in range(i + 1, n):
            src, dst = pair_matches(feats, i, j, a)
            if len(src) < 12:
                continue
            r = ransac_homography(src, dst, a.thresh, a.iters, seed=a.seed)
            if r["n_inliers"] < a.min_inliers:
                continue
            H = r["H"]                                   # maps image j -> image i
            f, err, fs, errs = estimate_focal(H, cx, cy)
            ang = rotation_angle_deg(H, f, cx, cy)
            idx = np.nonzero(r["inliers"])[0]
            boot = []
            for _ in range(a.boot):
                s = rng.choice(idx, max(8, len(idx) // 2), replace=False)
                boot.append(estimate_focal(dlt_homography(src[s], dst[s]), cx, cy)[0])
            name = f"{i + 1}-{j + 1}"
            rows.append({"pair": name, "inliers": r["n_inliers"], "f_px": f, "fit_error": err,
                         "rotation_deg": ang, "boot_median": float(np.median(boot)),
                         "boot_std": float(np.std(boot)), "H": H.tolist()})
            curves[name] = (fs, errs)
            print(f"pair {name}: inliers={r['n_inliers']:4d}  f={f:7.1f}px  fit_err={err:.4f}  "
                  f"rot={ang:5.1f} deg  bootstrap f = {np.median(boot):.1f} +- {np.std(boot):.1f}")
    if len(rows) < 2:
        raise SystemExit("Need at least two usable pairs.")
    Hs = [np.array(r["H"]) for r in rows]
    f_joint, fs_j, errs_j = minimise_f(lambda f: sum(ortho_error(H, f, cx, cy) ** 2 for H in Hs))
    fvals = np.array([r["f_px"] for r in rows])
    ang = np.array([r["rotation_deg"] for r in rows])
    big = ang >= a.min_angle                              # small rotations make f ill-conditioned
    summary = {"image_size_wh": [w, h], "f35_mm_used": f35, "f35_from_exif": any(exif),
               "f_expected_px": f_expected, "hfov_expected_deg": math.degrees(2 * math.atan(w / (2 * f_expected))),
               "f_joint_px": f_joint, "hfov_joint_deg": math.degrees(2 * math.atan(w / (2 * f_joint))),
               "f_median_all_pairs": float(np.median(fvals)),
               "f_median_pairs_with_rotation_ge_min": float(np.median(fvals[big])) if big.any() else None,
               "f_std_all_pairs": float(np.std(fvals)), "min_angle_deg": a.min_angle,
               "f_px_at_original_resolution_scale": "multiply by (original long edge / working long edge)"}
    save_json(d, {"summary": summary, "pairs": rows})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, (fs, errs) in curves.items():
        ax[0].plot(fs, errs, label=name)
    ax[0].axvline(f_expected, color="k", ls="--", label=f"{f35:g} mm equiv. -> {f_expected:.0f}px")
    ax[0].axvline(f_joint, color="r", ls=":", label=f"joint fit {f_joint:.0f}px")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("candidate focal length f (px)")
    ax[0].set_ylabel("rotation-orthogonality error")
    ax[0].set_title("Fit error per image pair")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    names = [r["pair"] for r in rows]
    ax[1].errorbar(range(len(rows)), fvals, yerr=[r["boot_std"] for r in rows], fmt="o", capsize=3)
    ax[1].axhline(f_expected, color="k", ls="--")
    ax[1].axhline(f_joint, color="r", ls=":")
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels([f"{n_}\n{g:.0f} deg" for n_, g in zip(names, ang)], fontsize=7)
    ax[1].set_ylabel("f (px), error bar = bootstrap std")
    ax[1].set_title("Estimate per pair (label: pair / rotation angle)")
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{d}/focal_plot.png", dpi=130)
    print(_dumps(summary, indent=1))
    print("saved to", d)


# ============================================ 5. cylindrical panorama ====
def cyl_dims(w, h, f):
    return int(math.ceil(2 * f * math.atan((w / 2) / f))), h


def cyl_warp(img, f, interp="bilinear"):
    """Inverse cylindrical warp (own implementation).  Output pixel (u, v) <-> ray
    (sin t, v/f, cos t) with t = u/f ; image point x = f tan t + cx, y = v / cos t + cy."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    Wc, Hc = cyl_dims(w, h, f)
    u, v = np.meshgrid(np.arange(Wc), np.arange(Hc))
    uu, vv = u - (Wc - 1) / 2, v - (Hc - 1) / 2
    t = uu / f
    x = f * np.tan(t) + cx
    y = vv / np.cos(t) + cy
    return sample_image(img, x, y, interp)


def cyl_coords(pts, f, w, h):
    """Forward map of image points to cylinder-image coordinates."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    Wc, Hc = cyl_dims(w, h, f)
    xp, yp = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([f * np.arctan2(xp, f) + (Wc - 1) / 2,
                     f * yp / np.sqrt(xp ** 2 + f ** 2) + (Hc - 1) / 2], 1)


def best_translation(pi, pj, thr):
    """RANSAC with 1-point samples: t = pi - pj. Returns (t, inlier mask)."""
    t = pi - pj
    best, bm = -1, None
    for k in range(len(t)):
        m = np.linalg.norm(t - t[k], axis=1) < thr
        if m.sum() > best:
            best, bm = int(m.sum()), m
    return t[bm].mean(0), bm


def blend_feather(layers, weights, offsets, size):
    W, H = size
    num = np.zeros((H, W, 3), np.float32)
    den = np.zeros((H, W), np.float32)
    for im, wt, (ox, oy) in zip(layers, weights, offsets):
        h, w = im.shape[:2]
        num[oy:oy + h, ox:ox + w] += im * wt[..., None]
        den[oy:oy + h, ox:ox + w] += wt
    return np.where(den[..., None] > 0, num / np.maximum(den[..., None], 1e-6), 0)


def cmd_cyl(a):
    d = outdir(a, "cyl", "seq")
    imgs, feats = load_features(a.images, a)
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    ref = n // 2
    # --- consecutive homographies (flat model) and matches
    Hp, M = [], []
    for i in range(n - 1):
        src, dst = pair_matches(feats, i, i + 1, a)
        r = ransac_homography(src, dst, a.thresh, a.iters, seed=a.seed)
        Hp.append(r["H"])
        M.append((src, dst))
    # --- cylinder scale f
    if a.f == "auto":
        f, _, _ = minimise_f(lambda ff: sum(ortho_error(H, ff, cx, cy) ** 2 for H in Hp))
        how = "estimated from the consecutive homographies (joint fit)"
    else:
        f, how = float(a.f), "given on the command line"
    print(f"cylinder scale f = {f:.1f} px  ({how});  horizontal FOV of one image = "
          f"{math.degrees(2 * math.atan(w / (2 * f))):.1f} deg")

    # --- flat panorama (chained H) with feather blending
    Hs = [None] * n
    Hs[ref] = np.eye(3)
    for k in range(ref + 1, n):
        Hs[k] = Hs[k - 1] @ Hp[k - 1]
    for k in range(ref - 1, -1, -1):
        Hs[k] = Hs[k + 1] @ np.linalg.inv(Hp[k])
    Hs = [H / H[2, 2] for H in Hs]
    T, fsize = compute_canvas([im.shape for im in imgs], Hs, max_pixels=80_000_000)
    Hc = [T @ H for H in Hs]
    wl, ml, wts = [], [], []
    for im, H in zip(imgs, Hc):
        x, m = warp_inverse(im, H, fsize)
        wt, _ = warp_inverse(feather_weight(im.shape), H, fsize)
        wl.append(x)
        wts.append(wt[..., 0] * m)
    flat = sum(wt[..., None] * x for wt, x in zip(wts, wl)) / np.maximum(sum(wts)[..., None], 1e-6)
    # --- cylindrical panorama
    cyl = [cyl_warp(im, f) for im in imgs]
    cwarp = [c[0] for c in cyl]
    # weights: distance to the border of the warped source image (map the source weight too)
    cw = [cyl_warp(feather_weight(im.shape), f)[0][..., 0] * c[1] for im, c in zip(imgs, cyl)]
    tp, resid = [], []
    for i in range(n - 1):
        src, dst = M[i]
        pc_i, pc_j = cyl_coords(dst, f, w, h), cyl_coords(src, f, w, h)
        t, mk = best_translation(pc_i, pc_j, a.thresh)
        tp.append(t)
        resid.append(float(np.linalg.norm(pc_i[mk] - (pc_j[mk] + t), axis=1).mean()))
    off = [np.zeros(2) for _ in range(n)]
    for k in range(ref + 1, n):
        off[k] = off[k - 1] + tp[k - 1]
    for k in range(ref - 1, -1, -1):
        off[k] = off[k + 1] - tp[k]
    mn = np.min(off, axis=0)
    ints = [np.round(o - mn).astype(int) for o in off]
    Wc, Hc_ = cyl_dims(w, h, f)
    csize = (max(i[0] for i in ints) + Wc, max(i[1] for i in ints) + Hc_)
    cpano = blend_feather(cwarp, cw, ints, csize)
    # --- closure check on non-consecutive pairs (i, i+2): how far does the chain drift?
    drift = []
    for i in range(n - 2):
        src, dst = pair_matches(feats, i, i + 2, a)
        if len(src) < 12:
            continue
        t, mk = best_translation(cyl_coords(dst, f, w, h), cyl_coords(src, f, w, h), a.thresh)
        if mk.sum() < 15:
            continue
        pred = off[i + 2] - off[i]
        drift.append({"pair": f"{i + 1}-{i + 3}", "direct_t": t.tolist(), "chained_t": pred.tolist(),
                      "inliers": int(mk.sum()),
                      "drift_px": float(np.linalg.norm(t - pred))})
    # footprint areas of the flat panorama
    area = [float(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / (w * h))
            for p in [apply_h(H, image_corners(im.shape)) for H, im in zip(Hs, imgs)]]
    total_fov = math.degrees(csize[0] / f)
    res = {"n_images": n, "reference_image": ref + 1, "cylinder_f_px": f, "f_how": how,
           "single_image_hfov_deg": math.degrees(2 * math.atan(w / (2 * f))),
           "total_hfov_deg_cylinder_canvas": total_fov,
           "flat_canvas": list(fsize), "cyl_canvas": list(csize),
           "flat_footprint_area_scale": area, "cyl_translation_per_pair_px": [t.tolist() for t in tp],
           "cyl_mean_match_residual_px": resid, "closure_drift_non_consecutive": drift}
    save_json(d, res)
    fu, cu = to_u8(flat), to_u8(cpano)
    cv2.imwrite(f"{d}/flat.png", fu)
    cv2.imwrite(f"{d}/cylindrical.png", cu)
    sh = 520
    cv2.imwrite(f"{d}/flat_vs_cylindrical.png", vcat([
        banner(fit_h(fu, sh), f"Flat (homography chain), canvas {fsize[0]}x{fsize[1]}"),
        banner(fit_h(cu, sh), f"Cylindrical (f = {f:.0f} px), canvas {csize[0]}x{csize[1]}")]))
    cv2.imwrite(f"{d}/cylindrical_annotated.png", draw_boundaries(
        cu, [(Hc_, Wc)] * n, [np.array([[1, 0, ints[k][0]], [0, 1, ints[k][1]], [0, 0, 1.0]]) for k in range(n)],
        [str(k + 1) + (" (ref)" if k == ref else "") for k in range(n)]))
    print(_dumps({k: res[k] for k in ("single_image_hfov_deg", "total_hfov_deg_cylinder_canvas", "flat_canvas",
                                           "cyl_canvas", "cyl_mean_match_residual_px", "closure_drift_non_consecutive")},
                     indent=1))
    print("saved to", d)


# ======================================== 6. exposure / gain compensation ====
SRGB_GAMMA = 2.2


def to_lin(x):
    return (np.clip(x, 0, 255) / 255.0) ** SRGB_GAMMA * 255.0


def from_lin(x):
    return (np.clip(x, 0, 255) / 255.0) ** (1 / SRGB_GAMMA) * 255.0


def solve_gains(means, counts, sig_n=10.0, sig_g=0.1):
    """Brown & Lowe style gain solve.  means[(i,j)] = mean of image i inside overlap(i,j).
    Minimise  sum_ij N_ij[ (g_i I_ij - g_j I_ji)^2 / sig_n^2 + (1-g_i)^2 / sig_g^2 ]  (linear system)."""
    n = 1 + max(max(k) for k in means)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for (i, j), mij in means.items():
        N = counts[(i, j)]
        mji = means[(j, i)]
        A[i, i] += N * (mij ** 2 / sig_n ** 2 + 1 / sig_g ** 2)
        A[i, j] -= N * mij * mji / sig_n ** 2
        b[i] += N / sig_g ** 2
    return np.linalg.solve(A, b)


def cmd_gain(a):
    d = outdir(a, "gain", os.path.splitext(os.path.basename(a.pair[0]))[0])
    A, B = load_image(a.pair[0], a.long_edge), load_image(a.pair[1], a.long_edge)
    if a.simulate:
        print("!! SIMULATED exposure change (B scaled by %.2f in linear light): for testing the code only. "
              "The assignment requires a REAL exposure-mismatched pair." % a.simulate)
        B = to_u8(from_lin(to_lin(B.astype(np.float32)) * a.simulate))
    _, pA, dA = detect_sift(A, a.nfeatures)
    _, pB, dB = detect_sift(B, a.nfeatures)
    m = match_descriptors(dA, dB, a.ratio)["matches"]
    src, dst = pB[m[:, 1]], pA[m[:, 0]]
    r = ransac_homography(src, dst, a.thresh, a.iters, seed=a.seed)
    print(f"matches={len(src)}  inliers={r['n_inliers']}")
    if r["n_inliers"] < 30:
        print("WARNING: very few inliers -> the geometry is not reliable; fix the photos first "
              "(more overlap, textured overlap region).")
    T, size = compute_canvas([A.shape, B.shape], [np.eye(3), r["H"]])
    Hc = [T, T @ r["H"]]
    (wA, mA), (wB, mB) = warp_inverse(A, Hc[0], size), warp_inverse(B, Hc[1], size)
    ov = mA & mB
    # exclude clipped pixels (under/over-exposed) from the statistics
    ok = ov & (wA.min(2) > 4) & (wA.max(2) < 250) & (wB.min(2) > 4) & (wB.max(2) < 250)
    if ok.sum() < 200:
        raise SystemExit("Overlap with well-exposed pixels is too small to estimate a gain.")
    lA, lB = to_lin(wA), to_lin(wB)
    if a.per_channel:
        gains = []
        for c in range(3):
            means = {(0, 1): float(lA[..., c][ok].mean()), (1, 0): float(lB[..., c][ok].mean())}
            gains.append(solve_gains(means, {(0, 1): int(ok.sum()), (1, 0): int(ok.sum())}, sig_g=a.sigma_g))
        g = np.array(gains).T                       # shape (2, 3)
    else:
        means = {(0, 1): float(lA[ok].mean()), (1, 0): float(lB[ok].mean())}
        g1 = solve_gains(means, {(0, 1): int(ok.sum()), (1, 0): int(ok.sum())}, sig_g=a.sigma_g)
        g = np.repeat(g1[:, None], 3, axis=1)
    cA = from_lin(lA * g[0][None, None, :])
    cB = from_lin(lB * g[1][None, None, :])

    def composite(x, y):
        out, _ = composite_overwrite([x, y], [mA, mB])
        return out
    before, after = composite(wA, wB), composite(cA, cB)

    def lum(x):
        return x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
    gap_b = float(abs(lum(wA)[ok].mean() - lum(wB)[ok].mean()))
    gap_a = float(abs(lum(cA)[ok].mean() - lum(cB)[ok].mean()))
    mad_b = float(np.abs(lum(wA) - lum(wB))[ok].mean())
    mad_a = float(np.abs(lum(cA) - lum(cB))[ok].mean())
    wtA = warp_inverse(feather_weight(A.shape), Hc[0], size)[0][..., 0] * mA
    wtB = warp_inverse(feather_weight(B.shape), Hc[1], size)[0][..., 0] * mB
    den = np.maximum(wtA + wtB, 1e-6)[..., None]
    after_feather = (wtA[..., None] * cA + wtB[..., None] * cB) / den
    res = {"inliers": r["n_inliers"], "overlap_pixels_used": int(ok.sum()), "simulated": a.simulate,
           "gains_A_B_per_channel_BGR": g.tolist(), "gain_ratio_B_over_A": float(np.mean(g[1] / g[0])),
           "model": "I_corrected = (g_i * I_lin)^(1/2.2); g solved from overlap means (Brown & Lowe), "
                    f"sigma_N=10, sigma_g={a.sigma_g}, clipped pixels excluded",
           "overlap_mean_luminance_gap_before": gap_b, "overlap_mean_luminance_gap_after": gap_a,
           "overlap_mean_abs_luminance_diff_before": mad_b, "overlap_mean_abs_luminance_diff_after": mad_a}
    save_json(d, res)
    sh = 560
    cv2.imwrite(f"{d}/before.png", to_u8(before))
    cv2.imwrite(f"{d}/after.png", to_u8(after))
    cv2.imwrite(f"{d}/after_feather.png", to_u8(after_feather))
    cv2.imwrite(f"{d}/before_after.png", hcat([
        banner(fit_h(to_u8(before), sh), "Before gain compensation"),
        banner(fit_h(to_u8(after), sh), f"After (gain A={g[0].mean():.2f}, B={g[1].mean():.2f})"),
        banner(fit_h(to_u8(after_feather), sh), "After + feather blend")]))
    cv2.imwrite(f"{d}/inputs.png", hcat([banner(fit_h(A, sh), "A"), banner(fit_h(B, sh), "B")]))
    print(_dumps(res, indent=1))
    print("saved to", d)


# ================================================================ main ====
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ls")
    add_common(p)
    p.add_argument("--pair", nargs=2, required=True)
    p.set_defaults(fn=cmd_ls)

    p = sub.add_parser("interp")
    add_common(p)
    p.add_argument("--pair", nargs=2, required=True)
    p.add_argument("--zoom", type=float, default=1.0, help="magnify the output canvas (makes differences larger)")
    p.add_argument("--win", type=int, default=100, help="crop window (px)")
    p.set_defaults(fn=cmd_interp)

    p = sub.add_parser("sweep")
    add_common(p)
    p.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "A", "B"))
    p.add_argument("--thresholds", default="0.5,1,2,3,5,8,12,20")
    p.add_argument("--crop_mode", choices=["seam", "maxdiff"], default="seam",
                   help="seam: crop on the border of image B at its most textured place (default); "
                        "maxdiff: window where the panoramas differ most")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("focal")
    add_common(p)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--boot", type=int, default=40, help="bootstrap repetitions")
    p.add_argument("--min_inliers", type=int, default=60)
    p.add_argument("--min_angle", type=float, default=15.0, help="rotation (deg) below which f is ill-conditioned")
    p.add_argument("--f35", type=float, default=26.0, help="35mm-equivalent focal length used if EXIF is missing "
                                                          "(iPhone 14 main camera: 26)")
    p.set_defaults(fn=cmd_focal)

    p = sub.add_parser("cyl")
    add_common(p)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--f", default="auto", help="cylinder scale in px, or 'auto'")
    p.set_defaults(fn=cmd_cyl)

    p = sub.add_parser("gain")
    add_common(p)
    p.add_argument("--pair", nargs=2, required=True)
    p.add_argument("--per_channel", action="store_true", help="separate gain for B, G, R")
    p.add_argument("--sigma_g", type=float, default=0.5,
                   help="prior std of the gains (Brown&Lowe use 0.1 for small mismatches; use a larger value "
                        "for a strong exposure difference so the prior does not under-correct)")
    p.add_argument("--simulate", type=float, default=None,
                   help="TEST ONLY: multiply B by this factor (linear light) to simulate an exposure change")
    p.set_defaults(fn=cmd_gain)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
