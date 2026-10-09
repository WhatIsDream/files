"""
run_sequence.py -- stitch an ordered sequence of N images (N >= 3) into one panorama.
(Additional feature: Multi-Image Panorama.)

Usage:
  python run_sequence.py --name rot_seq --images r1.jpg r2.jpg r3.jpg r4.jpg r5.jpg --out out
  python run_sequence.py --name rot_seq --images ... --ref center     (default; or: first | <index, 1-based>)

Method (state it in the report):
  * SIFT on every image; consecutive pairs (i, i+1) are matched with the own matcher + ratio test.
  * RANSAC (same code as the core) gives Hp[i], which maps image i+1 -> image i.
  * Chain to the reference image r:   Hs[r] = I
        k > r :  Hs[k] = Hs[k-1] @ Hp[k-1]          (k -> k-1 -> ... -> r)
        k < r :  Hs[k] = Hs[k+1] @ inv(Hp[k])       (k -> k+1 -> ... -> r)
  * Canvas bounds from all transformed corners, inverse warping, then compositing.
    Both overwrite and feather (own weight = distance to the image border) outputs are saved.
Run make_dashboard.py afterwards to see everything in report.html.
"""
import argparse
import json
import os
import numpy as np
import cv2
from stitchlib import (load_image, detect_sift, match_descriptors, ransac_homography,
                       compute_canvas, warp_inverse, composite_overwrite, to_u8,
                       draw_matches, draw_boundaries, apply_h, image_corners)


def feather_weight(shape):
    """Weight = distance (in pixels) to the nearest image border, so it is 0 at the edge."""
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([xs, w - 1 - xs, ys, h - 1 - ys]).astype(np.float32) + 1e-3
    return d


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="images in capture order (>= 3)")
    ap.add_argument("--name", default="sequence")
    ap.add_argument("--out", default="out")
    ap.add_argument("--ref", default="center", help="center | first | 1-based index")
    ap.add_argument("--long_edge", type=int, default=800)
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interp", choices=["nearest", "bilinear"], default="bilinear")
    a = ap.parse_args()

    n = len(a.images)
    if n < 3:
        ap.error("need at least 3 images (use run_core.py for pairs)")
    ref = n // 2 if a.ref == "center" else 0 if a.ref == "first" else int(a.ref) - 1
    if not 0 <= ref < n:
        ap.error("--ref out of range")
    d = os.path.join(a.out, a.name)
    os.makedirs(d, exist_ok=True)

    imgs = [load_image(p, a.long_edge) for p in a.images]
    feats = [detect_sift(im, a.nfeatures) for im in imgs]
    print("keypoints per image:", [len(f[1]) for f in feats])
    cv2.imwrite(f"{d}/inputs.png", np.hstack([cv2.resize(
        im, (int(im.shape[1] * 300 / im.shape[0]), 300)) for im in imgs]))

    Hp, pair_stats = [], []
    for i in range(n - 1):
        _, pA, dA = feats[i]
        _, pB, dB = feats[i + 1]
        m = match_descriptors(dA, dB, a.ratio)["matches"]
        if len(m) < 8:
            raise SystemExit(f"Pair {i + 1}-{i + 2}: only {len(m)} matches. Overlap too small or "
                             f"images not in capture order.")
        src, dst = pB[m[:, 1]], pA[m[:, 0]]            # H maps image i+1 -> image i
        r = ransac_homography(src, dst, a.thresh, a.iters, seed=a.seed)
        Hp.append(r["H"])
        fn = f"matches_pair{i + 1}.png"
        cv2.imwrite(f"{d}/{fn}", draw_matches(imgs[i], imgs[i + 1], dst, src, r["inliers"], 250))
        st = {"pair": f"{i + 1}-{i + 2}", "kp_a": len(pA), "kp_b": len(pB), "matches": len(m),
              "inliers": r["n_inliers"], "inlier_ratio": r["n_inliers"] / len(m),
              "err_before": r["err_before"], "err_after": r["err_after"], "file": fn}
        pair_stats.append(st)
        print(f"pair {st['pair']}: matches={st['matches']} inliers={st['inliers']} "
              f"({st['inlier_ratio']:.1%}) err {st['err_before']:.3f}->{st['err_after']:.3f}px")

    # ---- chain everything into the reference frame
    Hs = [None] * n
    Hs[ref] = np.eye(3)
    for k in range(ref + 1, n):
        Hs[k] = Hs[k - 1] @ Hp[k - 1]
    for k in range(ref - 1, -1, -1):
        Hs[k] = Hs[k + 1] @ np.linalg.inv(Hp[k])
    Hs = [H / H[2, 2] for H in Hs]

    shapes = [im.shape for im in imgs]
    try:
        T, size = compute_canvas(shapes, Hs)
    except ValueError as e:
        raise SystemExit(f"{e}\nThe flat (planar) canvas blows up for this field of view. Use a "
                         f"smaller --ref distance, fewer images, or the cylindrical script.")
    Hc = [T @ H for H in Hs]
    print("reference image:", ref + 1, " canvas:", size)

    warped, masks, weights = [], [], []
    for im, H in zip(imgs, Hc):
        w_, m_ = warp_inverse(im, H, size, a.interp)
        wt, _ = warp_inverse(feather_weight(im.shape), H, size, "bilinear")
        warped.append(w_)
        masks.append(m_)
        weights.append(wt[..., 0] * m_)

    order = sorted(range(n), key=lambda k: -abs(k - ref))      # reference drawn last (on top)
    over, _ = composite_overwrite([warped[k] for k in order], [masks[k] for k in order])
    num = sum(w[..., None] * x for w, x in zip(weights, warped))
    den = sum(weights)[..., None]
    feather = np.where(den > 0, num / np.maximum(den, 1e-6), 0)
    cv2.imwrite(f"{d}/panorama_overwrite.png", to_u8(over))
    cv2.imwrite(f"{d}/panorama_feather.png", to_u8(feather))
    cv2.imwrite(f"{d}/panorama_annotated.png", draw_boundaries(
        to_u8(feather), shapes, Hc,
        [f"{k + 1}" + (" (ref)" if k == ref else "") for k in range(n)]))

    area = [float(polygon_area(apply_h(H, image_corners(s))) / (s[0] * s[1])) for s, H in zip(shapes, Hs)]
    json.dump({"n_images": n, "ref": ref, "canvas_WxH": f"{size[0]}x{size[1]}",
               "area_scale": area, "pairs": pair_stats,
               "params": {"images": [os.path.basename(p) for p in a.images],
                          "SIFT nfeatures": a.nfeatures, "ratio": a.ratio, "RANSAC thr (px)": a.thresh,
                          "RANSAC iters": a.iters, "interp": a.interp,
                          "reference choice": a.ref}},
              open(f"{d}/sequence.json", "w"), indent=2)
    print("area scale per image (1.0 = undistorted size):", [round(x, 2) for x in area])
    print("Next: python make_dashboard.py --out", a.out)


if __name__ == "__main__":
    main()
