"""
check_pairs.py -- quick feasibility check for a set of photos (all pairs).
For every pair (i, j): ratio-test matches, RANSAC inliers, inlier ratio, and the overlap
fraction = (area of image j's footprint inside image i) / (area of image i).
Use it BEFORE choosing the Core pair (target overlap 30-50%).

  python check_pairs.py --images a.jpg b.jpg c.jpg d.jpg e.jpg
"""
import argparse
import itertools
import cv2
import numpy as np
from stitchlib import (load_image, detect_sift, match_descriptors, ransac_homography,
                       apply_h, image_corners)


def overlap_fraction(shape_a, shape_b, H_b_to_a):
    h, w = shape_a[:2]
    pa = np.zeros((h, w), np.uint8)
    pa[:] = 1
    poly = apply_h(H_b_to_a, image_corners(shape_b))
    if not np.all(np.isfinite(poly)) or np.abs(poly).max() > 1e5:
        return 0.0
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mb, [np.round(poly).astype(np.int32)], 1)
    return float(mb.sum()) / (h * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--long_edge", type=int, default=800)
    a = ap.parse_args()
    imgs = [load_image(p, a.long_edge) for p in a.images]
    feats = [detect_sift(im, 4000) for im in imgs]
    print(f"{'pair':6s} {'matches':>8s} {'inliers':>8s} {'ratio':>7s} {'j covers i':>11s} {'i covers j':>11s} {'mean':>6s}")
    for i, j in itertools.combinations(range(len(imgs)), 2):
        m = match_descriptors(feats[i][2], feats[j][2], a.ratio)["matches"]
        if len(m) < 8:
            print(f"{i + 1}-{j + 1:<4d} {len(m):8d}   (too few matches)")
            continue
        src, dst = feats[j][1][m[:, 1]], feats[i][1][m[:, 0]]
        r = ransac_homography(src, dst, a.thresh, 1500, seed=0)
        ok = r["n_inliers"] >= 15
        o1 = overlap_fraction(imgs[i].shape, imgs[j].shape, r["H"]) if ok else 0.0
        o2 = overlap_fraction(imgs[j].shape, imgs[i].shape, np.linalg.inv(r["H"])) if ok else 0.0
        print(f"{i + 1}-{j + 1:<4d} {len(m):8d} {r['n_inliers']:8d} {r['n_inliers'] / len(m):7.2f} "
              f"{o1:11.0%} {o2:11.0%} {(o1 + o2) / 2:6.0%}")


if __name__ == "__main__":
    main()
