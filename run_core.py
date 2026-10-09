"""
run_core.py -- core pipeline for the two required pairs.

Usage:
  python run_core.py --set1 s1_a.jpg s1_b.jpg --set2 s2_a.jpg s2_b.jpg --out out

Image A (first of each pair) is the REFERENCE; H maps image B -> image A.
Outputs (in --out/<setname>/): keypoints, matches before/after ratio test,
inliers/outliers, panorama, annotated panorama, plus results.json / table.md.
"""
import argparse
import json
import os
import time
import numpy as np
import cv2
from stitchlib import (load_image, detect_sift, match_descriptors, ransac_homography,
                       compute_canvas, warp_inverse, composite_overwrite, to_u8,
                       draw_keypoints, draw_matches, draw_boundaries, apply_h)


def process_pair(name, pa, pb, a, out):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    A, B = load_image(pa, a.long_edge), load_image(pb, a.long_edge)
    kA, pA, dA = detect_sift(A, a.nfeatures)
    kB, pB, dB = detect_sift(B, a.nfeatures)
    print(f"[{name}] keypoints A={len(pA)} B={len(pB)}  (nfeatures={a.nfeatures}, other SIFT params default)")

    m = match_descriptors(dA, dB, a.ratio, cross_check=a.cross_check)
    print(f"[{name}] ratio sweep (matches surviving) :",
          {r: int((m['ratios'] < r).sum()) for r in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)})
    mA, mB = m["matches"][:, 0], m["matches"][:, 1]
    src, dst = pB[mB], pA[mA]                      # H: B -> A
    print(f"[{name}] candidates={len(pA)}  after ratio test ({a.ratio})={len(src)}")

    rf = ransac_homography(src, dst, a.thresh, a.iters, adaptive=False, seed=a.seed)
    ra = ransac_homography(src, dst, a.thresh, a.iters, adaptive=True,
                           confidence=a.conf, seed=a.seed)
    print(f"[{name}] RANSAC fixed: inliers={rf['n_inliers']}/{rf['n_matches']} "
          f"err before={rf['err_before']:.3f} after={rf['err_after']:.3f} px")
    print(f"[{name}] RANSAC adaptive(p={a.conf}): iters={ra['iters_used']} inliers={ra['n_inliers']}")

    # ---------------- visualisations ----------------
    cv2.imwrite(f"{d}/input_A.png", A)
    cv2.imwrite(f"{d}/input_B.png", B)
    cv2.imwrite(f"{d}/keypoints_A.png", draw_keypoints(A, pA))
    cv2.imwrite(f"{d}/keypoints_B.png", draw_keypoints(B, pB))
    cand = draw_matches(A, B, pA, pB[m["nn"]], None, max_lines=300)
    cv2.imwrite(f"{d}/matches_before_ratio.png", cand)
    cv2.imwrite(f"{d}/matches_after_ratio.png", draw_matches(A, B, dst, src, None, max_lines=300))
    cv2.imwrite(f"{d}/matches_ransac.png",
                draw_matches(A, B, dst, src, rf["inliers"], max_lines=400))

    # ---------------- warping + compositing ----------------
    H = rf["H"]
    T, size = compute_canvas([A.shape, B.shape], [np.eye(3), H])
    Hc = [T, T @ H]
    warped, masks = zip(*[warp_inverse(im, h_, size, a.interp) for im, h_ in zip((A, B), Hc)])
    pano, occ = composite_overwrite(list(warped), list(masks))
    pano = to_u8(pano)
    cv2.imwrite(f"{d}/panorama.png", pano)
    cv2.imwrite(f"{d}/panorama_annotated.png",
                draw_boundaries(pano, [A.shape, B.shape], Hc, ["A (reference)", "B"]))
    np.save(f"{d}/H_B_to_A.npy", H)

    return {
        "pair": name, "kp_A": len(pA), "kp_B": len(pB), "candidates": len(pA),
        "ratio_matches": len(src), "ratio": a.ratio,
        "ransac_thresh_px": a.thresh, "ransac_iters_fixed": a.iters,
        "inliers": rf["n_inliers"], "inlier_ratio": round(rf["n_inliers"] / len(src), 4),
        "err_before_refit": round(rf["err_before"], 4),
        "err_after_refit": round(rf["err_after"], 4),
        "adaptive_iters": ra["iters_used"], "adaptive_inliers": ra["n_inliers"],
        "canvas_WxH": f"{size[0]}x{size[1]}", "interp": a.interp,
        "outliers": len(src) - rf["n_inliers"], "input_size": f"{A.shape[1]}x{A.shape[0]}",
        "nfeatures": a.nfeatures, "long_edge": a.long_edge, "conf": a.conf,
        "seed": a.seed, "cross_check": a.cross_check,
        "ratio_sweep": {str(r): int((m["ratios"] < r).sum())
                        for r in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set1", nargs=2, help="rotation pair (optional)")
    ap.add_argument("--set2", nargs=2, help="planar-translation pair (optional)")
    ap.add_argument("--pair", nargs=3, action="append", default=[], metavar=("NAME", "A", "B"),
                    help="any extra named pair; may be repeated")
    ap.add_argument("--out", default="out")
    ap.add_argument("--long_edge", type=int, default=800)
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--conf", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interp", choices=["nearest", "bilinear"], default="bilinear")
    ap.add_argument("--cross_check", action="store_true")
    a = ap.parse_args()

    jobs = []
    if a.set1:
        jobs.append(("set1_rotation", *a.set1))
    if a.set2:
        jobs.append(("set2_planar", *a.set2))
    jobs += [tuple(p) for p in a.pair]
    if not jobs:
        ap.error("give at least one of --set1 / --set2 / --pair NAME A B")
    rows = [process_pair(n, pa, pb, a, a.out) for n, pa, pb in jobs]
    # merge with earlier runs in the same --out folder (same name -> replaced)
    rp = os.path.join(a.out, "results.json")
    if os.path.exists(rp):
        new_names = {r["pair"] for r in rows}
        rows = [r for r in json.load(open(rp)) if r["pair"] not in new_names] + rows
    json.dump(rows, open(rp, "w"), indent=2)

    cols = [c for c in rows[0].keys() if c != "ratio_sweep"]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        md += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    open(os.path.join(a.out, "table.md"), "w").write(md)
    print("\n" + md)
    print("Next: python make_dashboard.py --out", a.out, " (builds report.html + figure panels)")


if __name__ == "__main__":
    main()
