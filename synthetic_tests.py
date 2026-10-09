"""
synthetic_tests.py -- evidence for core items 3 (DLT) and 5 (warp checks).
Run:  python synthetic_tests.py --out out
"""
import argparse
import os
import numpy as np
import cv2
from stitchlib import (apply_h, dlt_homography, warp_inverse, to_u8, hstack_pad)

H_TRUE = np.array([[1.10, 0.12, 25.0],
                   [-0.08, 0.95, -15.0],
                   [2e-4, 1e-4, 1.0]])


def test_dlt():
    # 4 points: irregular convex quadrilateral, no three collinear
    src4 = np.array([[50, 60], [560, 40], [600, 420], [80, 400]], float)
    # 12 points: random (seeded) in the image area
    rng = np.random.default_rng(1)
    src12 = rng.uniform([20, 20], [620, 460], (12, 2))
    rows = []
    for name, src in (("4 points (minimal)", src4), ("12 points (overdetermined)", src12)):
        dst = apply_h(H_TRUE, src)
        Hest = dlt_homography(src, dst)
        err = np.linalg.norm(apply_h(Hest, src) - dst, axis=1).mean()
        rows.append((name, len(src), err))
    print("Known H =\n", H_TRUE)
    for name, n, err in rows:
        print(f"{name:30s} n={n:2d}  mean reprojection error = {err:.3e} px")
    return rows


def make_test_image(w=400, h=300):
    img = np.full((h, w, 3), 255, np.uint8)
    for x in range(0, w, 50):
        cv2.line(img, (x, 0), (x, h - 1), (210, 210, 210), 1)
    for y in range(0, h, 50):
        cv2.line(img, (0, y), (w - 1, y), (210, 210, 210), 1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 3)
    # distinctly coloured discs = identifiable landmarks (BGR)
    marks = {"red": ((60, 60), (0, 0, 255)), "green": ((340, 60), (0, 200, 0)),
             "blue": ((60, 240), (255, 0, 0)), "magenta": ((340, 240), (255, 0, 255)),
             "orange": ((200, 150), (0, 128, 255))}
    for _, (c, col) in marks.items():
        cv2.circle(img, c, 14, col, -1)
    cv2.putText(img, "TOP", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img, marks


def centroid_of_color(img, col, tol=60):
    d = np.abs(img.astype(int) - np.array(col)[None, None]).sum(2)
    ys, xs = np.nonzero(d < tol)
    return (xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan)


def mark_expected(img, pts):
    out = img.copy()
    for x, y in pts:
        p = (int(round(x)), int(round(y)))
        cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_CROSS, 22, 2)
    return out


def test_warp(outdir):
    img, marks = make_test_image()
    h, w = img.shape[:2]
    centres = np.array([c for c, _ in marks.values()], float)
    res = {}

    # identity must reproduce the input
    W0, _ = warp_inverse(img, np.eye(3), (w, h), "bilinear")
    res["identity max abs diff"] = float(np.abs(W0 - img).max())

    # translation by (60, 40) on a larger fixed canvas
    Ht = np.array([[1, 0, 60], [0, 1, 40], [0, 0, 1.0]])
    size_t = (w + 100, h + 80)
    Wt, _ = warp_inverse(img, Ht, size_t, "bilinear")

    # rotation 20 deg + scale 0.8 about the image centre, canvas 500x450
    th, s = np.deg2rad(20), 0.8
    R = np.array([[s * np.cos(th), -s * np.sin(th), 0], [s * np.sin(th), s * np.cos(th), 0], [0, 0, 1]])
    size_r = (500, 450)
    Hr = (np.array([[1, 0, size_r[0] / 2], [0, 1, size_r[1] / 2], [0, 0, 1]]) @ R @
          np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]]))
    Wr, _ = warp_inverse(img, Hr, size_r, "bilinear")

    for tag, H, Wimg in (("translation (+60,+40)", Ht, Wt), ("rotation 20deg + scale 0.8", Hr, Wr)):
        exp = apply_h(H, centres)
        errs = []
        for (name, (c, col)), e in zip(marks.items(), exp):
            got = centroid_of_color(to_u8(Wimg), col)
            errs.append(np.hypot(got[0] - e[0], got[1] - e[1]))
        res[tag + " landmark err (px)"] = [round(float(x), 2) for x in errs]
        print(f"{tag}: landmark centroid error px =", np.round(errs, 2))
    print("identity max abs diff:", res["identity max abs diff"])

    os.makedirs(outdir, exist_ok=True)
    panel = hstack_pad([img,
                        mark_expected(to_u8(Wt), apply_h(Ht, centres)),
                        mark_expected(to_u8(Wr), apply_h(Hr, centres))])
    cv2.imwrite(os.path.join(outdir, "synthetic_warp.png"), panel)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    import json
    dlt = test_dlt()
    warp = test_warp(a.out)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"H_true": H_TRUE.tolist(),
               "dlt": [{"test": n, "n_points": k, "mean_err_px": e} for n, k, e in dlt],
               "warp": warp}, open(os.path.join(a.out, "synthetic.json"), "w"), indent=2)
