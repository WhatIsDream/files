"""
make_dashboard.py -- turn the outputs of run_core.py / synthetic_tests.py into
  (1) labeled, report-ready figure panels   -> <out>/panels/*.png
  (2) one HTML page with every table/number -> <out>/report.html  (opens in browser)

Usage:  python make_dashboard.py --out out
Run synthetic_tests.py and run_core.py first (same --out folder).
"""
import argparse
import html
import json
import os
import webbrowser
import cv2
import numpy as np

SET_TITLES = {"set1_rotation": "Set 1 - fixed-center rotation",
              "set2_planar": "Set 2 - camera translation, planar scene"}


# ---------------------------------------------------------------- helpers ---
def read(path):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    return im


def fit_height(im, h):
    return cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)


def fit_width(im, w):
    return cv2.resize(im, (w, round(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)


def banner(im, text, size=0.8):
    """Add a white caption strip on top of an image."""
    bar = np.full((44, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, size, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def hpad(imgs, pad=16):
    h = max(i.shape[0] for i in imgs)
    cols = []
    for i in imgs:
        c = np.full((h, i.shape[1] + pad, 3), 255, np.uint8)
        c[:i.shape[0], :i.shape[1]] = i
        cols.append(c)
    return np.hstack(cols)


def vpad(imgs, pad=16):
    w = max(i.shape[1] for i in imgs)
    rows = []
    for i in imgs:
        r = np.full((i.shape[0] + pad, w, 3), 255, np.uint8)
        r[:i.shape[0], :i.shape[1]] = i
        rows.append(r)
    return np.vstack(rows)


def legend(width, items):
    bar = np.full((36, width, 3), 255, np.uint8)
    x = 10
    for col, txt in items:
        cv2.rectangle(bar, (x, 10), (x + 22, 28), col, -1)
        cv2.putText(bar, txt, (x + 30, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        x += 40 + 11 * len(txt)
    return bar


# ----------------------------------------------------------------- panels ---
def make_panels(out, rows):
    pdir = os.path.join(out, "panels")
    os.makedirs(pdir, exist_ok=True)
    made = {}
    inputs_rows = []
    for r in rows:
        n = r["pair"]
        d = os.path.join(out, n)
        t = SET_TITLES.get(n, n)
        A, B = read(f"{d}/input_A.png"), read(f"{d}/input_B.png")
        inp = hpad([banner(fit_height(A, 420), "Image A (reference)"),
                    banner(fit_height(B, 420), "Image B")])
        inputs_rows.append(banner(inp, t, 0.9))

        kA, kB = read(f"{d}/keypoints_A.png"), read(f"{d}/keypoints_B.png")
        kp = hpad([banner(fit_height(kA, 520), f"A: {r['kp_A']} SIFT keypoints"),
                   banner(fit_height(kB, 520), f"B: {r['kp_B']} SIFT keypoints")])
        cv2.imwrite(f"{pdir}/{n}_keypoints.png", banner(kp, t, 0.9))

        bef = read(f"{d}/matches_before_ratio.png")
        aft = read(f"{d}/matches_after_ratio.png")
        w = 1200
        rat = vpad([banner(fit_width(bef, w), f"Before ratio test: {r['candidates']} candidate "
                                              f"nearest-neighbour matches (300 random lines drawn)"),
                    banner(fit_width(aft, w), f"After ratio test (ratio < {r['ratio']}): "
                                              f"{r['ratio_matches']} matches (300 random lines drawn)")])
        cv2.imwrite(f"{pdir}/{n}_ratio_test.png", banner(rat, t, 0.9))

        ran = read(f"{d}/matches_ransac.png")
        ran = fit_width(ran, 1200)
        ran = vpad([banner(ran, f"RANSAC: {r['inliers']} inliers / {r['outliers']} outliers "
                                f"(ratio {r['inlier_ratio']:.1%}, thr {r['ransac_thresh_px']} px, "
                                f"{r['ransac_iters_fixed']} iters)"),
                    legend(1200, [((0, 200, 0), "inlier"), ((0, 0, 255), "outlier")])], pad=4)
        cv2.imwrite(f"{pdir}/{n}_ransac.png", banner(ran, t, 0.9))

        P, PA = read(f"{d}/panorama.png"), read(f"{d}/panorama_annotated.png")
        pa = hpad([banner(fit_height(P, 620), f"Panorama ({r['canvas_WxH']}, {r['interp']})"),
                   banner(fit_height(PA, 620), "Boundaries: red = A (reference), green = B")])
        cv2.imwrite(f"{pdir}/{n}_panorama.png", banner(pa, t, 0.9))
        made[n] = [f"panels/{n}_keypoints.png", f"panels/{n}_ratio_test.png",
                   f"panels/{n}_ransac.png", f"panels/{n}_panorama.png"]
    cv2.imwrite(f"{pdir}/inputs_both_sets.png", vpad(inputs_rows, 24))
    return made


# ------------------------------------------------------------------- HTML ---
CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:1250px;margin:24px auto;padding:0 16px;color:#222}
h1{border-bottom:3px solid #7a0019;padding-bottom:6px} h2{margin-top:36px;color:#7a0019}
table{border-collapse:collapse;margin:10px 0;font-size:14px}
th,td{border:1px solid #bbb;padding:5px 10px;text-align:right} th{background:#f2e6e8}
td:first-child,th:first-child{text-align:left} img{max-width:100%;border:1px solid #ccc;margin:8px 0}
.ok{color:#0a7d2c;font-weight:600}.warn{color:#c47a00;font-weight:600}.bad{color:#b00020;font-weight:600}
.note{background:#fff8e1;border-left:4px solid #f5b800;padding:8px 12px;margin:10px 0}
code{background:#eee;padding:1px 4px}
"""


def table(headers, body):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in body)
    return f"<table><tr>{h}</tr>{b}</table>"


def verdict(r):
    """Automatic sanity check of one pair (rule-of-thumb, not the official grading)."""
    out = []
    ir = r["inlier_ratio"]
    out.append(("ok", f"inlier ratio {ir:.1%}") if ir >= 0.5 else
               ("warn", f"inlier ratio {ir:.1%} (low, check the match image)") if ir >= 0.3 else
               ("bad", f"inlier ratio {ir:.1%} (too low: re-shoot or inspect matches)"))
    out.append(("ok", f"{r['inliers']} inliers") if r["inliers"] >= 100 else
                ("warn", f"only {r['inliers']} inliers"))
    d = abs(r["adaptive_inliers"] - r["inliers"]) / max(r["inliers"], 1)
    out.append(("ok", "adaptive RANSAC agrees with fixed budget") if d < 0.1 else
               ("warn", "adaptive and fixed RANSAC disagree >10% (unstable matches)"))
    w, h = map(int, r["canvas_WxH"].split("x"))
    iw, ih = map(int, r["input_size"].split("x"))
    out.append(("ok", f"canvas {r['canvas_WxH']} is plausible") if w * h < 6 * iw * ih else
               ("bad", f"canvas {r['canvas_WxH']} is huge: homography probably diverged"))
    return out


def build_core(out, rows, syn, made):
    p = []
    p.append(f"<html><head><meta charset='utf-8'><title>HW1 results</title><style>{CSS}</style></head><body>")
    p.append("<h1>HW1 - Panorama Stitching: all results in one page</h1>")
    p.append("<div class='note'>Every number and figure below comes from the same run. "
             "Copy the tables into the report; figures are in <code>panels/</code> (and per-set folders).</div>")

    if not rows:
        return p
    # ---- parameters
    r0 = rows[0]
    p.append("<h2>Parameters (state these in the report)</h2>")
    p.append(table(["parameter", "value"], [
        ["working resolution (long edge)", f"{r0['long_edge']} px"],
        ["SIFT", f"cv2.SIFT_create(nfeatures={r0['nfeatures']}), other parameters default"],
        ["ratio threshold", r0["ratio"]], ["cross-check", r0["cross_check"]],
        ["RANSAC inlier threshold", f"{r0['ransac_thresh_px']} px (forward transfer error)"],
        ["RANSAC fixed iteration budget", r0["ransac_iters_fixed"]],
        ["adaptive RANSAC confidence", r0["conf"]], ["random seed", r0["seed"]],
        ["interpolation", r0["interp"]],
        ["homography direction", "H maps image B -> image A (A is the reference)"]]))

    # ---- main table
    p.append("<h2>Core results table (one row per required pair)</h2>")
    hdr = ["pair", "kp A", "kp B", "candidates", "ratio-test matches", "RANSAC inliers", "inlier ratio",
           "mean err before refit (px)", "mean err after refit (px)", "canvas W x H"]
    body = [[r["pair"], r["kp_A"], r["kp_B"], r["candidates"], r["ratio_matches"], r["inliers"],
             f"{r['inlier_ratio']:.3f}", f"{r['err_before_refit']:.3f}", f"{r['err_after_refit']:.3f}",
             r["canvas_WxH"]] for r in rows]
    p.append(table(hdr, body))

    p.append("<h2>Ratio-threshold sweep (use it to justify your ratio)</h2>")
    ks = list(rows[0]["ratio_sweep"].keys())
    p.append(table(["pair"] + [f"ratio<{k}" for k in ks],
                   [[r["pair"]] + [r["ratio_sweep"][k] for k in ks] for r in rows]))

    p.append("<h2>Fixed budget vs. confidence-based stopping (additional feature)</h2>")
    p.append(table(["pair", "fixed iters", "fixed inliers", "adaptive iters (p=%s)" % r0["conf"], "adaptive inliers"],
                   [[r["pair"], r["ransac_iters_fixed"], r["inliers"], r["adaptive_iters"], r["adaptive_inliers"]]
                    for r in rows]))

    # ---- automatic checks
    p.append("<h2>Automatic sanity checks</h2><ul>")
    for r in rows:
        p.append(f"<li><b>{r['pair']}</b><ul>")
        for lvl, msg in verdict(r):
            p.append(f"<li class='{lvl}'>{html.escape(msg)}</li>")
        p.append("</ul></li>")
    p.append("</ul>")

    # ---- synthetic
    if syn:
        p.append("<h2>Synthetic DLT test (core item 3)</h2>")
        p.append("<p>Known H =</p><pre>" + "\n".join("  ".join(f"{v: .4g}" for v in row) for row in syn["H_true"]) + "</pre>")
        p.append(table(["test", "points", "mean reprojection error (px)"],
                       [[d["test"], d["n_points"], f"{d['mean_err_px']:.2e}"] for d in syn["dlt"]]))
        w = syn["warp"]
        p.append("<h2>Synthetic warp test (core item 5)</h2>")
        p.append(table(["check", "result"], [[k, v] for k, v in w.items()]))
        p.append("<p>Left to right: original, translated by (+60,+40), rotated 20 deg + scaled 0.8. "
                 "Black crosses = expected landmark positions H&middot;p.</p>")
        p.append("<img src='synthetic_warp.png'>")

    # ---- figures
    p.append("<h2>Input images</h2><img src='panels/inputs_both_sets.png'>")
    for r in rows:
        n = r["pair"]
        p.append(f"<h2>{html.escape(SET_TITLES.get(n, n))}</h2>")
        for name, f in zip(["Keypoints", "Matches before / after ratio test",
                            "RANSAC inliers vs outliers", "Panorama + boundaries"], made[n]):
            p.append(f"<h3>{name}</h3><img src='{f}'>")
    return p


def seq_section(out):
    """Sections for every <out>/<name>/sequence.json written by run_sequence.py."""
    p = []
    for name in sorted(os.listdir(out)):
        f = os.path.join(out, name, "sequence.json")
        if not os.path.isfile(f):
            continue
        s = json.load(open(f))
        p.append(f"<h2>Multi-image panorama: {html.escape(name)} "
                 f"({s['n_images']} images, reference = image {s['ref'] + 1})</h2>")
        p.append(table(["parameter", "value"], [[k, v] for k, v in s["params"].items()] +
                       [["canvas W x H", s["canvas_WxH"]], ["homography chain",
                        "H_k maps image k -> reference by composing consecutive pair homographies"]]))
        p.append("<h3>Consecutive-pair statistics</h3>")
        p.append(table(["pair", "kp i", "kp i+1", "ratio matches", "inliers", "inlier ratio",
                        "err before refit", "err after refit"],
                       [[q["pair"], q["kp_a"], q["kp_b"], q["matches"], q["inliers"],
                         f"{q['inlier_ratio']:.3f}", f"{q['err_before']:.3f}", f"{q['err_after']:.3f}"]
                        for q in s["pairs"]]))
        p.append("<h3>Per-image footprint (accumulated distortion)</h3>")
        p.append(table(["image", "chain length to ref", "area scale vs. original"],
                       [[i + 1, abs(i - s["ref"]), f"{a:.2f}"] for i, a in enumerate(s["area_scale"])]))
        p.append(f"<img src='{name}/inputs.png'>")
        p.append("<h3>Final panorama (feather blending)</h3>" + f"<img src='{name}/panorama_feather.png'>")
        p.append("<h3>Same panorama, overwrite compositing</h3>" + f"<img src='{name}/panorama_overwrite.png'>")
        p.append("<h3>Input boundaries (one colour per image)</h3>" + f"<img src='{name}/panorama_annotated.png'>")
        p.append("<h3>RANSAC inliers of each consecutive pair</h3>")
        for q in s["pairs"]:
            p.append(f"<img style='max-width:48%' src='{name}/{q['file']}'>")
    return p


def extras_section(out):
    """Everything written by extras.py: <out>/extras/<cmd>_<name>/ (result.json + PNG figures)."""
    p = []
    root = os.path.join(out, "extras")
    if not os.path.isdir(root):
        return p
    for name in sorted(os.listdir(root)):
        dd = os.path.join(root, name)
        if not os.path.isdir(dd):
            continue
        p.append(f"<h2>Additional feature output: {html.escape(name)}</h2>")
        jf = os.path.join(dd, "result.json")
        if os.path.isfile(jf):
            txt = json.dumps(json.load(open(jf)), indent=1)
            p.append(f"<details><summary>result.json (numbers for the report)</summary><pre>{html.escape(txt)}</pre></details>")
        for f in sorted(os.listdir(dd)):
            if f.lower().endswith(".png"):
                p.append(f"<h3>{html.escape(f)}</h3><img src='extras/{name}/{f}'>")
    return p


def build_html(out, rows, syn, made):
    p = build_core(out, rows, syn, made)
    p += seq_section(out)
    p += extras_section(out)
    p.append("</body></html>")
    open(os.path.join(out, "report.html"), "w", encoding="utf-8").write("\n".join(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--no_open", action="store_true")
    a = ap.parse_args()
    rp = os.path.join(a.out, "results.json")
    rows = json.load(open(rp)) if os.path.exists(rp) else []
    sp = os.path.join(a.out, "synthetic.json")
    syn = json.load(open(sp)) if os.path.exists(sp) else None
    if syn is None:
        print("(synthetic.json not found: run synthetic_tests.py --out", a.out, "to include DLT/warp tests)")
    made = make_panels(a.out, rows) if rows else {}
    build_html(a.out, rows, syn, made)
    path = os.path.abspath(os.path.join(a.out, "report.html"))
    print("Wrote", path)
    if not a.no_open:
        webbrowser.open("file:///" + path.replace("\\", "/"))


if __name__ == "__main__":
    main()
