"""eval_metrics.py - one single metric code path for every number in the docs.

Background (Layer-0 evidence, 2026-10):
    `docs/project/项目文档.md` claimed "baseline MAE ~0.18 -> SAM3 0.0673 (-63%)".
    The 0.18 never came from a working MAE.  The historical log printed
        MAE=46.07612228393555
    while the grader kept GT in [0, 255] and the prediction in [0, 1]; note
        46.076 / 255 = 0.1807
    which equals the CAMO-test GT foreground ratio (measured here: 0.181167),
    i.e. that "baseline MAE" was really the average GT intensity.

    This script recomputes the full COD metric set with PySODMetrics V2 (the same
    library family `新metric.py` wraps) for any (pred_dir, gt_dir) pair, so that
    baseline / ablation / prior-on-off numbers all come from ONE code path.

Metric names follow the COD convention:
    S_alpha = sm, E_phi = adpem (adaptive E), meanE = avgem (mean E),
    F_beta^w = wfm, plus maxF_beta(maxfm) and adpf1.

Naming fix (2026-10): the column that used to be printed as ``Ephi`` was
``adpem``, i.e. *adaptive* E-measure, and it was reported next to other papers'
*mean* E.  Both are now emitted (``adpE`` and ``meanE``) so a table can be
compared with either convention; ``新metric.py`` already computed ``avgem``.

Denominator fix (2026-10): ``set(preds) & set(gts)`` silently shrank the
denominator, so a run that failed to write 30 of 250 maps was scored on 220
images.  ``evaluate()`` now raises ``ManifestError`` unless ``--allow_missing``
is passed, and always reports ``n_gt`` / ``n_pred`` / ``n_evaluated``.

Usage
-----
single pair::
    python eval_metrics.py --pair baseline:results/OUR/CAMO:Dataset/TestDataset/CAMO/GT

side-by-side (this is the Layer-0 evidence produced by `results/OUR/CAMO` vs
`results_sam3/SAM3/CAMO`)::
    python eval_metrics.py ^
        --pair baseline:results/OUR/CAMO:Dataset/TestDataset/CAMO/GT ^
        --pair sam3:results_sam3/SAM3/CAMO:Dataset/TestDataset/CAMO/GT ^
        --out outputs/metrics/layer0_camo.json ^
        --csv outputs/metrics/summary.csv

`--pair` syntax is ``NAME:PRED_DIR:GT_DIR``; split is done as
``arg.split(":", 1)`` + ``rest.rsplit(":", 1)`` so Windows drive letters
(``C:\\...``) are handled.

min-max note
------------
`MyTest_SAM3.py` / `MyTrain.py` historically wrote
``(res - res.min()) / (res.max() - res.min()) * 255`` (per-image contrast
stretch) before saving.  Old prediction folders therefore have the stretch
baked in and it cannot be undone here.
=> Both inference scripts are raw-sigmoid by default (``MyTest_SAM3.py`` always
   was; ``MyTest.py`` used to stretch unconditionally).  Re-run inference without
   ``--minmax`` and you can obtain BOTH numbers from the SAME prediction set:
       no stretch  -> `eval_metrics.py`            (fair / standard protocol)
       historical  -> `eval_metrics.py --minmax`   (reproduces the old numbers)
   ``--pred_protocol {auto,raw,minmax}`` states which protocol a folder was
   written with; it is recorded in the JSON/CSV/table so two rows that were not
   measured the same way can never be silently averaged again.
"""
import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# metric backend
# ----------------------------------------------------------------------------
METRIC_NAMES = ["mae", "sm", "wfm", "em", "fm", "f1", "iou", "dice", "pre", "rec"]

REPORT_COLUMNS = [
    ("mae", "MAE", "down"),
    ("sm", "Salpha", "up"),
    ("adpem", "adpE", "up"),
    # 新metric.py already computed avgem; it was simply never reported, so this
    # table's "E" could not be compared with papers that quote *mean* E.
    ("avgem", "meanE", "up"),
    ("wfm", "Fbeta_w", "up"),
    ("adpfm", "adpFbeta", "up"),
    ("maxfm", "maxFbeta", "up"),
    ("adpf1", "adpF1", "up"),
    ("adpiou", "IoU", "up"),
    ("adpdice", "Dice", "up"),
]


def load_metric_module():
    """Import 新metric.py by path so the non-ASCII module name is never imported."""
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "新metric.py")
    if not os.path.isfile(path):
        raise FileNotFoundError("metric backend not found: %s" % path)
    spec = importlib.util.spec_from_file_location("sod_metric_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Recorder:
    """Thin wrapper around GrayscaleMetricRecorderV2 with a stable interface."""

    def __init__(self):
        self._mod = load_metric_module()
        self._rec = self._mod.GrayscaleMetricRecorderV2(metric_names=METRIC_NAMES)

    def step(self, pred_u8, gt_u8, verbose=False):
        self._rec.step(pred_u8, gt_u8)

    def show(self):
        return dict(self._rec.show(num_bits=6))


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
PRED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz")


def index_dir(d, exts, prefer=(".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz")):
    """stem -> path for every file in `d` whose extension is in `exts`."""
    if not os.path.isdir(d):
        raise NotADirectoryError("directory not found: %s" % d)
    found = {}
    for f in sorted(os.listdir(d)):
        stem, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext not in exts:
            continue
        old = found.get(stem)
        if old is None or prefer.index(ext) < prefer.index(os.path.splitext(old)[1].lower()):
            found[stem] = os.path.join(d, f)
    return found


def load_pred(path, minmax):
    """Load a prediction map -> uint8 in [0, 255] (PySODMetrics requirement)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        p = np.load(path).astype(np.float32).squeeze()
    elif ext == ".npz":
        with np.load(path, allow_pickle=True) as d:
            p = d[sorted(d.keys())[0]].astype(np.float32).squeeze()
    else:
        p = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if p is None:
            raise IOError("cannot read prediction: %s" % path)
    if p.dtype == np.uint8:
        if not minmax:
            return p
        f = p.astype(np.float32) / 255.0
    else:
        f = p.astype(np.float32)
        if float(f.max()) > 1.5:          # stored on a 0..255 scale
            f = f / 255.0
    f = np.clip(f, 0.0, 1.0)
    if minmax:
        f = (f - float(f.min())) / (float(f.max()) - float(f.min()) + 1e-8)
    return np.clip(np.round(f * 255.0), 0, 255).astype(np.uint8)


def load_gt(path):
    """Load a GT mask -> uint8 with values in {0, 255} (PySODMetrics requirement)."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise IOError("cannot read GT: %s" % path)
    if g.dtype != np.uint8:
        f = g.astype(np.float32)
        if float(f.max()) <= 1.5:
            f = f * 255.0
        return np.clip(np.round(f), 0, 255).astype(np.uint8)
    if int(g.max()) <= 1:                 # 0/1 encoded masks
        return (g.astype(np.uint16) * 255).astype(np.uint8)
    return g


def parse_pair(arg):
    """`NAME:PRED_DIR:GT_DIR` (Windows drive letters are tolerated)."""
    if ":" not in arg:
        raise ValueError("--pair expects NAME:PRED_DIR:GT_DIR, got: %r" % arg)
    name, rest = arg.split(":", 1)
    if ":" not in rest:
        raise ValueError("--pair expects NAME:PRED_DIR:GT_DIR, got: %r" % arg)
    pred_dir, gt_dir = rest.rsplit(":", 1)
    return name.strip(), pred_dir.strip(), gt_dir.strip()


class ManifestError(RuntimeError):
    """A (pred, gt) manifest is incomplete -> the metric would be mis-scoped.

    Raised instead of quietly intersecting the two file lists: the historical
    behaviour scored whatever happened to match, so ``n_images`` in an old JSON
    report is not a trustworthy denominator.
    """


class ProtocolError(RuntimeError):
    """Rows that must not be put in one table (different split / protocol).

    A reader cannot see from ``MAE=0.043`` whether the number came from the
    CAMO test split, the CAMO train split (used as a proxy validation set), or a
    folder that still has the historical min-max stretch baked in.  Anything that
    would average those together is refused instead of printed.
    """


def load_manifest(path):
    """Read ``[{name, pred_dir, gt_dir, dataset, split, pred_protocol}, ...]``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):                 # tolerate {"pairs": [...]}
        data = data.get("pairs", data.get("rows", []))
    if not isinstance(data, list) or not data:
        raise ValueError("%s: expected a non-empty list of rows" % path)
    rows = []
    for i, e in enumerate(data):
        if not isinstance(e, dict):
            raise ValueError("%s: row %d is not an object" % (path, i))
        pred_dir, gt_dir = e.get("pred_dir"), e.get("gt_dir")
        if not pred_dir or not gt_dir:
            raise ValueError("%s: row %d needs pred_dir and gt_dir" % (path, i))
        rows.append({
            "name": e.get("name") or e.get("pair") or "row%d" % i,
            "pred_dir": pred_dir,
            "gt_dir": gt_dir,
            "dataset": e.get("dataset"),
            "split": e.get("split"),
            "pred_protocol": e.get("pred_protocol"),
        })
    return rows


def check_consistent(results, allow_mixed=False):
    """Refuse a table whose rows for one dataset differ in split or protocol.

    Returns the list of problems (empty when consistent); ``allow_mixed=True``
    turns the refusal into a printed warning.
    """
    problems = []
    by_ds = {}
    for r in results:
        by_ds.setdefault(r.get("dataset") or r["pair"], []).append(r)
    for ds, rows in sorted(by_ds.items()):
        for field in ("split", "pred_protocol"):
            vals = sorted({str(r.get(field)) for r in rows})
            if len(vals) > 1:
                problems.append("%s: %d different %s values in one table: %s"
                                % (ds, len(vals), field, ", ".join(vals)))
    if problems and not allow_mixed:
        raise ProtocolError(
            "mixed split / protocol -> the rows are not comparable:\n  "
            + "\n  ".join(problems)
            + "\n  (fix the manifest, or pass --allow_mixed_split to override)")
    return problems


def aggregate(results, group_by):
    """Image-count weighted mean of each metric, grouped by dataset / split.

    ``print_table``'s rows are per-folder means; averaging two folders of 250 and
    1200 images with a plain ``mean`` silently over-weights the small one, which
    is how a CAMO-only trend used to be quoted as the COD10K result.
    """
    if group_by == "none":
        return []
    groups = {}
    for r in results:
        key = r.get(group_by) if r.get(group_by) is not None else r["pair"]
        groups.setdefault(str(key), []).append(r)
    out = []
    for key, rows in sorted(groups.items()):
        n = sum(r["n_images"] for r in rows)
        metrics = {}
        for mk, _, _ in REPORT_COLUMNS:
            vals = [(r["metrics"][mk], r["n_images"])
                    for r in rows if mk in r["metrics"]]
            metrics[mk] = (sum(v * w for v, w in vals) / sum(w for _, w in vals)
                           if vals and sum(w for _, w in vals) else None)
        out.append({"group": key, "n_images": n, "n_folders": len(rows),
                    "metrics": metrics})
    return out


def evaluate(name, pred_dir, gt_dir, minmax, gt_ext=".png", verbose=False,
             allow_missing=False, pred_protocol=None, dataset=None, split=None):
    preds = index_dir(pred_dir, PRED_EXTS)
    gts = index_dir(gt_dir, (gt_ext,))
    stems = sorted(set(preds) & set(gts))
    missing_pred = sorted(set(gts) - set(preds))
    missing_gt = sorted(set(preds) - set(gts))
    if not stems:
        raise ManifestError(
            "no matched pairs between %s and %s (pred=%d, gt=%d)"
            % (pred_dir, gt_dir, len(preds), len(gts)))
    if (missing_pred or missing_gt) and not allow_missing:
        raise ManifestError(
            "[%s] manifest incomplete: gt=%d pred=%d evaluated=%d "
            "missing_pred=%d missing_gt=%d (first: %s). "
            "A ratio over %d instead of %d images is not comparable with any "
            "other row; fix the run or pass --allow_missing to score the "
            "intersection on purpose."
            % (name, len(gts), len(preds), len(stems), len(missing_pred),
               len(missing_gt), (missing_pred or missing_gt)[:3],
               len(stems), len(gts)))

    rec = Recorder()
    fg_ratio, pred_mean, gt_mean = [], [], []
    bug_sum, bug_cnt = 0.0, 0
    fg_px_sum = 0.0
    n_resized = 0
    for stem in stems:
        gt = load_gt(gts[stem])
        pred = load_pred(preds[stem], minmax)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            n_resized += 1
        rec.step(pred, gt, verbose=verbose)
        g = gt.astype(np.float32)
        p = pred.astype(np.float32)
        fg_ratio.append(float((g > 127).mean()))
        gt_mean.append(float(g.mean()) / 255.0)
        pred_mean.append(float(p.mean()) / 255.0)
        # the historical bug: |pred in [0,1] - gt in [0,255]|
        bug_sum += float(np.abs(p / 255.0 - g).sum())
        bug_cnt += g.size
        fg_px_sum += float((g > 127).sum())

    m = rec.show()
    out = {
        "pair": name,
        "pred_dir": os.path.abspath(pred_dir),
        "gt_dir": os.path.abspath(gt_dir),
        "minmax": bool(minmax),
        # which protocol the *folder* was written with (independent of what this
        # invocation re-applies); "auto" is resolved to raw/minmax_baked here.
        "pred_protocol": pred_protocol or ("minmax_baked" if minmax else "raw"),
        "dataset": dataset,
        "split": split,
        "n_images": len(stems),          # == n_evaluated (kept for old readers)
        "n_gt": len(gts),
        "n_pred": len(preds),
        "n_evaluated": len(stems),
        "allow_missing": bool(allow_missing),
        "n_resized": n_resized,
        "n_missing_pred": len(missing_pred),
        "n_missing_gt": len(missing_gt),
        "gt_fg_ratio": float(np.mean(fg_ratio)),
        "gt_fg_ratio_pixelwise": fg_px_sum / max(bug_cnt, 1),
        "gt_mean_intensity": float(np.mean(gt_mean)),
        "pred_mean": float(np.mean(pred_mean)),
        # sanity: reproduces the buggy historical "MAE" (mean |pred/255 - gt_255|)
        "mae_bug_unnormalized_gt": bug_sum / max(bug_cnt, 1),
        "metrics": {k: float(m[k]) for k, _, _ in REPORT_COLUMNS if k in m},
        "metrics_all": {k: (float(v) if np.isscalar(v) else None) for k, v in m.items()},
    }
    return out


def print_table(results):
    cols = REPORT_COLUMNS
    head = ["pair", "n/nGT", "minmax", "protocol"] + [c[1] for c in cols] \
        + ["GTfg%", "GTfgPx%", "GTmean", "MAE_bug"]
    widths = [max(len(head[0]), max(len(r["pair"]) for r in results)), 9, 6, 12]
    widths += [max(len(c[1]), 7) for c in cols] + [7, 8, 7, 8]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in results:
        cells = [r["pair"].ljust(widths[0]),
                 ("%d/%d" % (r["n_images"], r.get("n_gt", r["n_images"]))).rjust(widths[1]),
                 ("yes" if r["minmax"] else "no").rjust(widths[2]),
                 str(r.get("pred_protocol") or "?").rjust(widths[3])]
        for i, (key, _, _) in enumerate(cols):
            v = r["metrics"].get(key)
            cells.append(("%.4f" % v).rjust(widths[4 + i]) if v is not None else "n/a".rjust(widths[4 + i]))
        cells.append(("%.2f" % (r["gt_fg_ratio"] * 100)).rjust(widths[-4]))
        cells.append(("%.2f" % (r["gt_fg_ratio_pixelwise"] * 100)).rjust(widths[-3]))
        cells.append(("%.4f" % r["gt_mean_intensity"]).rjust(widths[-2]))
        cells.append(("%.3f" % r["mae_bug_unnormalized_gt"]).rjust(widths[-1]))
        print("  ".join(cells))
    print("-" * len(line))
    print("minmax   = the stretch re-applied by THIS invocation (--minmax)")
    print("protocol = the protocol the prediction FOLDER was written with; a row with")
    print("           'minmax_baked' next to a 'raw' row is not a like-for-like comparison")
    print("n/nGT    = images actually scored / GT images available; a mismatch means")
    print("           the run is incomplete (and the row is NOT comparable)")
    print("GTfg%    = image-averaged fraction of GT pixels >127 (CAMO-test: 18.12%)")
    print("GTfgPx%  = pixel-weighted foreground ratio (differs because image sizes vary)")
    print("MAE_bug = mean |pred/255 - gt_255| computed with an UN-normalized GT = the historical")
    print("          bug; it is ~= GTfgPx% x 255, i.e. an average GT intensity - never a MAE.")
    print("          (0.1807 x 255 = 46.08 in 模型pth/SAM3/log.log == the image-averaged GT ratio.)\n")


CSV_HEADER = ["pair", "dataset", "split", "pred_protocol",
              "n_images", "n_gt", "minmax", "MAE", "Salpha", "adpE", "meanE",
              "Fbeta_w", "adpFbeta", "maxFbeta", "adpF1", "IoU", "Dice",
              "GTfg_ratio", "GTfgPx_ratio", "GTmean", "MAE_bug"]


def append_csv(path, results):
    """Append rows, but only to a CSV written with exactly this schema.

    The old writer appended blindly: a run made before the ``meanE`` /
    ``n_gt`` / ``protocol`` columns existed produced a file where column 9 means
    two different things on two different rows, and every later pivot was wrong
    without raising.  A mismatching header is now a hard error (exit code 3).
    """
    import csv
    file_existed = os.path.exists(path) and os.path.getsize(path) > 0
    if file_existed:
        with open(path, newline="", encoding="utf-8-sig") as f:
            old_header = next(csv.reader(f), None)
        if old_header != CSV_HEADER:
            raise ProtocolError(
                "CSV schema mismatch for %s\n  existing: %s\n  current : %s\n"
                "  append would silently mix columns -> write to a new --csv path"
                % (path, old_header, CSV_HEADER))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not file_existed:
            w.writerow(CSV_HEADER)
        for r in results:
            row = [r["pair"], r.get("dataset") or "", r.get("split") or "",
                   r.get("pred_protocol") or "",
                   r["n_images"], r.get("n_gt", r["n_images"]), int(r["minmax"])]
            row += [("%.6f" % r["metrics"][k]) if k in r["metrics"] else "" for k, _, _ in REPORT_COLUMNS]
            row += ["%.6f" % r["gt_fg_ratio"], "%.6f" % r["gt_fg_ratio_pixelwise"],
                    "%.6f" % r["gt_mean_intensity"], "%.6f" % r["mae_bug_unnormalized_gt"]]
            w.writerow(row)
    print("[csv] appended %d row(s) -> %s" % (len(results), os.path.abspath(path)))


def print_aggregate(rows, group_by):
    """Print the image-count weighted table produced by ``aggregate``."""
    if not rows:
        return
    cols = REPORT_COLUMNS
    head = [group_by, "n", "folders"] + [c[1] for c in cols]
    widths = [max(len(head[0]), max(len(r["group"]) for r in rows)), 7, 8]
    widths += [max(len(c[1]), 7) for c in cols]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print("\n[aggregate by %s] (image-count weighted)" % group_by)
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [r["group"].ljust(widths[0]), str(r["n_images"]).rjust(widths[1]),
                 str(r["n_folders"]).rjust(widths[2])]
        for i, (key, _, _) in enumerate(cols):
            v = r["metrics"].get(key)
            cells.append(("%.4f" % v).rjust(widths[3 + i])
                         if v is not None else "n/a".rjust(widths[3 + i]))
        print("  ".join(cells))


def main():
    p = argparse.ArgumentParser(
        description="Offline COD metric evaluation (PySODMetrics V2) - one code path for all numbers.")
    p.add_argument("--pair", action="append", default=[], metavar="NAME:PRED_DIR:GT_DIR",
                   help="repeatable; e.g. --pair sam3:results_sam3/SAM3/CAMO:Dataset/TestDataset/CAMO/GT")
    p.add_argument("--manifest", type=str, default=None,
                   help="JSON list of {name, pred_dir, gt_dir, dataset, split, pred_protocol}; "
                        "the only supported way to put several splits in one run")
    p.add_argument("--name", type=str, default="run")
    p.add_argument("--pred_dir", type=str, default=None, help="single-pair convenience mode")
    p.add_argument("--gt_dir", type=str, default=None)
    p.add_argument("--gt_ext", type=str, default=".png")
    p.add_argument("--minmax", action="store_true",
                   help="re-apply per-image min-max stretch (reproduces the historical protocol)")
    p.add_argument("--pred_protocol", choices=["auto", "raw", "minmax"], default="auto",
                   help="protocol the prediction FOLDER was written with "
                        "(auto = follow --minmax); recorded in json/csv/table")
    p.add_argument("--group_by", choices=["none", "dataset", "split"], default="none",
                   help="also print an image-count weighted mean (never a plain mean "
                        "over folders of different size)")
    p.add_argument("--allow_mixed_split", action="store_true",
                   help="print a warning instead of refusing a table whose rows for one "
                        "dataset come from different splits / protocols")
    p.add_argument("--out", type=str, default=None, help="write a JSON report here")
    p.add_argument("--csv", type=str, default=None, help="append rows to this CSV (for the ablation table)")
    p.add_argument("--allow_missing", action="store_true",
                   help="score the intersection when GT and prediction file lists "
                        "differ (default: abort -- a quietly shrunk denominator made "
                        "old rows incomparable)")
    opt = p.parse_args()

    # Which protocol the folders were written with, and therefore whether the
    # min-max stretch has to be re-applied before scoring.
    base_protocol = "minmax_baked" if opt.minmax else "raw"
    if opt.pred_protocol != "auto":
        base_protocol = "minmax_baked" if opt.pred_protocol == "minmax" else "raw"
        if (base_protocol == "minmax_baked") != opt.minmax:
            print(">>> [protocol] --pred_protocol %s overrides --minmax "
                  "(load with minmax=%s)"
                  % (opt.pred_protocol, base_protocol == "minmax_baked"))

    pairs = []
    for a in opt.pair:
        name, pred_dir, gt_dir = parse_pair(a)
        pairs.append({"name": name, "pred_dir": pred_dir, "gt_dir": gt_dir,
                      "dataset": None, "split": None, "pred_protocol": None})
    if opt.pred_dir and opt.gt_dir:
        pairs.append({"name": opt.name, "pred_dir": opt.pred_dir,
                      "gt_dir": opt.gt_dir, "dataset": None, "split": None,
                      "pred_protocol": None})
    if opt.manifest:
        try:
            pairs.extend(load_manifest(opt.manifest))
        except (OSError, ValueError) as exc:
            p.error("cannot read --manifest: %s" % exc)
    if not pairs:
        p.error("nothing to evaluate: pass --pair NAME:PRED_DIR:GT_DIR "
                "(or --pred_dir/--gt_dir, or --manifest)")

    print(">>> eval_metrics.py  (backend: 新metric.py / PySODMetrics V2)")
    print(">>> folder protocol: %s | this run re-applies minmax: %s"
          % (base_protocol, "yes" if opt.minmax else "no"))
    print(">>> rows: %d (group_by=%s)" % (len(pairs), opt.group_by))
    if opt.minmax:
        print("    note: --minmax reproduces the historical (stretched) protocol and is")
        print("          only comparable with other rows evaluated the same way.")

    results = []
    for row in pairs:
        row_protocol = row.get("pred_protocol") or base_protocol
        row_minmax = row_protocol == "minmax_baked"
        print("\n>>> evaluating [%s]  dataset=%s split=%s protocol=%s"
              % (row["name"], row.get("dataset"), row.get("split"), row_protocol))
        print("    pred: %s\n    gt  : %s" % (row["pred_dir"], row["gt_dir"]))
        try:
            r = evaluate(row["name"], row["pred_dir"], row["gt_dir"], row_minmax,
                         gt_ext=opt.gt_ext, verbose=False,
                         allow_missing=opt.allow_missing,
                         pred_protocol=row_protocol,
                         dataset=row.get("dataset"), split=row.get("split"))
        except ManifestError as exc:
            # A wrong denominator is a *result*, not a warning: never report the
            # remaining rows as if this pair had been evaluated.
            print("    !! MANIFEST ERROR: %s" % exc)
            return 2
        except (NotADirectoryError, FileNotFoundError) as exc:
            # a typo'd path is a different failure from a short prediction set
            print("    !! skipped: %s" % exc)
            continue
        print("    manifest: gt=%d pred=%d evaluated=%d"
              % (r["n_gt"], r["n_pred"], r["n_evaluated"]))
        results.append(r)

    if not results:
        print("\n!! no pair could be evaluated.")
        return 1

    try:
        problems = check_consistent(results, allow_mixed=opt.allow_mixed_split)
    except ProtocolError as exc:
        print("\n!! REFUSED: %s" % exc)
        return 3
    for msg in problems:
        print("[warn] mixed: %s" % msg)

    print_table(results)
    if opt.group_by != "none":
        print_aggregate(aggregate(results, opt.group_by), opt.group_by)
    for r in results:
        if r["n_missing_pred"] or r["n_missing_gt"] or r["n_resized"]:
            print("[warn] %s: missing_pred=%d missing_gt=%d resized=%d"
                  % (r["pair"], r["n_missing_pred"], r["n_missing_gt"], r["n_resized"]))

    if opt.out:
        d = os.path.dirname(os.path.abspath(opt.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(opt.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("[json] %s" % os.path.abspath(opt.out))
    if opt.csv:
        try:
            append_csv(opt.csv, results)
        except ProtocolError as exc:
            print("!! CSV REFUSED: %s" % exc)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
