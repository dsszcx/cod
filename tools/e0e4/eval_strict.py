"""tools/e0e4/eval_strict.py - strict 4-metric evaluation of E0-E4 (no silent repair).

WHAT "STRICT" MEANS HERE
------------------------
* Every stage must have **exactly one** PNG per image in the id file.  A missing
  file is a HOLD (exit 3), never a smaller denominator.  (``eval_metrics.py``
  already refuses a short prediction folder; this stage refuses before it even
  starts, and it does so for *all* stages at once.)
* Prediction and GT must have **identical HxW**.  `eval_metrics.evaluate` can
  resize; here a resize would change the number, so it is a HOLD.
* The metric backend is the same one `eval_metrics.py` uses:
  ``新metric.GrayscaleMetricRecorderV2(metric_names=METRIC_NAMES)``, i.e.
  MAE / S_alpha / wF_beta / mean E (avgem) + adaptive E (adpem), IoU, Dice.
* Macro average over images per stratum (CAMO / COD10K-CAM / COD10K-NonCAM /
  ALL); no image weighting, no per-dataset tuning.
* Paired deltas (E1-E0, E3-E2, E4-E0, E4-E3) with a bootstrap 95% CI over
  images; a delta whose CI crosses 0 is labelled ``n.s.`` in the table.

OUTPUTS (under ``<out>/metrics``)
--------------------------------
    per_image_metrics.csv      one row per (image, stage)
    aggregate_metrics.csv/.json
    metrics_table.md           the 4 main metrics + extras
    paired_deltas.csv
and, when the run is refused, ``<out>/EVAL_HOLD.json`` + ``EVAL_HOLD.md``.

Usage::

    python tools/e0e4/eval_strict.py --config configs/e0e4_v1.yaml \
        --ids outputs/E0_E4_ZEROTRAIN_V1/manifest_p0.txt
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

STAGES = ("E0", "E1", "E2", "E3", "E4")
MAIN_METRICS = ("mae", "sm", "avgem", "wfm")
EXTRA_METRICS = ("adpem", "adpiou", "adpdice", "maxfm", "maxem", "pre", "rec")
# lower is better for MAE; everything else is higher-is-better
METRIC_DIRECTION = {"mae": "down"}
PAIRS = (("E1", "E0"), ("E3", "E2"), ("E4", "E0"), ("E4", "E3"), ("E2", "E0"))


def metric_direction(name):
    return METRIC_DIRECTION.get(name, "up")


def bootstrap_ci(values, n=2000, alpha=0.05, seed=20261020):
    """Mean and percentile bootstrap CI of a paired-difference sample."""
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if vals.size == 0:
        return None, None, None
    mean = float(vals.mean())
    rng = random.Random(int(seed))
    k = int(n)
    boots = []
    for _ in range(k):
        sample = [vals[rng.randrange(vals.size)] for _ in range(vals.size)]
        boots.append(float(np.mean(sample)))
    boots.sort()
    lo = boots[int((alpha / 2.0) * (k - 1))]
    hi = boots[int((1.0 - alpha / 2.0) * (k - 1))]
    return mean, lo, hi


def hold(issues, out_dir, rows=()):
    """Write the HOLD report (no metrics) and return the exit code."""
    root = os.path.abspath(out_dir)
    C.ensure_dir(root)
    payload = {"status": "HOLD", "n_issues": len(issues), "issues": list(issues),
               "utc": C.utc_now(), "n_manifest_rows": len(list(rows))}
    C.write_json(os.path.join(root, "EVAL_HOLD.json"), payload)
    lines = ["# EVAL HOLD (no metric was computed)", "",
             "The strict evaluation refused to run: %d issue(s)." % len(issues), ""]
    lines += ["- %s" % msg for msg in issues[:200]]
    if len(issues) > 200:
        lines.append("- ... (%d more)" % (len(issues) - 200))
    with open(os.path.join(root, "EVAL_HOLD.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(">>> EVAL HOLD: %d issue(s) -> %s/EVAL_HOLD.md" % (len(issues), root))
    return 3


# ---------------------------------------------------------------------------
# inventory / loading
# ---------------------------------------------------------------------------
def pred_path(out_dir, stage, row):
    return os.path.join(os.path.abspath(out_dir), "predictions", stage,
                        row["dataset"], row["stem"] + ".png")


def inventory(rows, stages, out_dir):
    """Every (stage, image) must have exactly one ``<stem>.png`` -> (paths, issues)."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz")
    paths, issues = {}, []
    for stage in stages:
        root = os.path.join(os.path.abspath(out_dir), "predictions", stage)
        if not os.path.isdir(root):
            issues.append("stage %s: prediction dir missing: %s" % (stage, root))
            continue
        for dataset in sorted({r["dataset"] for r in rows}):
            d = os.path.join(root, dataset)
            if not os.path.isdir(d):
                issues.append("stage %s: dataset dir missing: %s" % (stage, d))
                continue
            found = defaultdict(list)
            for fn in sorted(os.listdir(d)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() in exts:
                    found[stem].append(fn)
            for row in rows:
                if row["dataset"] != dataset:
                    continue
                files = found.get(row["stem"], [])
                key = row["image_id"]
                if len(files) != 1:
                    issues.append("stage %s: %d file(s) for %s (expected exactly 1 PNG)"
                                  % (stage, len(files), key))
                    continue
                if files[0] != row["stem"] + ".png":
                    issues.append("stage %s: %s is not a PNG (%s)"
                                  % (stage, key, files[0]))
                    continue
                paths[(stage, key)] = os.path.join(d, files[0])
    return paths, issues


def load_pred_u8(path):
    """Same loader as `eval_metrics.py` (no min-max: raw protocol)."""
    import eval_metrics

    return eval_metrics.load_pred(path, minmax=False)


def load_gt_u8(path):
    import eval_metrics

    return eval_metrics.load_gt(path)


def compute_per_image(rows, stages, paths, cfg, out_dir):
    """One metric record per (image, stage); stops collecting on shape mismatch."""
    import eval_metrics

    mod = eval_metrics.load_metric_module()
    per_image, issues = [], []
    for idx, row in enumerate(rows, 1):
        _img, gt_path = C.rel_id_to_paths(row, cfg)
        gt = load_gt_u8(gt_path)
        for stage in stages:
            path = paths.get((stage, row["image_id"]))
            if path is None:
                continue
            pred = load_pred_u8(path)
            if pred.shape != gt.shape:
                issues.append("%s/%s: pred %s vs GT %s (resize is forbidden)"
                              % (stage, row["image_id"], pred.shape, gt.shape))
                continue
            rec = mod.GrayscaleMetricRecorderV2(
                metric_names=eval_metrics.METRIC_NAMES)
            rec.step(pred, gt)
            # num_bits=6: the default (3) would already round MAE 0.0054 -> 0.005
            vals = {k: round(float(v), 4) for k, v in rec.show(num_bits=6).items()}
            out = {"image_id": row["image_id"], "group": row["group"],
                   "stage": stage,
                   "pred_fg": round(float((pred > 127).mean()), 6),
                   "gt_fg": round(float((gt > 127).mean()), 6)}
            out.update({k: vals.get(k) for k in MAIN_METRICS + EXTRA_METRICS})
            per_image.append(out)
        if idx % 25 == 0 or idx == len(rows):
            C.progress(idx, len(rows), "strict metrics")
    return per_image, issues


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _mean(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate_rows(per_image, metrics=MAIN_METRICS + EXTRA_METRICS):
    """Macro (per-image) mean per (stage, group) and per (stage, ALL)."""
    buckets = defaultdict(list)
    for row in per_image:
        buckets[(row["stage"], row["group"])].append(row)
        buckets[(row["stage"], "ALL")].append(row)
    out = []
    for stage in STAGES:
        for group in list(C.GROUPS) + ["ALL"]:
            rows = buckets.get((stage, group))
            if not rows:
                continue
            rec = {"stage": stage, "group": group, "n": len(rows)}
            rec.update({m: _mean(r.get(m) for r in rows) for m in metrics})
            out.append(rec)
    return out


def status_ok_ids(out_dir, stage):
    """Image ids whose SAM3/fusion stage actually produced something (subset stats)."""
    root = os.path.abspath(out_dir)
    paths = {
        "E1": os.path.join(root, "sam3", "e1", "SAM3_e1_STATUS.csv"),
        "E2": os.path.join(root, "sam3", "e2", "SAM3_e2_STATUS.csv"),
        "E3": os.path.join(root, "sam3", "e3", "SAM3_e3_STATUS.csv"),
        "E4": os.path.join(root, "E4_STATUS.csv"),
    }
    path = paths.get(stage)
    if not path or not os.path.isfile(path):
        return None
    return {r["image_id"] for r in C.read_csv(path) if r.get("status") == "OK"}


def compute_deltas(per_image, pairs=PAIRS, n=2000, alpha=0.05, seed=20261020):
    """Paired per-image deltas with a bootstrap CI (same images, same stage pair)."""
    by_key = {(r["image_id"], r["stage"]): r for r in per_image}
    image_ids = sorted({r["image_id"] for r in per_image})
    out = []
    for a, b in pairs:
        for metric in MAIN_METRICS + ("adpem", "adpiou", "adpdice"):
            diffs = []
            for ident in image_ids:
                ra, rb = by_key.get((ident, a)), by_key.get((ident, b))
                if not ra or not rb:
                    continue
                va, vb = ra.get(metric), rb.get(metric)
                if va is None or vb is None:
                    continue
                diffs.append(float(va) - float(vb))
            mean, lo, hi = bootstrap_ci(diffs, n=n, alpha=alpha, seed=seed)
            if mean is None:
                continue
            significant = (lo > 0.0) or (hi < 0.0)
            better = None
            if metric_direction(metric) == "down":
                better = mean < 0
            else:
                better = mean > 0
            out.append({"pair": "%s-%s" % (a, b), "metric": metric,
                        "direction": metric_direction(metric),
                        "n_pairs": len(diffs),
                        "mean_delta": round(mean, 6),
                        "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                        "significant": bool(significant),
                        "favours_a": bool(better) if significant else None,
                        "note": "" if significant else "n.s."})
    return out


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
HEAD = ("stage", "group", "n", "mae", "sm", "avgem", "wfm",
        "adpem", "adpiou", "adpdice")


def write_metrics_table(agg, deltas, path):
    lines = ["# Strict COD metrics (macro average over images)", "",
             "`mae`/`sm`/`avgem`/`wfm` are the four pre-registered metrics; "
             "`adpem`/`adpiou`/`adpdice` are extra columns.", "",
             "| " + " | ".join(HEAD) + " |",
             "|" + "---|" * len(HEAD)]
    for row in agg:
        cells = []
        for key in HEAD:
            v = row.get(key)
            cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if deltas:
        lines += ["", "## Paired deltas (per image; bootstrap 95% CI)", "",
                  "| pair | metric | n | mean delta | 95% CI | significant |",
                  "|---|---|---|---|---|---|"]
        for d in deltas:
            lines.append("| %s | %s | %d | %.4f | [%.4f, %.4f] | %s |"
                         % (d["pair"], d["metric"], d["n_pairs"],
                            d["mean_delta"], d["ci_lo"], d["ci_hi"],
                            "yes" if d["significant"] else "n.s."))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def coverage(out_dir, stages):
    """Status counts per stage, read from the stage CSVs (no recomputation)."""
    root = os.path.abspath(out_dir)
    files = {"E1": os.path.join(root, "sam3", "e1", "SAM3_e1_STATUS.csv"),
             "E2": os.path.join(root, "sam3", "e2", "SAM3_e2_STATUS.csv"),
             "E3": os.path.join(root, "sam3", "e3", "SAM3_e3_STATUS.csv"),
             "E4": os.path.join(root, "E4_STATUS.csv")}
    out = {}
    for stage in stages:
        path = files.get(stage)
        if not path or not os.path.isfile(path):
            continue
        counts = {}
        for row in C.read_csv(path):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        out[stage] = counts
    roi = os.path.join(root, "ROI_STATUS.csv")
    if os.path.isfile(roi):
        counts = {}
        for row in C.read_csv(roi):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        out["ROI"] = counts
    return out


def run(cfg, out_dir, ids, stages=STAGES, overwrite=False):
    """Strict eval.  Returns 0 on success, 3 on HOLD (no numbers written)."""
    C.guard_config(cfg)
    ecfg = cfg.get("eval") or {}
    stages = [s.upper() for s in stages]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise C.StageError("unknown stage(s): %s" % ", ".join(bad))

    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    selected = C.rows_for_ids(rows, ids)

    root = os.path.abspath(out_dir)
    paths, issues = inventory(selected, stages, root)
    if issues:
        return hold(issues, root, selected)

    per_image, shape_issues = compute_per_image(selected, stages, paths, cfg, root)
    if shape_issues:
        return hold(shape_issues, root, selected)

    agg = aggregate_rows(per_image)
    subset_agg = []
    for stage in stages:
        ok_ids = status_ok_ids(root, stage)
        if not ok_ids or len(ok_ids) >= len(selected):
            continue
        sub = [r for r in per_image
               if r["stage"] == stage and r["image_id"] in ok_ids]
        for rec in aggregate_rows(sub):
            rec["subset"] = "sam3_valid"
            rec["n_total"] = len(selected)
            subset_agg.append(rec)

    deltas = compute_deltas(per_image, n=int(ecfg.get("bootstrap_n", 2000)),
                            alpha=float(ecfg.get("bootstrap_alpha", 0.05)),
                            seed=int(ecfg.get("bootstrap_seed", 20261020)))
    metrics_dir = os.path.join(root, "metrics")
    C.ensure_dir(metrics_dir)
    C.write_csv(os.path.join(metrics_dir, "per_image_metrics.csv"), per_image,
                columns=["image_id", "group", "stage", "pred_fg", "gt_fg"]
                        + list(MAIN_METRICS + EXTRA_METRICS))
    C.write_csv(os.path.join(metrics_dir, "aggregate_metrics.csv"), agg,
                columns=list(HEAD) + list(EXTRA_METRICS))
    C.write_json(os.path.join(metrics_dir, "aggregate_metrics.json"),
                 {"rows": agg, "subsets": subset_agg, "n_images": len(selected),
                  "stages": stages,
                  "config": {"version": cfg.get("_version"),
                             "sha256": cfg.get("_sha256")}})
    if subset_agg:
        C.write_csv(os.path.join(metrics_dir, "aggregate_subsets.csv"), subset_agg,
                    columns=["stage", "group", "subset", "n", "n_total"]
                            + list(MAIN_METRICS + EXTRA_METRICS))
    C.write_csv(os.path.join(metrics_dir, "paired_deltas.csv"), deltas)
    write_metrics_table(agg, deltas, os.path.join(metrics_dir, "metrics_table.md"))
    C.write_json(os.path.join(root, "EVAL_REPORT.json"), {
        "status": "OK", "utc": C.utc_now(), "n_images": len(selected),
        "denominator_ok": True, "stages": stages,
        "groups": {g: sum(1 for r in selected if r["group"] == g) for g in C.GROUPS},
        "coverage": coverage(root, stages),
        "main_metrics": list(MAIN_METRICS),
        "config": {"version": cfg.get("_version"), "sha256": cfg.get("_sha256")},
    })
    print(">>> eval OK: %d image(s) x %d stage(s) -> %s"
          % (len(selected), len(stages), metrics_dir))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict E0-E4 metrics")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        stages = [s.strip() for s in opt.stages.split(",") if s.strip()]
        return run(cfg, out_dir, C.read_id_file(opt.ids), stages=stages)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
