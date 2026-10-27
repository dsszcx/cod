"""tools/e0e4/make_review.py - review sheets, blind text-quality sample, failure analysis.

Outputs (under ``<out>``)
-------------------------
``REVIEW_SHEETS/<dataset>/<stem>.png``
    original | GT | E0 | E1 | E2 | E3 | E4 in one row, with the E2/E3 text and
    the stage statuses underneath.  GT appears **only here** (human review), never
    in any inference stage.
``TEXT_QUALITY_REVIEW.csv`` + ``TEXT_QUALITY_REVIEW_KEY.csv``
    the pre-registered 60-image sample from P1 with the two texts anonymised as
    A/B (the key is a separate file: fill the sheet before opening it).
``FAILURE_ANALYSIS.md``
    where the errors are: NO_ROI / NO_TEXT / SAM3 EMPTY counts, text-only
    candidates that E4 refused to add, E4 fallbacks, and false-positive area on
    the NonCAM stratum (the only place a GT pixel is read).

Usage::

    python tools/e0e4/make_review.py --config configs/e0e4_v1.yaml \
        --ids outputs/E0_E4_ZEROTRAIN_V1/manifest_p1.txt
"""
import argparse
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

STAGES = ("E0", "E1", "E2", "E3", "E4")
REVIEW_COLUMNS = ("Original", "GT", "E0", "E1", "E2", "E3", "E4")
TEXT_ROWS = 3
TEXT_LINE_H = 16


def _load(path, color=False):
    import cv2

    if not os.path.isfile(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)


def _fit(img, height):
    import cv2

    h, w = img.shape[:2]
    width = max(1, int(round(w * float(height) / max(1, h))))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _to_bgr(img):
    if img.ndim == 2:
        return np.dstack([img, img, img])
    return img


def _texts(rec):
    if not rec:
        return "<no record>"
    objs = rec.get("objects") or []
    body = "; ".join("%s -> %s" % (o.get("name"), o.get("sam_text")) for o in objs)
    return "%s | %s" % (rec.get("status", "?"), body or "-")


def build_sheet(row, cfg, out_dir, height=160, texts=None, statuses=None):
    """One review row: original | GT | E0..E4 + texts/statuses underneath."""
    import cv2

    root = os.path.abspath(out_dir)
    image_path, gt_path = C.rel_id_to_paths(row, cfg)
    original = _load(image_path, color=True)
    gt = _load(gt_path)
    if original is None or gt is None:
        return None
    panels = [_to_bgr(_fit(original, height)), _to_bgr(_fit(gt, height))]
    placeholders = []
    for stage in STAGES:
        path = os.path.join(root, "predictions", stage, row["dataset"],
                            row["stem"] + ".png")
        p = _load(path)
        if p is None:
            placeholders.append(stage)
            p = np.full(gt.shape, 80, dtype=np.uint8)
        panels.append(_to_bgr(_fit(p, height)))

    gap, label_h = 6, 18
    width = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    text_h = TEXT_ROWS * TEXT_LINE_H + 6
    canvas = np.full((label_h + height + text_h, width, 3), 255, dtype=np.uint8)
    x = 0
    for name, panel in zip(REVIEW_COLUMNS, panels):
        canvas[label_h:label_h + height, x:x + panel.shape[1]] = panel
        cv2.putText(canvas, name, (x + 2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
        x += panel.shape[1] + gap

    y0 = label_h + height + 14
    lines = []
    if texts is not None:
        lines.append("E2: " + texts.get("e2", ""))
        lines.append("E3: " + texts.get("e3", ""))
    if statuses:
        lines.append("status: " + ", ".join("%s=%s" % (k, v)
                                            for k, v in statuses.items()))
    if placeholders:
        lines.append("MISSING: " + ",".join(placeholders))
    max_chars = max(20, int(width / 7))
    for i, line in enumerate(lines[:TEXT_ROWS]):
        cv2.putText(canvas, line[:max_chars], (4, y0 + i * TEXT_LINE_H),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

    out_path = os.path.join(root, "REVIEW_SHEETS", row["dataset"],
                            row["stem"] + ".png")
    C.ensure_dir(os.path.dirname(out_path))
    if not cv2.imwrite(out_path, canvas):
        raise C.StageError("cv2.imwrite failed: %s" % out_path)
    return out_path


# ---------------------------------------------------------------------------
# blind text-quality sample
# ---------------------------------------------------------------------------
def _sample_sizes(sample_size, rows):
    """Proportional per-stratum sample, capped by what actually exists.

    P0 has 30 images but the review asks for 60: without the cap this raised
    StageError("stratum CAMO has 10 candidate(s) but 20 requested") on Gate C.
    """
    counts = Counter(r["group"] for r in rows)
    total = max(1, len(rows))
    target = min(int(sample_size), total)
    sizes = {g: min(counts.get(g, 0),
                    int(round(target * counts.get(g, 0) / float(total))))
             for g in C.GROUPS}
    remaining = target - sum(sizes.values())
    for group in sorted(C.GROUPS, key=lambda g: counts.get(g, 0), reverse=True):
        while remaining > 0 and sizes[group] < counts.get(group, 0):
            sizes[group] += 1
            remaining -= 1
    return {g: max(0, n) for g, n in sizes.items()}


def write_text_review(cfg, out_dir, rows, texts_e2, texts_e3, sample_size, seed):
    root = os.path.abspath(out_dir)
    sizes = _sample_sizes(sample_size, rows)
    picked = C.pick_stratified(rows, sizes, seed)
    rng = random.Random("blind|%d" % seed)
    review, key = [], []
    for i, row in enumerate(picked, 1):
        ident = row["image_id"]
        r2, r3 = texts_e2.get(ident), texts_e3.get(ident)
        if not r2 or not r3:
            continue
        if rng.random() < 0.5:
            a, b, a_branch, b_branch = r2, r3, "e2", "e3"
        else:
            a, b, a_branch, b_branch = r3, r2, "e3", "e2"
        sample_id = "S%03d" % i
        review.append({
            "sample_id": sample_id, "image_id": ident, "group": row["group"],
            "text_A": _texts(a), "text_B": _texts(b),
            "category_correct_A": "", "category_correct_B": "",
            "shape_texture_visible_A": "", "shape_texture_visible_B": "",
            "hallucinated_A": "", "hallucinated_B": "",
            "missed_other_objects_A": "", "missed_other_objects_B": "",
            "correct_abstain_A": "", "correct_abstain_B": "", "notes": "",
        })
        key.append({"sample_id": sample_id, "image_id": ident,
                    "text_A_branch": a_branch, "text_B_branch": b_branch})
    C.write_csv(os.path.join(root, "TEXT_QUALITY_REVIEW.csv"), review)
    C.write_csv(os.path.join(root, "TEXT_QUALITY_REVIEW_KEY.csv"), key)
    with open(os.path.join(root, "TEXT_QUALITY_REVIEW_README.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("# Text quality review (blind A/B)\n\n"
                 "- %d images sampled from the id file, seed `%d`, strata "
                 "proportional to their count.\n"
                 "- Fill `TEXT_QUALITY_REVIEW.csv` FIRST; `..._KEY.csv` reveals "
                 "which text came from E2 (original image) and which from E3 "
                 "(ROI composite).\n"
                 "- Use 1 = yes, 0 = no, blank = unclear.\n"
                 % (len(review), seed))
    return review, key


# ---------------------------------------------------------------------------
# failure analysis
# ---------------------------------------------------------------------------
def _status_table(path):
    if not os.path.isfile(path):
        return {}
    out = defaultdict(Counter)
    for row in C.read_csv(path):
        out[row.get("group", "?")][row.get("status", "?")] += 1
    return out


def write_failure_analysis(cfg, out_dir, rows):
    root = os.path.abspath(out_dir)
    roi = _status_table(os.path.join(root, "ROI_STATUS.csv"))
    e1 = _status_table(os.path.join(root, "sam3", "e1", "SAM3_e1_STATUS.csv"))
    e2 = _status_table(os.path.join(root, "sam3", "e2", "SAM3_e2_STATUS.csv"))
    e3 = _status_table(os.path.join(root, "sam3", "e3", "SAM3_e3_STATUS.csv"))
    e4 = _status_table(os.path.join(root, "E4_STATUS.csv"))

    unmatched = Counter()
    n_unmatched, n_region_tagged = 0, 0
    for row in rows:
        rec = C.read_json(os.path.join(root, "fusions", row["dataset"],
                                      row["stem"] + ".json"))
        for item in (rec or {}).get("unmatched_text", []):
            n_unmatched += 1
            unmatched[str(item.get("object_name"))] += 1
            if item.get("region_id") is not None:
                n_region_tagged += 1

    # NonCAM false-positive area: the ONLY analysis that opens GT masks here.
    import cv2
    import eval_metrics
    fp = defaultdict(list)
    for row in rows:
        if row["group"] != C.GROUP_COD_NONCAM:
            continue
        _img, gt_path = C.rel_id_to_paths(row, cfg)
        gt = eval_metrics.load_gt(gt_path) > 127
        background = max(1, int((~gt).sum()))
        for stage in ("E0", "E4"):
            path = os.path.join(root, "predictions", stage, row["dataset"],
                                row["stem"] + ".png")
            pred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if pred is None:
                continue
            fp[stage].append(float(((pred > 127) & (~gt)).sum()) / background)

    statuses = sorted({s for t in (roi, e1, e2, e3, e4) for c in t.values()
                       for s in c})
    lines = ["# E0-E4 failure analysis", "",
             "All counts come from the stage CSVs, not from a re-run.", ""]

    def table_block(table, title):
        if not table:
            return []
        return (["### %s" % title, "",
                 "| group | " + " | ".join(statuses) + " |",
                 "|" + "---|" * (1 + len(statuses))]
                + ["| %s | %s |" % (g, " | ".join(
                    str(table[g].get(s, 0)) for s in statuses))
                   for g in list(C.GROUPS) + ["ALL"] if g in table] + [""])

    for table, title in ((roi, "ROI (M0) status"), (e1, "E1 box SAM3"),
                         (e2, "E2 text SAM3"), (e3, "E3 text SAM3"),
                         (e4, "E4 fusion")):
        lines += table_block(table, title)

    q2 = C.read_json(os.path.join(root, "captions", "QWEN_e2_SUMMARY.json"), {}) or {}
    q3 = C.read_json(os.path.join(root, "captions", "QWEN_e3_SUMMARY.json"), {}) or {}
    lines += ["### Qwen parsing", "",
              "| branch | images | status counts | retried | objects |",
              "|---|---|---|---|---|",
              "| E2 | %s | %s | %s | %s |" % (q2.get("n_images", "-"),
                                              q2.get("status_counts", "-"),
                                              q2.get("n_retried", "-"),
                                              q2.get("n_objects", "-")),
              "| E3 | %s | %s | %s | %s |" % (q3.get("n_images", "-"),
                                              q3.get("status_counts", "-"),
                                              q3.get("n_retried", "-"),
                                              q3.get("n_objects", "-")),
              "", "### Text-only candidates (matched no box; NOT added to E4)", "",
              "- candidates: **%d** (of which %d had a region_id tag)"
              % (n_unmatched, n_region_tagged),
              "- top labels: " + ", ".join("%s x%d" % (k, v)
                                          for k, v in unmatched.most_common(10)),
              "", "### NonCAM false-positive area (GT used only here)", "",
              "| stage | mean FP area fraction | n |", "|---|---|---|"]
    for stage in ("E0", "E4"):
        vals = fp.get(stage) or []
        lines.append("| %s | %s | %d |"
                     % (stage, ("%.4f" % float(np.mean(vals))) if vals else "-",
                        len(vals)))
    lines += ["", "### Notes", "",
              "- `NO_ROI`: M0 produced no component above the ROI floor, so "
              "E1/E4 fall back to E0 by construction.",
              "- `NO_TEXT`: Qwen abstained or the reply failed the schema; E2/E3 "
              "then write a black mask, which stays in the denominator.",
              "- `FALLBACK_E0*`: no box/text pair reached the IoU rule; the E4 "
              "map is then exactly E0."]
    path = os.path.join(root, "FAILURE_ANALYSIS.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# stage driver
# ---------------------------------------------------------------------------
def _status_map(path, column="status"):
    if not os.path.isfile(path):
        return {}
    return {r["image_id"]: r.get(column, "") for r in C.read_csv(path)}


def run_stage(cfg, out_dir, ids, limit=None, sheets=True):
    root = os.path.abspath(out_dir)
    ecfg = cfg.get("eval") or {}
    seed = int(cfg.get("seed", 20261020))
    sample_size = int(ecfg.get("review_samples", 60))

    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    selected = C.rows_for_ids(rows, ids)
    if limit:
        selected = selected[:int(limit)]

    texts_e2 = {r["image_id"]: r for r in
                C.read_jsonl(os.path.join(root, "captions", "e2.jsonl"))}
    texts_e3 = {r["image_id"]: r for r in
                C.read_jsonl(os.path.join(root, "captions", "e3.jsonl"))}
    if not texts_e2 or not texts_e3:
        raise C.StageError("review needs captions/e2.jsonl and e3.jsonl first")

    status = {
        "roi": _status_map(os.path.join(root, "ROI_STATUS.csv")),
        "e1": _status_map(os.path.join(root, "sam3", "e1", "SAM3_e1_STATUS.csv")),
        "e2": _status_map(os.path.join(root, "sam3", "e2", "SAM3_e2_STATUS.csv")),
        "e3": _status_map(os.path.join(root, "sam3", "e3", "SAM3_e3_STATUS.csv")),
        "e4": _status_map(os.path.join(root, "E4_STATUS.csv")),
    }

    n_sheets = 0
    if sheets:
        for idx, row in enumerate(selected, 1):
            ident = row["image_id"]
            st = {k: status[k].get(ident, "-") for k in ("roi", "e1", "e2", "e3", "e4")}
            path = build_sheet(row, cfg, root,
                               texts={"e2": _texts(texts_e2.get(ident)),
                                      "e3": _texts(texts_e3.get(ident))},
                               statuses=st)
            if path:
                n_sheets += 1
            C.progress(idx, len(selected), "review sheet %s" % ident)

    review, key = write_text_review(cfg, root, selected, texts_e2, texts_e3,
                                    sample_size, seed)
    if len(review) < sample_size:
        print(">>> [review] only %d/%d image(s) available for the blind sample"
              % (len(review), sample_size))
    failure = write_failure_analysis(cfg, root, selected)
    C.write_json(os.path.join(root, "REVIEW_REPORT.json"), {
        "utc": C.utc_now(), "n_images": len(selected), "n_sheets": n_sheets,
        "n_text_review": len(review), "text_review_seed": seed,
        "failure_analysis": os.path.basename(failure),
        "config": {"version": cfg.get("_version"), "sha256": cfg.get("_sha256")},
    })
    print(">>> review: %d sheet(s), %d blind text sample(s) -> %s"
          % (n_sheets, len(review), root))
    return {"n_sheets": n_sheets, "n_text_review": len(review)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="review sheets + blind text sample")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_sheets", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        run_stage(cfg, out_dir, C.read_id_file(opt.ids), limit=opt.limit,
                  sheets=not opt.no_sheets)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
