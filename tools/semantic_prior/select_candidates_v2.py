"""Step 3 v2: single-candidate visual judgments, then a separate decision."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.e0e4.caption_qwen import QwenCaptioner
from tools.semantic_prior.common import extract_json, resolve
from tools.semantic_prior.select_candidates import flatten, make_sheet

VERSION = "single_candidate_then_selection_v2"
MIN_SCORE = 60
SYSTEM = "Inspect visual evidence carefully. Reply with one JSON object. Do not copy an example answer."


def score_prompt(c):
    return (
        "Top panel is the original image. Bottom panel is ONE candidate mask tinted magenta. "
        "A tiny mask may also have a contextual zoom inset. Inspect the tinted pixels, not just the tag. "
        "Judge this candidate only; do not select regions yet.\n"
        "Candidate metadata: " + json.dumps({k: c[k] for k in ("candidate_id", "tag", "proposed_kind")}) + "\n"
        "Output fields (descriptions, NOT example values):\n"
        "id: the exact candidate_id above.\n"
        "role: target, part, context, or unrelated. target means a whole potential camouflaged object; "
        "part means a component of that potential object; context means surrounding environmental region; "
        "unrelated means incorrect or irrelevant mask. Correct the proposed kind when necessary. "
        "A clearly visible animal is not automatically camouflaged.\n"
        "target_score: numeric 0-100 suitability as a whole camouflaged-object prior. "
        "Use zero for part/context/unrelated, which cannot serve as a whole target. "
        "For target role, 60 or above means suitable, below 60 means insufficient evidence or poor mask.\n"
        "aux_score: numeric 0-100 usefulness as a supporting part or environmental prior for the potential "
        "object in the original. 60 or above means useful. Whole-target/unrelated masks get zero here. "
        "A large region is not useful merely because it is background. Parts are supporting evidence, not negative masks.\n"
        "reason: one short sentence describing what the mask actually covers and why it is useful or not. "
        "Scores are subjective relevance ratings, not calibrated probabilities. No markdown."
    )


def parse_score(raw, candidate_id):
    d = extract_json(raw)
    if not isinstance(d, dict) or d.get("id") != candidate_id:
        raise ValueError("id must match this candidate")
    role = d.get("role")
    if role not in ("target", "part", "context", "unrelated"):
        raise ValueError("Invalid role")
    for key in ("target_score", "aux_score"):
        v = d.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100:
            raise ValueError("Each score must be finite numeric 0-100")
    if role != "target" and d["target_score"] != 0:
        raise ValueError("part/context/unrelated cannot be a whole target: target_score must be zero")
    if role in ("target", "unrelated") and d["aux_score"] != 0:
        raise ValueError("Whole target/unrelated is not an auxiliary: aux_score must be zero")
    if not isinstance(d.get("reason"), str) or not d["reason"].strip():
        raise ValueError("A visual reason is required")
    return {k: d[k] for k in ("id", "role", "target_score", "aux_score", "reason")}


def decision_prompt(scores, max_aux, overlaps):
    return (
        "Top is original RGB, lower panels are numbered candidate masks. Tiny regions may have zoom insets. "
        "Make a separate main/auxiliary selection using the image and these prior per-candidate judgments:\n"
        + json.dumps(scores) + "\nNear-duplicate pairs (mask IoU above 0.9): " + json.dumps(overlaps) + "\n"
        "The scores are subjective relevance ratings, not probabilities. Eligible main candidates have role target "
        "and target_score at least 60. Choose one eligible main, or abstain if there are none. "
        "If you disagree with a prior judgment, describe it in reason, but do not silently contradict its scores. "
        "Such disagreement will need a later review, not a fabricated replacement mask. "
        "Do not choose background or parts as main. Output fields, without example values:\n"
        "decision: select or abstain.\nmain_id: an eligible candidate ID for select, JSON null for abstain.\n"
        "aux_ids: up to %d distinct candidate IDs of role part/context with aux_score at least 60, "
        "ordered by usefulness, belonging to or supporting the chosen main object. Avoid duplicate masks. "
        "When abstaining, return no auxiliary IDs.\nreason: one short sentence explaining the decision. "
        "Do not output scores again. Do not output markdown."
    ) % max_aux


def parse_decision(raw, scores, max_aux, overlaps=()):
    d = extract_json(raw)
    if not isinstance(d, dict) or d.get("decision") not in ("select", "abstain"):
        raise ValueError("decision must be select or abstain")
    if "main_id" not in d:
        raise ValueError("Missing main_id")
    by_id = {s["id"]: s for s in scores}
    eligible = {s["id"] for s in scores if s["role"] == "target" and s["target_score"] >= MIN_SCORE}
    main, aux = d["main_id"], d.get("aux_ids")
    if not isinstance(aux, list) or any(not isinstance(a, str) for a in aux):
        raise ValueError("aux_ids must be a list of IDs")
    if len(aux) > max_aux or len(set(aux)) != len(aux):
        raise ValueError("Too many or duplicate auxiliaries")
    if not isinstance(d.get("reason"), str) or not d["reason"].strip():
        raise ValueError("Decision reason required")
    if d["decision"] == "abstain":
        if main is not None or aux:
            raise ValueError("Abstention must have null main and no auxiliaries")
        if eligible:
            raise ValueError("Contradiction: abstention despite suitable target scores; review this decision")
    elif not isinstance(main, str) or main not in eligible:
        raise ValueError("Selected main must be a target scored at least 60")
    for a in aux:
        if a not in by_id or a == main or by_id[a]["role"] not in ("part", "context") or by_id[a]["aux_score"] < MIN_SCORE:
            raise ValueError("Auxiliary must be an eligible part/context, distinct from main")
    chosen = set(aux + ([main] if main else []))
    if any(a in chosen and b in chosen for a, b in overlaps):
        raise ValueError("Chosen regions contain near-duplicate masks")
    return {k: d[k] for k in ("decision", "main_id", "aux_ids", "reason")}


def call_checked(model, sheet, prompt, parser):
    attempts, parsed, error = [], None, None
    for _ in range(2):
        actual_prompt = prompt
        if error:
            actual_prompt += "\nPrevious answer failed this check: " + error + "\nPrevious answer:\n" + attempts[-1]["raw"] + "\nReturn a corrected complete object."
        raw = model.caption(str(sheet), SYSTEM, actual_prompt)
        try:
            parsed, error = parser(raw), None
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        attempts.append({"prompt": actual_prompt, "raw": raw, "error": error})
        if error is None:
            break
    return parsed, error, attempts


def duplicate_pairs(candidates, out):
    import numpy as np
    from PIL import Image
    masks = []
    for c in candidates:
        with Image.open(out / c["mask_path"]) as im:
            masks.append(np.asarray(im.convert("L")) > 0)
    return [(candidates[i]["candidate_id"], candidates[j]["candidate_id"])
            for i in range(len(masks)) for j in range(i)
            if np.logical_and(masks[i], masks[j]).sum() / max(1, np.logical_or(masks[i], masks[j]).sum()) > .9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    root = out / "selection_v2"
    root.mkdir(exist_ok=False)
    rows = C.read_csv(str(out / "PILOT_MANIFEST.csv"))
    model = QwenCaptioner(cfg["paths"]["qwen_model"], max_tokens=700)
    statuses, table, started = Counter(), [], time.time()
    for index, row in enumerate(rows, 1):
        source = C.read_json(str(out / "candidates" / (row["image_id"] + ".json")))
        if not source or C.sha256_file(row["image_path"]) != source["image_sha256"]:
            raise ValueError("Candidate missing or image changed")
        candidates = flatten(source)
        scores, traces, errors, decision = [], [], [], None
        for c in candidates:
            sheet = root / "single_sheets" / row["image_id"] / (c["candidate_id"] + ".jpg")
            make_sheet(row["image_path"], [c], out, sheet)
            score, error, attempts = call_checked(model, sheet, score_prompt(c),
                                                  lambda raw: parse_score(raw, c["candidate_id"]))
            traces.append({"candidate_id": c["candidate_id"], "attempts": attempts})
            if error:
                errors.append(c["candidate_id"] + ": " + error)
            else:
                scores.append(score)
            print(row["image_id"], c["candidate_id"], score, "error=", error, flush=True)
        decision_attempts = []
        overlaps = duplicate_pairs(candidates, out)
        if not candidates:
            status = "NO_CANDIDATES"
        elif errors:
            status = "SCORE_ERROR"
        else:
            sheet = root / "overview" / (row["image_id"] + ".jpg")
            make_sheet(row["image_path"], candidates, out, sheet)
            decision, error, decision_attempts = call_checked(
                model, sheet, decision_prompt(scores, cfg["selection"]["max_aux"], overlaps),
                lambda raw: parse_decision(raw, scores, cfg["selection"]["max_aux"], overlaps))
            if error:
                errors.append(error)
            status = "DECISION_ERROR" if error else "SELECTED" if decision["main_id"] else "ABSTAIN"
        main_id = decision["main_id"] if decision and not errors else None
        aux_ids = decision["aux_ids"] if decision and not errors else []
        if main_id:
            selected = [next(c for c in candidates if c["candidate_id"] == cid) for cid in [main_id] + aux_ids]
            make_sheet(row["image_path"], selected, out, root / "selected_sheets" / (row["image_id"] + ".jpg"))
        target_rank = [s["id"] for s in sorted(scores, key=lambda s: (-s["target_score"], s["id"]))]
        aux_rank = [s["id"] for s in sorted(scores, key=lambda s: (-s["aux_score"], s["id"]))]
        rec = {"image_id": row["image_id"], "group": row["group"], "image_sha256": row["image_sha256"],
               "status": status, "candidates": candidates, "scores": scores, "decision": decision,
               "errors": errors, "score_traces": traces, "decision_attempts": decision_attempts,
               "target_ranking": target_rank, "aux_ranking": aux_rank, "near_duplicate_pairs": overlaps,
               "system_prompt": SYSTEM, "version": VERSION, "min_score": MIN_SCORE,
               "config_sha256": cfg["_sha256"], "script_sha256": C.sha256_file(__file__), "gt_used": False}
        C.write_json(str(root / (row["image_id"] + ".json")), rec)
        for s in scores:
            table.append({"image_id": row["image_id"], **s, "selected_main": s["id"] == main_id,
                          "selected_aux": s["id"] in aux_ids, "target_rank": target_rank.index(s["id"]) + 1,
                          "aux_rank": aux_rank.index(s["id"]) + 1})
        statuses[status] += 1
        print("[%d/%d] %s %s main=%s aux=%s errors=%s" %
              (index, len(rows), row["image_id"], status, main_id, aux_ids, errors), flush=True)
    C.write_csv(str(out / "SELECTION_V2_TABLE.csv"), table)
    C.write_json(str(out / "SELECTION_V2_SUMMARY.json"), {
        "n_images": len(rows), "status_counts": dict(statuses), "n_candidates_scored": len(table),
        "n_selected_aux": sum(r["selected_aux"] for r in table),
        "n_positive_aux_scores": sum(r["aux_score"] > 0 for r in table),
        "elapsed_s": round(time.time()-started, 1), "version": VERSION, "min_score": MIN_SCORE,
        "gt_used": False, "training": False,
        "score_meaning": "Subjective relevance ratings; threshold60 is a pilot rule, not calibrated"})
    if statuses["SCORE_ERROR"] or statuses["DECISION_ERROR"]:
        return 2
    print("SEMANTIC_SELECTION_V2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
