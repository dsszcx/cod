"""Recover original v2 ratings without zero constraints; rerun decisions only."""
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
from tools.semantic_prior.select_candidates_v2 import call_checked, duplicate_pairs, parse_decision as strict_decision

VERSION = "preserved_raw_scores_role_gated_selection_v3"


def parse_score(raw, cid):
    d = extract_json(raw)
    if not isinstance(d, dict) or d.get("id") != cid:
        raise ValueError("Candidate ID mismatch")
    if d.get("role") not in ("target", "part", "context", "unrelated"):
        raise ValueError("Invalid role")
    for key in ("target_score", "aux_score"):
        v = d.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100:
            raise ValueError("Scores must be finite numeric 0-100")
    if not isinstance(d.get("reason"), str) or not d["reason"].strip():
        raise ValueError("Missing visual reason")
    return {k: d[k] for k in ("id", "role", "target_score", "aux_score", "reason")}


def score_warnings(score):
    warnings = []
    if score["role"] != "target" and score["target_score"] > 0:
        warnings.append("non_target_has_target_score")
    if score["role"] in ("target", "unrelated") and score["aux_score"] > 0:
        warnings.append("non_aux_role_has_aux_score")
    return warnings


def recover_scores(previous, candidates):
    traces = previous.get("score_traces", [])
    ids = [t["candidate_id"] for t in traces]
    if len(ids) != len(set(ids)) or set(ids) != {c["candidate_id"] for c in candidates}:
        raise ValueError("Stored trace IDs do not match candidates")
    by_id = {t["candidate_id"]: t for t in traces}
    scores, provenance, errors = [], [], []
    for c in candidates:
        cid = c["candidate_id"]
        failures = []
        # First structurally valid ORIGINAL answer, not highest score and not
        # last retry: avoids selecting whichever response looks most useful.
        for index, attempt in enumerate(by_id[cid]["attempts"]):
            try:
                score = parse_score(attempt["raw"], cid)
            except (ValueError, TypeError) as exc:
                failures.append(str(exc))
                continue
            scores.append(score)
            provenance.append({"candidate_id": cid, "attempt_index": index,
                               "raw": attempt["raw"], "original_prompt": attempt["prompt"],
                               "original_error": attempt.get("error"),
                               "warnings": score_warnings(score)})
            break
        else:
            errors.append({"candidate_id": cid, "errors": failures})
    return scores, provenance, errors


def rankings(scores):
    target = sorted((s for s in scores if s["role"] == "target"), key=lambda s: (-s["target_score"], s["id"]))
    aux = sorted((s for s in scores if s["role"] in ("part", "context")), key=lambda s: (-s["aux_score"], s["id"]))
    return [s["id"] for s in target], [s["id"] for s in aux]


def parse_decision(raw, scores, max_aux, overlaps=()):
    d = extract_json(raw)
    # Abstention can disagree with single-candidate ratings. Preserve it and
    # flag for review instead of forcing the model/program to choose a mask.
    if isinstance(d, dict) and d.get("decision") == "abstain":
        if "main_id" not in d or d["main_id"] is not None or d.get("aux_ids") != []:
            raise ValueError("Abstention requires null main and no auxiliary IDs")
        if not isinstance(d.get("reason"), str) or not d["reason"].strip():
            raise ValueError("Abstention needs a visual reason")
        return {k: d[k] for k in ("decision", "main_id", "aux_ids", "reason")}
    return strict_decision(raw, scores, max_aux, overlaps)


def build_prompt(scores, max_aux, overlaps):
    target, aux = rankings(scores)
    return (
        "Top panel is original RGB. Below are numbered magenta candidate masks, with zooms for tiny regions. "
        "Use visual evidence to choose a whole potential camouflaged target and useful supporting regions. "
        "Earlier single-candidate judgments follow. Their roles and subjective ratings may be wrong. "
        "Both original scores are preserved; nonzero scores for an incompatible role are warnings, not eligibility.\n"
        + json.dumps(scores) + "\nTarget-only ranking: " + json.dumps(target)
        + "\nPart/context-only ranking: " + json.dumps(aux)
        + "\nNear-duplicate pairs (IoU>0.9): " + json.dumps(overlaps) + "\n"
        "Only role target with target_score>=60 is eligible as main. Only part/context with aux_score>=60 "
        "is eligible as auxiliary. Never treat context as main just because target_score is high. "
        "60 is an uncalibrated pilot rule, not a probability. You may abstain even if a prior target rating "
        "is high, but explain the visual disagreement. Do not invent or revise scores. "
        "Avoid duplicate selected masks. Parts are supporting evidence, not negative masks. "
        "Output one JSON object with fields described here, NOT example values:\n"
        "decision: select or abstain.\nmain_id: an eligible candidate ID if selecting; JSON null if abstaining.\n"
        "aux_ids: a list of at most %d distinct eligible part/context IDs useful for the chosen main, "
        "ordered by usefulness. Empty if none is useful or if abstaining.\n"
        "reason: one short visual explanation. No markdown."
    ) % max_aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semantic_prior_pilot_v1.yaml")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    out = resolve(cfg["paths"]["out_dir"])
    rows = C.read_csv(str(out / "PILOT_MANIFEST.csv"))
    jobs = []
    for row in rows:
        path = out / "selection_v2" / (row["image_id"] + ".json")
        previous = C.read_json(str(path))
        source = C.read_json(str(out / "candidates" / (row["image_id"] + ".json")))
        if not previous or not source or C.sha256_file(row["image_path"]) != previous["image_sha256"] or source["image_sha256"] != previous["image_sha256"]:
            raise ValueError("Missing inputs or image mismatch")
        candidates = flatten(source)
        if candidates != previous["candidates"]:
            raise ValueError("Candidates changed since v2")
        scores, provenance, errors = recover_scores(previous, candidates)
        jobs.append((row, path, candidates, scores, provenance, errors))
    print("V3 recovered %d/%d original candidate judgments; no score inference rerun." %
          (sum(len(j[3]) for j in jobs), sum(len(j[2]) for j in jobs)), flush=True)
    if args.check_only:
        for row, _, _, scores, _, errors in jobs:
            print(row["image_id"], "scores=", len(scores), "errors=", errors, flush=True)
        return 2 if any(j[5] for j in jobs) else 0
    root = out / "selection_v3"
    root.mkdir(exist_ok=False)
    model = QwenCaptioner(cfg["paths"]["qwen_model"], max_tokens=700)
    statuses, table, started, warning_images = Counter(), [], time.time(), 0
    for index, (row, source_path, candidates, scores, provenance, errors) in enumerate(jobs, 1):
        warnings = [{"id": p["candidate_id"], "code": w} for p in provenance for w in p["warnings"]]
        decision, attempts, error = None, [], None
        overlaps = duplicate_pairs(candidates, out)
        if not candidates:
            status = "NO_CANDIDATES"
        elif not scores:
            status = "SCORE_ERROR"
        else:
            valid_ids = {s["id"] for s in scores}
            usable = [c for c in candidates if c["candidate_id"] in valid_ids]
            sheet = root / "overview" / (row["image_id"] + ".jpg")
            make_sheet(row["image_path"], usable, out, sheet)
            decision, error, attempts = call_checked(model, sheet,
                build_prompt(scores, cfg["selection"]["max_aux"], overlaps),
                lambda raw: parse_decision(raw, scores, cfg["selection"]["max_aux"], overlaps))
            status = "DECISION_ERROR" if error else "SELECTED" if decision["main_id"] else "ABSTAIN"
            if not error and decision["decision"] == "abstain" and any(s["role"] == "target" and s["target_score"] >= 60 for s in scores):
                warnings.append({"code": "abstained_despite_high_target_rating"})
            if errors and not error:
                status = "PARTIAL_" + status
                warnings.append({"code": "some_candidate_scores_unavailable"})
        main_id = decision["main_id"] if decision and not error else None
        aux_ids = decision["aux_ids"] if decision and not error else []
        if main_id:
            chosen = [next(c for c in candidates if c["candidate_id"] == cid) for cid in [main_id] + aux_ids]
            make_sheet(row["image_path"], chosen, out, root / "selected_sheets" / (row["image_id"] + ".jpg"))
        target_rank, aux_rank = rankings(scores)
        C.write_json(str(root / (row["image_id"] + ".json")), {
            "image_id": row["image_id"], "group": row["group"], "image_sha256": row["image_sha256"],
            "status": status, "candidates": candidates, "scores": scores, "score_provenance": provenance,
            "score_source_sha256": C.sha256_file(str(source_path)), "score_recovery_rule": "first structurally valid stored attempt",
            "score_errors": errors, "decision": decision, "decision_error": error, "decision_attempts": attempts,
            "warnings": warnings, "target_ranking": target_rank, "aux_ranking": aux_rank,
            "near_duplicate_pairs": overlaps, "version": VERSION, "gt_used": False,
            "config_sha256": cfg["_sha256"], "script_sha256": C.sha256_file(__file__)})
        for s in scores:
            table.append({"image_id": row["image_id"], **s,
                          "target_rank": target_rank.index(s["id"])+1 if s["id"] in target_rank else "",
                          "aux_rank": aux_rank.index(s["id"])+1 if s["id"] in aux_rank else "",
                          "selected_main": s["id"] == main_id, "selected_aux": s["id"] in aux_ids,
                          "warnings": ";".join(score_warnings(s))})
        statuses[status] += 1
        warning_images += bool(warnings)
        print("[%d/%d] %s %s main=%s aux=%s warnings=%d error=%s" %
              (index, len(jobs), row["image_id"], status, main_id, aux_ids, len(warnings), error), flush=True)
    C.write_csv(str(out / "SELECTION_V3_TABLE.csv"), table)
    summary = {"n_images": len(jobs), "status_counts": dict(statuses), "n_candidates_scored": len(table),
               "n_selected_aux": sum(r["selected_aux"] for r in table), "n_warning_images": warning_images,
               "score_inference_rerun": False, "score_source": "v2 stored raw replies under original v2 prompts",
               "score_recovery_rule": "first structurally valid stored attempt; no numerical/role changes",
               "version": VERSION, "elapsed_s": round(time.time()-started, 1), "gt_used": False,
               "training": False, "min_score": 60, "threshold_calibrated": False}
    C.write_json(str(out / "SELECTION_V3_SUMMARY.json"), summary)
    if any("ERROR" in s or s.startswith("PARTIAL_") for s in statuses):
        print("SEMANTIC_SELECTION_V3_FINISHED_WITH_ERRORS", flush=True)
        return 2
    print("SEMANTIC_SELECTION_V3_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
