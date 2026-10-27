"""V3 regression checks: preserve ratings, role-gate ranking, no forced choice."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.select_candidates_v3 import parse_score, parse_decision, score_warnings, rankings, recover_scores


def must_fail(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid data accepted")


def main():
    target = {"id": "t01_c01", "role": "target", "target_score": 85, "aux_score": 70, "reason": "Whole fish."}
    context = {"id": "t02_c01", "role": "context", "target_score": 99, "aux_score": 90, "reason": "Surrounding sand."}
    for score in (target, context):
        assert parse_score(json.dumps(score), score["id"]) == score
        assert score_warnings(score)
    assert rankings([target, context]) == ([target["id"]], [context["id"]])
    decision = {"decision": "select", "main_id": target["id"], "aux_ids": [context["id"]], "reason": "Fish plus surrounding sand."}
    assert parse_decision(json.dumps(decision), [target, context], 3) == decision
    must_fail(lambda: parse_decision(json.dumps(dict(decision, main_id=context["id"], aux_ids=[])), [target, context], 3))
    abstain = {"decision": "abstain", "main_id": None, "aux_ids": [], "reason": "Whole object is not sufficiently camouflaged."}
    assert parse_decision(json.dumps(abstain), [target, context], 3) == abstain
    must_fail(lambda: parse_score(json.dumps(dict(target, aux_score=True)), target["id"]))
    must_fail(lambda: parse_score(json.dumps(dict(target, target_score=float("nan"))), target["id"]))
    must_fail(lambda: parse_score(json.dumps(dict(target, id="unknown")), target["id"]))
    must_fail(lambda: parse_decision(json.dumps(decision), [target, context], 3, [(target["id"], context["id"])]))
    candidates = [{"candidate_id": target["id"]}, {"candidate_id": context["id"]}]
    old = {"score_traces": [
        {"candidate_id": s["id"], "attempts": [
            {"raw": json.dumps(s), "prompt": "original", "error": "old zero-only restriction"},
            {"raw": json.dumps(dict(s, target_score=100)), "prompt": "retry", "error": None}]}
        for s in (target, context)]}
    scores, provenance, errors = recover_scores(old, candidates)
    assert not errors and scores == [target, context]
    assert all(p["attempt_index"] == 0 for p in provenance)
    old["score_traces"][1]["attempts"] = [{"raw": "not json", "prompt": "original"}]
    scores, _, errors = recover_scores(old, candidates)
    assert scores == [target] and len(errors) == 1  # one invalid candidate does not erase the other
    print("V3 score preservation, role-gated ranking, abstention, recovery and invalid-data checks passed.")


if __name__ == "__main__":
    main()
