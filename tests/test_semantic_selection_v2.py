"""Regression checks for the observed score/decision contradictions."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.select_candidates_v2 import parse_score, parse_decision, score_prompt, call_checked


def must_fail(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return
    raise AssertionError("Contradiction was accepted")


def main():
    target = {"id": "t01_c01", "role": "target", "target_score": 100, "aux_score": 0, "reason": "Whole object."}
    part = {"id": "t02_c01", "role": "part", "target_score": 0, "aux_score": 80, "reason": "Visible component."}
    assert parse_score(json.dumps(target), target["id"]) == target
    must_fail(lambda: parse_score(json.dumps(dict(part, target_score=90)), part["id"]))
    must_fail(lambda: parse_score(json.dumps(dict(target, aux_score=True)), target["id"]))
    must_fail(lambda: parse_score(json.dumps(dict(target, target_score=float("nan"))), target["id"]))
    decision = {"decision": "select", "main_id": target["id"], "aux_ids": [part["id"]], "reason": "Whole object and useful part."}
    assert parse_decision(json.dumps(decision), [target, part], 3) == decision
    abstain = {"decision": "abstain", "main_id": None, "aux_ids": [], "reason": "No suitable whole object."}
    must_fail(lambda: parse_decision(json.dumps(abstain), [target, part], 3))
    weak = dict(target, target_score=30)
    assert parse_decision(json.dumps(abstain), [weak, part], 3) == abstain
    must_fail(lambda: parse_decision(json.dumps(decision), [weak, part], 3))
    must_fail(lambda: parse_decision(json.dumps(decision), [target, part], 3, [(target["id"], part["id"])]))
    must_fail(lambda: parse_decision(json.dumps(dict(decision, main_id=part["id"])), [target, part], 3))
    must_fail(lambda: parse_decision(json.dumps(dict(decision, aux_ids=[part["id"], part["id"]])), [target, part], 3))
    candidate = {"candidate_id": "t01_c01", "tag": "entity", "proposed_kind": "object"}
    prompt = score_prompt(candidate)
    assert '"target_score":0' not in prompt and '"main_id":null' not in prompt
    class FakeModel:
        def __init__(self):
            self.prompts = []
        def caption(self, sheet, system, prompt):
            self.prompts.append(prompt)
            return json.dumps(abstain if len(self.prompts) == 1 else decision)
    model = FakeModel()
    parsed, error, attempts = call_checked(model, Path("unused.jpg"), "choose", lambda raw: parse_decision(raw, [target, part], 3))
    assert error is None and parsed == decision and len(attempts) == 2
    assert "Contradiction" in model.prompts[1]
    print("V2 score-role, high-score abstention, duplicate masks, valid abstention and feedback retry checks passed.")


if __name__ == "__main__":
    main()
