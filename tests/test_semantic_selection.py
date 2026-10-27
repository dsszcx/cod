"""CPU checks for complete scoring, role constraints, abstention and ranking."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.select_candidates import parse_selection, build_prompt


def main():
    candidates = [{"candidate_id": "t01_c01", "tag": "whole entity", "proposed_kind": "object", "sam3_score": .8},
                  {"candidate_id": "t02_c01", "tag": "visible component", "proposed_kind": "part", "sam3_score": .9}]
    data = {"scores": [{"id": "t01_c01", "role": "target", "target_score": 85, "aux_score": 20},
                       {"id": "t02_c01", "role": "part", "target_score": 10, "aux_score": 90}],
            "main_id": "t01_c01", "aux_ids": ["t02_c01"]}
    parsed = parse_selection(json.dumps(data), candidates, 3)
    assert parsed["target_ranking"] == ["t01_c01", "t02_c01"]
    assert parsed["aux_ranking"] == ["t02_c01", "t01_c01"]
    data["main_id"] = None
    assert parse_selection(json.dumps(data), candidates, 3)["main_id"] is None
    invalid = []
    for key, value in (("main_id", "t02_c01"), ("aux_ids", ["t01_c01"]),
                       ("aux_ids", ["t02_c01", "t02_c01"]), ("scores", data["scores"][:1])):
        invalid.append(dict(data, **{key: value}))
    invalid.append(dict(data, scores=[dict(data["scores"][0], target_score=101), data["scores"][1]]))
    for bad in invalid:
        try:
            parse_selection(json.dumps(bad), candidates, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid selection accepted")
    prompt = build_prompt(candidates, 3)
    assert "SAM3" in prompt and "0-100" in prompt
    assert "frog" not in prompt
    print("Complete scoring, ranking, roles, null-main and invalid selection checks passed.")


if __name__ == "__main__":
    main()
