"""CPU tests for multi-component tags and deterministic pilot selection."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.common import choose_pilot, normalize_tags
from tools.semantic_prior.generate_tags import SYSTEM_PROMPT, user_prompt


def main():
    raw = json.dumps({"tags": [
        {"tag": " White bird ", "kind": "object"},
        {"tag": "wing", "kind": "part"},
        {"tag": "snow", "kind": "context"},
        {"tag": "flying", "kind": "action"},
        {"tag": "white bird", "kind": "object"},
        {"tag": "<category>", "kind": "object"},
        {"tag": "a phrase with far too many different words", "kind": "object"},
    ]})
    tags, rejected = normalize_tags("```json\n" + raw + "\n```")
    assert [t["kind"] for t in tags] == ["object", "part", "context", "action"]
    assert [t["segmentable"] for t in tags] == [True, True, True, False]
    assert len(rejected) == 3
    assert len(normalize_tags(raw, max_tags=2)[0]) == 2
    assert normalize_tags('{"tags":[]}') == ([], [])
    assert normalize_tags('```json\n[{"tag":"bird","kind":"object"}]\n```')[0][0]["tag"] == "bird"
    assert normalize_tags('[]') == ([], [])
    for raw_bad in ('{"tags":1}', 'not JSON'):
        try:
            normalize_tags(raw_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid response accepted")
    rows = [{"group": g, "image_id": g + "/" + str(i)} for g in ("A", "B") for i in (2, 0, 1)]
    selected = choose_pilot(rows, ["A", "B"], 2)
    assert [r["image_id"] for r in selected] == ["A/0", "A/1", "B/0", "B/1"]
    assert "frog" not in (SYSTEM_PROMPT + user_prompt(8, 6)).lower()
    print("Semantic tag parsing, roles, action handling and sampling passed.")


if __name__ == "__main__":
    main()
