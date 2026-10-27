"""CPU tests of explicit input, schema and no-category-example prompts."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.semantic_prior.generate_manifest_tags import read_inputs, parse_tags, build_prompt, SYSTEM
from tools.e0e4 import common as C


def main():
    tags, _ = parse_tags('```json\n[{"tag":"entity","kind":"object"},{"tag":"component","kind":"part"}]\n```', 8, 6)
    assert len(tags) == 2
    assert parse_tags('{"tags":[]}', 8, 6) == ([], [])
    try:
        parse_tags('{"tags":[{"tag":"<category>","kind":"object"}]}', 8, 6)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid-only tags accepted as empty success")
    assert "frog" not in (SYSTEM+build_prompt()).lower()
    with tempfile.TemporaryDirectory(dir=str(ROOT/".test_tmp")) as temp:
        path = Path(temp)/"input.csv"
        row = {"image_id":"TrainVal/test", "image_path":"original.jpg", "image_sha256":"hash", "width":20, "height":20}
        C.write_csv(str(path), [row])
        assert read_inputs(path)[0]["group"] == "TrainVal"
        C.write_csv(str(path), [dict(row, gt_path="forbidden")])
        try:
            read_inputs(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Training GT manifest was accepted")
    print("MANIFEST_TAG_TESTS_PASSED")


if __name__ == "__main__":
    main()
