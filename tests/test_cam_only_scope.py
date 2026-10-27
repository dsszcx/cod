"""Category filtering is independent of GT/predictions and preserves split rows."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.prepare_cam_only import category, filter_rows, assert_splits


def main():
    train = [{"image_id":"TrainVal/COD10K-CAM-1-Frog-1", "image_sha256":"a", "split":"train", "gt_empty":True},
             {"image_id":"TrainVal/COD10K-NonCAM-4-Deer-2", "image_sha256":"b", "split":"train", "gt_empty":False},
             {"image_id":"TrainVal/camourflage_00001", "image_sha256":"c", "split":"train"}]
    kept, dropped = filter_rows(train)
    assert kept == [train[0],train[2]] and len(dropped) == 1
    assert train[1].get("reason") is None  # inputs unchanged
    changed = [dict(r,gt_empty=not r.get("gt_empty",False),prediction_error=999) for r in train]
    assert [r["image_id"] for r in filter_rows(changed)[0]] == [r["image_id"] for r in kept]
    assert category({"image_id":"COD10K/cod10k-noncam-1", "image_path":"x"}) == "NonCAM"
    assert category({"image_id":"CAMO/a"}) == "CAMO"
    assert category({"image_id":"CAMO/a", "image_path":r"C:\data\COD10K-NonCAM-x.jpg"}) == "NonCAM"
    val = [{"image_id":"TrainVal/COD10K-CAM-2", "image_sha256":"d"}]
    test = [{"image_id":"CAMO/camourflage_test", "image_sha256":"e"}]
    assert_splits(kept,val,test)
    for bad in ([dict(test[0],image_sha256="a")], [dict(test[0],image_id="COD10K/COD10K-NonCAM-3")]):
        try:
            assert_splits(kept,val,bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid scope/leakage accepted")
    try:
        category({"image_id":"TrainVal/unclassified"})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown category silently accepted")
    print("CAM_ONLY_SCOPE_TESTS_PASSED")


if __name__ == "__main__":
    main()
