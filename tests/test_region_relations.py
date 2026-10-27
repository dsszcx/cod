"""CPU regression tests for geometry, variable-N batches and source packing."""
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.region_relations import region_geometry, prepare_region_sample, transform_region_sample, collate_region_samples, load_region_package
from tools.semantic_prior.pack_regions import pack_one
from tools.e0e4 import common as C


def sample(masks):
    n, h, w = masks.shape
    roles = np.zeros((n, 4), np.float32)
    roles[:, 3] = 1
    return {"masks": masks, "rgb": np.arange(3*h*w, dtype=np.float32).reshape(3, h, w),
            "role_hint": roles, "role_hint_valid": np.zeros(n, bool)}


def must_fail(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Invalid input accepted")


def test_geometry():
    m = np.zeros((4, 20, 20), np.uint8)
    m[0, 5:7, 5:7] = 1
    m[1, 2:12, 2:12] = 1
    m[2, 15:18, 15:18] = 1
    m[3] = m[1]
    g, pairs, valid = region_geometry(m)
    assert valid.all() and pairs.sum() == 12
    assert np.isclose(g[0, 1, 0], .04) and g[0, 1, 1] == 1
    assert np.isclose(g[0, 1, 2], .04)
    assert g[1, 3, 7] == 1 and g[1, 3, 0] == 1
    assert g[1, 2, 6] > 0
    np.testing.assert_allclose(g[:, :, 0], g[:, :, 0].T)
    np.testing.assert_allclose(g[:, :, 1], g[:, :, 2].T)
    for k in (3, 4, 5):
        np.testing.assert_allclose(g[:, :, k], -g[:, :, k].T, atol=1e-7)
    for k in (6, 7):
        np.testing.assert_allclose(g[:, :, k], g[:, :, k].T)
    assert not pairs.diagonal().any() and not g[np.arange(4), np.arange(4)].any()
    order = [2, 0, 3, 1]
    permuted = region_geometry(m[order])[0]
    np.testing.assert_allclose(permuted, g[order][:, order])
    full = np.ones((2, 3, 5), np.uint8)
    assert region_geometry(full)[0][0, 1, 0] == 1
    assert region_geometry(np.zeros((0, 20, 20), np.uint8))[0].shape == (0, 0, 8)
    zero = region_geometry(np.zeros((2, 20, 20), np.uint8))
    assert not zero[0].any() and not zero[1].any() and not zero[2].any()
    assert np.isfinite(zero[0]).all()
    must_fail(lambda: region_geometry(np.full((1, 3, 3), 255)))
    must_fail(lambda: region_geometry(np.full((1, 3, 3), np.nan)))
    return m


def test_transforms_and_batch(m):
    s = prepare_region_sample(sample(m))
    flipped = transform_region_sample(s, hflip=True)
    np.testing.assert_array_equal(flipped["rgb"], s["rgb"][:, :, ::-1])
    np.testing.assert_allclose(flipped["relation_geom"][:, :, 4], -s["relation_geom"][:, :, 4])
    np.testing.assert_allclose(flipped["relation_geom"][:, :, 5], s["relation_geom"][:, :, 5])
    cropped = transform_region_sample(s, crop=(0, 0, 12, 12), output_hw=(24, 24))
    assert not cropped["region_valid"][2]
    assert not cropped["relation_valid"][2].any()
    assert not cropped["relation_geom"][2].any()
    assert np.isfinite(cropped["relation_geom"]).all()
    assert np.isclose(cropped["relation_geom"][0, 1, 1], 1)
    empty = sample(np.zeros((0, 20, 20), np.uint8))
    one = sample(m[:1])
    batch = collate_region_samples([s, empty, one], as_torch=False)
    assert batch["region_masks"].shape == (3, 4, 20, 20)
    assert batch["region_valid"].sum() == 5
    assert not batch["relation_valid"][1:].any()
    assert not batch["role_hint"].any()  # unknown/absent hints do not become context
    assert not batch["region_masks"][1].any()
    assert "sam3_score" not in batch and "image_id" not in batch
    all_empty = collate_region_samples([empty, empty], as_torch=False)
    assert all_empty["region_masks"].shape == (2, 1, 20, 20)
    assert not all_empty["region_valid"].any()
    assert not all_empty["relation_geom"].any()
    tensors = collate_region_samples([s, empty])
    import torch
    assert tensors["region_valid"].dtype == torch.bool
    assert tensors["rgb"].dtype == torch.float32
    resized_empty = transform_region_sample(empty, output_hw=(8, 8))
    assert resized_empty["masks"].shape == (0, 8, 8)
    must_fail(lambda: collate_region_samples([]))


def test_pack():
    (ROOT / ".test_tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="region_relations_", dir=str(ROOT / ".test_tmp")) as temp:
        root = Path(temp)
        source, output = root / "source", root / "packages"
        source.mkdir()
        rgb = root / "NonCAM_name_not_a_model_feature.png"
        Image.fromarray(np.zeros((12, 14, 3), np.uint8)).save(rgb)
        mask = np.zeros((12, 14), np.uint8)
        mask[1:9, 2:10] = 255
        mask_path = source / "candidate.png"
        Image.fromarray(mask).save(mask_path)
        row = {"image_id": "fixture/sample", "image_path": str(rgb), "image_sha256": C.sha256_file(str(rgb))}
        record = {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "source_size": [12, 14],
                  "tags": [{"tag_id": "t01", "tag": "visible entity", "kind": "object", "status": "OK", "segmentable": True,
                            "candidates": [{"candidate_id": "c01", "mask_path": "candidate.png", "sam3_score": .8},
                                           {"candidate_id": "c02", "mask_path": "candidate.png", "sam3_score": .6}]}]}
        path = source / "candidates/fixture/sample.json"
        C.write_json(str(path), record)
        result = pack_one(row, source, output)
        assert result["n_regions"] == 2  # exact duplicates preserved, not deleted
        loaded = load_region_package(output / "fixture/sample/regions.npz")
        np.testing.assert_array_equal(loaded["masks"], np.stack([mask > 0, mask > 0]))
        assert loaded["role_hint"][:, 0].all()
        assert loaded["relation_geom"][0, 1, 7] == 1
        meta = C.read_json(str(output / "fixture/sample/regions.json"))
        assert not meta["gt_used"] and not meta["selection_scores_used"]
        assert meta["nodes"][0]["tag"] == "visible entity"
        record["image_id"] = row["image_id"] = "fixture/empty"
        record["tags"] = [{"tag_id": "t02", "tag": "surface", "kind": "context", "status": "EMPTY", "segmentable": True, "candidates": []}]
        C.write_json(str(source / "candidates/fixture/empty.json"), record)
        assert pack_one(row, source, output)["status"] == "NO_CANDIDATES"
        assert load_region_package(output / "fixture/empty/regions.npz")["masks"].shape == (0, 12, 14)


def main():
    m = test_geometry()
    test_transforms_and_batch(m)
    test_pack()
    print("REGION_RELATIONS_TESTS_PASSED: geometry, empty/padding, transforms, permutation, numeric packages and source preservation")


if __name__ == "__main__":
    main()
