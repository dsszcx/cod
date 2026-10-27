"""CPU synthetic training-loader tests; no model or optimizer required."""
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataloader_relation import RelationDataset, collate_relation_batch, spatial_plan
from tools.semantic_prior.prepare_training import split_rows, pilot_rows
from tools.semantic_prior.pack_regions import pack_one
from tools.e0e4 import common as C


def main():
    rows = [{"image_id": "id"+str(i), "image_sha256": str(i//2)} for i in range(20)]
    train, val = split_rows(rows, .2, 22)
    assert not ({r["image_sha256"] for r in train} & {r["image_sha256"] for r in val})
    assert split_rows(list(reversed(rows)), .2, 22) == (train, val)
    pilot = pilot_rows(train, 3)
    assert len(pilot) == 4  # keep the whole duplicate group
    assert spatial_plan((40, 50), True, 22, 0, "x") == spatial_plan((40, 50), True, 22, 0, "x")
    assert spatial_plan((40, 50), False, 22, 9, "x") == (None, False)
    (ROOT / ".test_tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="relation_training_", dir=str(ROOT / ".test_tmp")) as temp:
        root = Path(temp)
        source, packages = root / "source", root / "packages"
        source.mkdir()
        rows = []
        for index in range(2):
            mask = np.zeros((40, 50), np.uint8)
            mask[8:30, 10:35] = 255
            image_path, gt_path = root / ("rgb%d.png" % index), root / ("gt%d.png" % index)
            Image.fromarray(np.stack([mask, mask, mask], axis=-1)).save(image_path)
            Image.fromarray(mask).save(gt_path)
            Image.fromarray(mask).save(source / ("mask%d.png" % index))
            image_id = "TrainVal/sample%d" % index
            record = {"image_id": image_id, "image_sha256": C.sha256_file(str(image_path)), "source_size": [40, 50],
                      "tags": [{"tag_id": "t01", "tag": "region", "kind": "object", "status": "OK" if index == 0 else "EMPTY", "segmentable": True,
                                "candidates": [{"candidate_id": "c01", "sam3_score": .8, "mask_path": "mask0.png"}] if index == 0 else []}]}
            C.write_json(str(source / "candidates" / (image_id+".json")), record)
            row = {"image_id": image_id, "image_path": str(image_path), "image_sha256": record["image_sha256"],
                   "gt_path": str(gt_path), "region_package": str(packages / image_id / "regions.npz"), "split": "train"}
            pack_one(row, source, packages)
            rows.append(row)
        manifest = root / "train.csv"
        C.write_csv(str(manifest), rows)
        relation = RelationDataset(manifest, "train", "relation", size=32, seed=22)
        direct = RelationDataset(manifest, "train", "direct", size=32, seed=22)
        rgb = RelationDataset(manifest, "train", "rgb", size=32, seed=22)
        item = relation[0]
        np.testing.assert_array_equal(item["gt"].numpy()[0], item["inputs"]["masks"][0])
        np.testing.assert_array_equal(item["inputs"]["rgb"], rgb[0]["inputs"]["rgb"])
        np.testing.assert_array_equal(item["inputs"]["rgb"], direct[0]["inputs"]["rgb"])
        assert item["spatial_plan"] == rgb[0]["spatial_plan"]
        assert relation[1]["inputs"]["masks"].shape == (0, 32, 32)
        batch = collate_relation_batch([item, relation[1]])
        assert batch["gt"].shape == (2, 1, 32, 32)
        assert "gt" not in batch["inputs"] and "image_ids" not in batch["inputs"]
        assert not batch["inputs"]["region_valid"][1].any()
        # Changing GT changes loss targets, but no model input or relationship.
        with Image.open(rows[0]["gt_path"]) as im:
            altered = 255 - np.asarray(im)
        Image.fromarray(altered).save(rows[0]["gt_path"])
        changed = relation[0]
        assert not torch.equal(changed["gt"], item["gt"])
        for key in ("rgb", "masks", "relation_geom", "role_hint"):
            np.testing.assert_array_equal(changed["inputs"][key], item["inputs"][key])
        missing_rows = [dict(rows[0], region_package=str(root / "missing/regions.npz"))]
        C.write_csv(str(root / "missing.csv"), missing_rows)
        try:
            RelationDataset(root / "missing.csv", "train", "relation")[0]
        except ValueError:
            pass
        else:
            raise AssertionError("Missing prior silently replaced")
        val_rows = [dict(r, split="val") for r in rows]
        C.write_csv(str(root / "val.csv"), val_rows)
        validation = RelationDataset(root / "val.csv", "val", size=32)
        assert validation[0]["spatial_plan"] == (None, False)
    print("RELATION_TRAINING_DATA_TESTS_PASSED: split, shared augmentation, GT isolation, empty/missing priors and batch")


if __name__ == "__main__":
    main()
