"""Quick regression test for the Layer-0 (P0) consistency fixes.

No SAM3 / Qwen / GPU / dataset needed.  Covers:
  1) prior preprocessing  : the shared `prepare_prior_at` binarizes AFTER resizing
     (the old inference code interpolated without binarizing -> 0~1 grey edges).
  2) single prior loader  : `dataloader_sam3.load_test_prior` == training pipeline
     for the same npz, and a missing npz is reported (`found=False`), never silent.
  3) neg_flag rule        : `is_noncam` is the one COD10K 'NonCAM' rule (+ real
     dataset counts when the dataset is present).
  4) checkpoint guard     : a legacy 7ch / fusion-less file is rejected, the
     current 8ch + prior_fusion network state is accepted.

Usage:
  python tests/test_prior_consistency.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = ROOT

import dataloader_sam3 as DL
from check_ckpt_channels import assert_ckpt_channels


def _toy_masks(h=100, w=100):
    """Binary target + 3 aux masks with hard edges (like real SAM3 output)."""
    target = np.zeros((1, h, w), np.float32)
    target[0, 20:60, 30:70] = 1.0
    aux = np.zeros((3, h, w), np.float32)
    aux[0, 5:25, 5:25] = 1.0
    aux[1, 70:95, 40:95] = 1.0
    aux[2, 0:100, 90:100] = 1.0
    return target, aux


def test_binarization():
    target, aux = _toy_masks()
    hw = (352, 352)
    # old behaviour: plain interpolation -> intermediate (non-binary) values
    old = torch.nn.functional.interpolate(torch.from_numpy(target).unsqueeze(0),
                                          size=hw, mode="bilinear",
                                          align_corners=False).squeeze(0)
    n_grey = int(((old > 0) & (old < 1)).sum())
    assert n_grey > 0, "sanity: bilinear upsampling must create grey boundaries"
    # new behaviour: strictly binary, same shape as the training tensors
    t2, a2 = DL.prepare_prior_at(torch.from_numpy(target), torch.from_numpy(aux), hw)
    assert tuple(t2.shape) == (1,) + hw, t2.shape
    assert tuple(a2.shape) == (3,) + hw, a2.shape
    for name, m in (("target", t2), ("aux", a2)):
        uniq = set(torch.unique(m).tolist())
        assert uniq <= {0.0, 1.0}, f"{name} must be binary, got {sorted(uniq)[:8]}"
    print(f"    PASS: prepare_prior_at is binary; old path had {n_grey} grey px")


def test_single_loader():
    target, aux = _toy_masks()
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "test_toy_jpg.npz")
        np.savez(npz, target_mask=target, aux_masks=aux, bbox=np.zeros(4, np.float32))

        t_new, a_new, found = DL.load_test_prior("test", "toy.jpg", td, (352, 352))
        assert found, "an existing npz must be reported as found"

        # reference = training pipeline: raw npz -> shared prepare_prior_at
        raw = np.load(npz, allow_pickle=True)
        t_tr, a_tr = DL.prepare_prior_at(torch.from_numpy(raw["target_mask"]).float(),
                                        torch.from_numpy(raw["aux_masks"]).float(),
                                        (352, 352))
        raw.close()   # Windows: release the file handle before the temp dir cleanup
        assert torch.equal(t_new, t_tr), "inference/training target prior disagree"
        assert torch.equal(a_new, a_tr), "inference/training aux prior disagree"

        # a missing npz must be LOUD: zeros + found=False
        t_z, a_z, found_z = DL.load_test_prior("test", "nope.jpg", td, (352, 352))
        assert not found_z, "missing npz must return found=False"
        assert float(t_z.abs().sum()) == 0.0 and tuple(t_z.shape) == (1, 352, 352)
        assert float(a_z.abs().sum()) == 0.0 and tuple(a_z.shape) == (3, 352, 352)

        # regression: the same file reached through a full path (MyTrain_SAM3
        # used an os.sep root, so dataloader.test_dataset handed back the whole
        # path as `name` and the val prior silently became all zeros)
        t_p, a_p, found_p = DL.load_test_prior("test", os.path.join(td, "toy.png"),
                                               td, (352, 352))
        assert found_p, "a full-path name must still resolve to its prior"
        assert torch.equal(t_p, t_tr) and torch.equal(a_p, a_tr)

        # regression: a '.png' image must resolve to its own ('.png') npz, not to
        # a rewritten '.jpg' one
        np.savez(os.path.join(td, "test_toy2_png.npz"),
                 target_mask=target, aux_masks=aux, bbox=np.zeros(4, np.float32))
        assert DL.load_test_prior("test", "toy2.png", td, (352, 352))[2], \
            "a '.png' image must match its '.png' npz"
    print("    PASS: one loader, train==test, full path / .png / missing handled")


def test_noncam_rule():
    assert DL.is_noncam("COD10K-NonCAM-1-Amphibian-10.jpg")
    assert DL.is_noncam("COD10K-NonCAM-1-Amphibian-10.png")
    assert DL.is_noncam(os.path.join("some", "dir", "COD10K-NonCAM-1-Foo-1.jpg"))
    assert not DL.is_noncam("camourflage_00001.jpg")
    assert not DL.is_noncam("COD10K-CAM-1-Aquatic-1-BatFish-2.jpg")
    print("    PASS: is_noncam rule (NonCAM -> neg_flag 1)")
    # real counts (informational; expected 2960/7000 and 1974/4000)
    for tag, d in (("train", os.path.join(HERE, "Dataset", "TrainValDataset", "Imgs")),
                   ("COD10K-test", os.path.join(HERE, "Dataset", "TestDataset",
                                                "COD10K", "Imgs"))):
        if os.path.isdir(d):
            names = os.listdir(d)
            n = sum(1 for f in names if DL.is_noncam(f))
            print(f"    INFO: {tag}: neg_flag=1 on {n}/{len(names)} images")


def test_ckpt_guard():
    with tempfile.TemporaryDirectory() as td:
        # 1) legacy 7ch file without prior_fusion must be rejected
        p7 = os.path.join(td, "fake_7ch.pth")
        torch.save({"backbone.patch_embed1.proj.weight": torch.zeros(64, 7, 7, 7)}, p7)
        try:
            assert_ckpt_channels(p7, expected=8, verbose=False)
        except ValueError as e:
            print(f"    PASS: 7ch rejected -> {str(e)[:58]}...")
        else:
            raise AssertionError("a 7ch checkpoint must be rejected")

        # 2) the real 8ch + prior_fusion network state must be accepted
        from Network_SAM3 import Network
        state = Network(channel=64, imagenet_pretrained=False).state_dict()
        p8 = os.path.join(td, "fake_8ch.pth")
        torch.save(state, p8)
        assert_ckpt_channels(p8, expected=8, verbose=False)
        print("    PASS: 8ch + prior_fusion accepted")

        # 3) missing file -> FileNotFoundError (never a silent zero-init load)
        try:
            assert_ckpt_channels(os.path.join(td, "nope.pth"), verbose=False)
        except FileNotFoundError:
            print("    PASS: missing checkpoint rejected")
        else:
            raise AssertionError("a missing checkpoint must be rejected")

    legacy = os.path.join(HERE, "model_pth", "SAM3", "KKK_best_41.pth")
    if os.path.isfile(legacy):
        try:
            assert_ckpt_channels(legacy, expected=8, verbose=False)
        except ValueError:
            print("    INFO: model_pth/SAM3/KKK_best_41.pth is NOT 8ch (pre-neg_flag "
                  "file) -> retrain instead of evaluating it")
        else:
            print("    INFO: model_pth/SAM3/KKK_best_41.pth is a valid 8ch checkpoint")


if __name__ == "__main__":
    print("=== Layer-0 prior / neg_flag / checkpoint consistency (P0 regression) ===")
    print("[1] prior binarization")
    test_binarization()
    print("[2] single prior loader (train == test)")
    test_single_loader()
    print("[3] neg_flag rule")
    test_noncam_rule()
    print("[4] checkpoint channel guard")
    test_ckpt_guard()
    print("\nALL P0 CHECKS PASSED")
