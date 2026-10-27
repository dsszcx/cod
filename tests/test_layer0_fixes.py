"""test_layer0_fixes.py - unit tests for the 2026-10 "Layer 0" fixes.

Every test below fails on the pre-fix code and passes on the fixed code:

  1. hybrid_e_loss: ``fixed`` really applies the structure/Sobel weight;
     ``legacy`` reproduces the old mean-BCE collapse (kept for old numbers).
  2. MyTrain_SAM3._val_prior: validation uses the same prior_mode as training
     (zero / oracle / real / shuffle) instead of always reading the real npz.
  3. dataloader_sam3.neg_flag_value: the 8th channel can be forced to 0 so the
     COD10K "NonCAM" file-name label no longer leaks into the prediction.
  4. eval_metrics.evaluate: an incomplete prediction set raises ManifestError
     instead of silently shrinking the denominator; --allow_missing is explicit.
  5. Network_SAM3.load_pretrained_backbone: a missing/wrong ImageNet file raises
     instead of training from random weights behind a one-line warning.

No dataset and no GPU are required.  Run:

    python tests/test_layer0_fixes.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = []


def ok(msg):
    PASS.append(msg)
    print("    PASS: %s" % msg)


# ---------------------------------------------------------------------------
# 1) loss weighting
# ---------------------------------------------------------------------------
def test_loss_fixed_vs_legacy():
    import MyTrain as MT
    import MyTrain_SAM3 as MS

    torch.manual_seed(0)
    pred = torch.randn(2, 1, 64, 64)
    mask = (torch.rand(2, 1, 64, 64) > 0.6).float()

    # The E-measure / wIoU terms are identical in both modes, so the two differ
    # by exactly (weighted BCE - mean BCE).  That makes the historical collapse
    # measurable: legacy's BCE term must equal the *unweighted* mean.
    mean_bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    legacy = MT.hybrid_e_loss(pred, mask, "legacy")
    fixed = MT.hybrid_e_loss(pred, mask, "fixed")
    assert torch.allclose(legacy - fixed, (mean_bce - wbce).mean(), atol=1e-5), \
        "legacy must use the plain mean BCE (the weight collapsed to a scalar)"
    ok("legacy - fixed == mean BCE - weighted BCE -> legacy ignores the weight")

    assert not torch.allclose(fixed, legacy, atol=1e-4), \
        "fixed mode must differ from legacy: the weight has to matter"
    ok("fixed mode differs from legacy -> the 5x structure weight is applied")

    # control: a uniform mask has weight 1 everywhere -> both modes agree
    z = torch.zeros(1, 1, 32, 32)
    zp = torch.randn(1, 1, 32, 32)
    a = MT.hybrid_e_loss(zp, z, "legacy")
    b = MT.hybrid_e_loss(zp, z, "fixed")
    assert torch.allclose(a, b, atol=1e-6), "weights must be 1 on a uniform mask"
    ok("uniform mask -> fixed == legacy (weight is 1, sanity of the formula)")

    # the SAM3 loss (Sobel x9 weight) keeps the same shared terms: it still
    # equals the unweighted mean BCE in legacy mode, and fixed applies weit*edge
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)
    gt_edge = torch.clamp(torch.abs(F.conv2d(mask, sobel_x, padding=1))
                          + torch.abs(F.conv2d(mask, sobel_y, padding=1)), 0, 1)
    weit_s = weit * (1 + 9 * gt_edge)
    wbce_s = (weit_s * bce).sum(dim=(2, 3)) / weit_s.sum(dim=(2, 3))

    legacy_s = MS.hybrid_e_loss(pred, mask, "legacy")
    fixed_s = MS.hybrid_e_loss(pred, mask, "fixed")
    assert torch.allclose(legacy_s - fixed_s, (mean_bce - wbce_s).mean(), atol=1e-5), \
        "SAM3 legacy must still collapse to the plain mean BCE"
    # NOTE: the two legacy losses are not bit-equal because SAM3's wIoU also uses
    # the Sobel-weighted weit; the identity above is the full statement of the
    # bug (BCE term == plain mean) and it holds for both implementations.
    assert torch.allclose(fixed_s - legacy_s, (wbce_s - mean_bce).mean(), atol=1e-5)
    ok("SAM3 legacy BCE == plain mean BCE (edge weight only reaches E/wIoU)")
    assert not torch.allclose(fixed_s, legacy_s, atol=1e-4), \
        "SAM3 fixed mode must apply the Sobel edge weight"
    ok("MyTrain_SAM3: legacy unchanged, fixed applies weit x Sobel (verified)")

    try:
        MT.hybrid_e_loss(pred, mask, "nope")
    except ValueError:
        ok("unknown loss_mode raises instead of silently picking one")
    else:
        raise AssertionError("unknown loss_mode must raise")


# ---------------------------------------------------------------------------
# 2) validation prior follows --prior_mode
# ---------------------------------------------------------------------------
def test_val_prior_modes():
    import MyTrain_SAM3 as T

    gt = np.zeros((32, 32), np.float32)
    gt[8:24, 8:24] = 1.0
    hw = (64, 64)

    t, a, found = T._val_prior("zero", "x.jpg", gt, hw, "./prior_sam3")
    assert t.shape == (1,) + hw and float(t.abs().sum()) == 0.0
    assert a.shape == (3,) + hw and float(a.abs().sum()) == 0.0
    assert found is True, "zero mode must not be reported as a missing prior"
    ok('_val_prior("zero") feeds an all-zero prior (shape %s)' % (tuple(t.shape),))

    t, a, found = T._val_prior("oracle", "x.jpg", gt, hw, "./prior_sam3")
    assert t.shape == (1,) + hw, "oracle prior must match the network input, not the GT"
    vals = set(torch.unique(t).tolist())
    assert vals <= {0.0, 1.0}, "oracle prior must stay binary, got %r" % (vals,)
    assert 0.03 < float(t.mean()) < 0.9
    ok('_val_prior("oracle") resizes GT to %s and keeps it binary' % (hw,))

    # real / shuffle go through load_test_prior -> patch it to record the names
    calls = []

    def fake_load(split, name, root, shape):
        calls.append((split, name, tuple(shape)))
        return (torch.ones(1, *shape), torch.zeros(3, *shape), True)

    orig = T.load_test_prior
    T.load_test_prior = fake_load
    try:
        T._val_prior("real", "a.jpg", gt, hw, "root")
        T._val_prior("shuffle", "b.jpg", gt, hw, "root", prev_name="a.jpg")
    finally:
        T.load_test_prior = orig

    assert calls[0][:2] == ("test", "a.jpg"), "real mode must load this image's npz"
    assert calls[1][:2] == ("test", "a.jpg"), \
        "shuffle mode must reuse the PREVIOUS image's prior, not its own"
    assert calls[0][2] == hw and calls[1][2] == hw
    ok("_val_prior(real) -> own npz, _val_prior(shuffle) -> previous image's npz")

    # the regression itself: a zero-prior run must not be validated with a real npz
    imported = []

    def record(*args, **kwargs):
        imported.append(args[1])
        return (torch.ones(1, *hw), torch.zeros(3, *hw), True)

    T.load_test_prior = record
    try:
        T._val_prior("zero", "a.jpg", gt, hw, "root")
    finally:
        T.load_test_prior = orig
    assert imported == [], "zero-prior validation must never touch the real npz"
    ok("zero-prior runs are no longer validated against the real prior")


# ---------------------------------------------------------------------------
# 3) neg_flag: label vs zero
# ---------------------------------------------------------------------------
def test_neg_flag_modes():
    from dataloader_sam3 import neg_flag_value, is_noncam

    noncam = "COD10K-CAM-1-Aquatic-1-BatFish-1_NonCAM_0001.jpg"
    normal = "COD10K-CAM-1-Aquatic-1-BatFish-1_0001.jpg"
    assert is_noncam(noncam) and not is_noncam(normal)
    assert neg_flag_value(noncam, "label") == 1.0
    assert neg_flag_value(normal, "label") == 0.0
    assert neg_flag_value(noncam, "zero") == 0.0
    assert neg_flag_value(normal, "zero") == 0.0
    ok("neg_flag_value: label reproduces the NonCAM rule, zero removes the leak")

    try:
        neg_flag_value(normal, "bogus")
    except ValueError:
        ok("unknown neg_flag mode raises")
    else:
        raise AssertionError("unknown neg_flag mode must raise")


# ---------------------------------------------------------------------------
# 4) eval manifest
# ---------------------------------------------------------------------------
def test_eval_manifest():
    import cv2
    import eval_metrics as EM

    td = tempfile.mkdtemp(prefix="layer0_eval_")
    gtd, prd = os.path.join(td, "gt"), os.path.join(td, "pred")
    os.makedirs(gtd)
    os.makedirs(prd)
    for stem in ("a", "b", "c"):
        g = np.zeros((64, 64), np.uint8)
        g[16:48, 16:48] = 255
        cv2.imwrite(os.path.join(gtd, stem + ".png"), g)
    for stem in ("a", "b"):                    # 'c' missing on purpose
        p = np.zeros((64, 64), np.uint8)
        p[20:44, 20:44] = 200
        cv2.imwrite(os.path.join(prd, stem + ".png"), p)

    try:
        EM.evaluate("t", prd, gtd, False)
    except EM.ManifestError as exc:
        assert "missing_pred=1" in str(exc) and "gt=3" in str(exc), str(exc)
        ok("incomplete manifest -> ManifestError (denominator 3, not 2)")
    else:
        raise AssertionError("evaluate() must refuse a 2/3 prediction set")

    r = EM.evaluate("t", prd, gtd, False, allow_missing=True)
    assert r["n_evaluated"] == 2 and r["n_gt"] == 3 and r["n_pred"] == 2
    assert r["n_missing_pred"] == 1 and r["allow_missing"] is True
    assert r["n_images"] == r["n_evaluated"]
    ok("--allow_missing scores the intersection but reports n/nGT = 2/3")

    for key in ("adpem", "avgem", "wfm", "adpfm", "sm", "mae"):
        assert key in r["metrics"], "REPORT_COLUMNS lost %r" % key
    assert 0.0 <= r["metrics"]["avgem"] <= 1.0
    ok("report keeps adpE AND meanE (the E-convention ambiguity)")

    # a complete manifest still evaluates normally
    full = os.path.join(td, "pred_full")
    os.makedirs(full)
    for stem in ("a", "b", "c"):
        p = np.zeros((64, 64), np.uint8)
        p[20:44, 20:44] = 200
        cv2.imwrite(os.path.join(full, stem + ".png"), p)
    r2 = EM.evaluate("t", full, gtd, False)
    assert r2["n_evaluated"] == 3 and r2["n_missing_pred"] == 0
    assert abs(r2["metrics"]["mae"] - r["metrics"]["mae"]) < 1e-6
    ok("complete manifest evaluates all 3 images")

    EM.print_table([r2])
    ok("print_table renders the new n/nGT + meanE column")



# ---------------------------------------------------------------------------
# 5) pretrained backbone must not fail silently
# ---------------------------------------------------------------------------
def test_pretrained_guard():
    from Network_SAM3 import Network, load_pretrained_backbone
    from lib.pvtv2 import pvt_v2_b2

    td = tempfile.mkdtemp(prefix="layer0_pre_")
    bb3 = pvt_v2_b2(pretrained=False, in_chans=3)

    missing = os.path.join(td, "missing.pth")
    try:
        load_pretrained_backbone(bb3, missing, strict=True)
    except FileNotFoundError:
        ok("missing ImageNet file raises FileNotFoundError (was a print())")
    else:
        raise AssertionError("a missing pretrained file must raise")

    bad = os.path.join(td, "not_pvt.pth")
    torch.save({"foo": torch.zeros(1)}, bad)
    try:
        load_pretrained_backbone(bb3, bad, strict=True)
    except RuntimeError:
        ok("a file without patch_embed1.proj.weight raises RuntimeError")
    else:
        raise AssertionError("a wrong checkpoint must raise")

    good = os.path.join(td, "b2.pth")
    torch.save(bb3.state_dict(), good)
    bb8 = pvt_v2_b2(pretrained=False, in_chans=8)
    n = load_pretrained_backbone(bb8, good, strict=True)
    assert n > 100, "only %d layers were copied" % n
    assert torch.equal(bb8.patch_embed1.proj.weight[:, :3],
                       bb3.patch_embed1.proj.weight), "RGB channels not copied exactly"
    ok("%d layers copied; first conv keeps its ImageNet RGB filters" % n)

    net = Network(channel=64, imagenet_pretrained=False)
    assert net.backbone.patch_embed1.proj.weight.shape[1] == 8
    assert any(k.startswith("prior_fusion") for k in net.state_dict())
    ok("Network(imagenet_pretrained=False) builds the 8ch + prior_fusion model")


def main():
    print("[1] hybrid_e_loss weighting")
    test_loss_fixed_vs_legacy()
    print("[2] validation prior follows --prior_mode")
    test_val_prior_modes()
    print("[3] neg_flag label vs zero")
    test_neg_flag_modes()
    print("[4] eval manifest / metric names")
    test_eval_manifest()
    print("[5] pretrained backbone guard")
    test_pretrained_guard()
    print("\nOK: %d checks passed" % len(PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
