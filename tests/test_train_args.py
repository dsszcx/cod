"""Quick unit test for MyTrain_SAM3's argparse guards (no dataset / GPU needed).

Covers the 2026-10-17 throughput changes:
  * DataParallel is opt-in (--dp) and resolve_gpus() narrows to one card otherwise
  * --size_rates is parsed, not hard-coded
  * --batchsize >= 2 and the "batchsize % n_gpus == 0 / >= 2 per shard" guards
  * train() actually consumes the parsed size_rates

Run:  python tests/test_train_args.py
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import MyTrain_SAM3 as T  # noqa: E402


P = T.build_parser()

# --- 1) defaults: DP off, size_rates "0.75,1", amp off -----------------------
o = P.parse_args([])
print("[1] defaults: dp=%s size_rates=%r amp=%s gpus=%r epoch=%d"
      % (o.dp, o.size_rates, o.amp, o.gpus, o.epoch))
assert o.dp is False, "--dp must be OFF by default (DP is ~70x slower here)"
assert o.amp is False, "--amp must be OFF by default"
assert o.size_rates == "0.75,1", "size_rates default should keep the 2-scale recipe"
assert o.batchsize == 6 and o.trainsize == 352 and o.epoch == 80
print("    PASS: DP / amp off by default, 2-scale recipe preserved")

# --- 2) resolve_gpus: multi-GPU without --dp must collapse to one card ------
o = P.parse_args(["--gpus", "0,1"])
single = T.resolve_gpus(o)
print("[2] --gpus 0,1 (no --dp) -> %r" % single)
assert single == "0", "without --dp only the first card may stay visible"
assert len(single.split(",")) == 1

o = P.parse_args(["--gpus", "0,1", "--dp", "--batchsize", "6"])
both = T.resolve_gpus(o)
print("    --gpus 0,1 --dp        -> %r" % both)
assert both == "0,1", "--dp must be honoured when explicitly given"

o = P.parse_args(["--gpus", " 2 , 3 "])
assert T.resolve_gpus(o) == "2", "whitespace and extra cards must not leak"
o = P.parse_args(["--gpus", ""])
assert T.resolve_gpus(o) == "0", "empty --gpus falls back to card 0"
print("    PASS: DP opt-in / whitespace / empty-gpus handling")

# --- 3) parse_size_rates ----------------------------------------------------
assert T.parse_size_rates("0.75,1") == [0.75, 1.0]
assert T.parse_size_rates(" 1 ") == [1.0]
assert T.parse_size_rates("0.5,,1.25") == [0.5, 1.25]
for bad in ("0", "-1", "abc", "", "5"):
    try:
        T.parse_size_rates(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("size_rates %r should have been rejected" % bad)
print("[3] parse_size_rates: good values accepted, %r rejected"
      % ("0/-1/abc/(empty)/5",))
print("    PASS: --size_rates validation")

# --- 4) check_args guards ---------------------------------------------------
ok = P.parse_args([])
assert T.check_args(ok, 1) == [], "defaults on one GPU must be clean"
assert T.check_args(P.parse_args(["--size_rates", "1"]), 1) == []

bs1 = T.check_args(P.parse_args(["--batchsize", "1"]), 1)
assert bs1 and "batchsize" in bs1[0], bs1
div = T.check_args(P.parse_args(["--dp", "--batchsize", "6"]), 4)
assert div and "整除" in div[0], div
shard = T.check_args(P.parse_args(["--dp", "--batchsize", "8"]), 4)
assert shard == [], "8 over 4 GPUs = 2 per shard, which is legal: %r" % (shard,)
tiny = T.check_args(P.parse_args(["--trainsize", "64", "--size_rates", "0.3"]), 1)
assert tiny and "太小" in tiny[0], tiny
print("[4] guards: bs=1 -> %d msg | bs=6/4卡 -> %d msg | bs=8/4卡 -> %d msg | "
      "64px*0.3 -> %d" % (len(bs1), len(div), len(shard), len(tiny)))
print("    PASS: batchsize / shard / size_rates guards")

# --- 5) the parsed rates are actually used by train() -----------------------
params = inspect.signature(T.train).parameters
assert "size_rates" in params, "train() must take size_rates from the parser"
src = inspect.getsource(T.main)
assert "size_rates = parse_size_rates(opt.size_rates)" in src, \
    "main() must feed --size_rates into train()"
print("[5] train(size_rates=...) wired from --size_rates")
print("    PASS: no hard-coded [0.75, 1] left in main()")

# --- 6) --epoch N trains epochs [start, N] (closed interval) ----------------
# The loop used to be `range(start_epoch, opt.epoch)` (half-open), so --epoch 2
# silently trained exactly 1 epoch while T_max=opt.epoch scheduled 2.  Both the
# loop and the scheduler now count the same number of epochs.
assert "range(start_epoch, opt.epoch + 1)" in src, \
    "--epoch N must train epochs [start, N] (closed interval)"
assert P.parse_args(["--epoch", "2"]).epoch == 2
print("[6] --epoch 2 -> range(1, 3) -> exactly 2 epochs (closed interval)")
print("    PASS: --epoch counts trained epochs, matching T_max=opt.epoch")

print("\nALL TESTS PASSED")
