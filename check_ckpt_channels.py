"""check_ckpt_channels.py - Layer-0 guard: do the checkpoints match the current networks?

Why this exists
---------------
The repo holds checkpoints from several network generations:

  * ``Network_PVTv2.Network``  baseline, 3-channel RGB (``pvt_v2_b2`` default in_chans)
  * ``Network_SAM3.Network``   SAM3 v3: 7-channel input (RGB + target + aux x3)
  * ``Network_SAM3.Network``   SAM3 v4: 8-channel input (RGB + target + aux x3 + neg_flag)

Loading a v3 file into the current (v4) network fails with a cryptic

    RuntimeError: size mismatch for backbone.patch_embed1.proj.weight:
    copying a param with shape torch.Size([64, 7, 7, 7]) ... current model is [64, 8, 7, 7]

(observed with ``model_pth/SAM3/KKK_best_41.pth``).  Run this BEFORE any
inference / evaluation: "which network produced which number" must be answered by
measurement, not by a folder name.

Usage
-----
  python check_ckpt_channels.py                          # model_pth/**/*.pth
  python check_ckpt_channels.py a.pth b.pth              # explicit files
  python check_ckpt_channels.py --glob "model_pth/SAM3/*.pth"
  python check_ckpt_channels.py --strict                 # exit 1 if any ckpt != 8ch
  python check_ckpt_channels.py --expected 7 --strict    # require 7ch instead
  python check_ckpt_channels.py --no-probe-networks      # skip the model comparison
"""
import argparse
import glob
import os
import sys

import torch

# where the input-channel count lives in a PVTv2 state_dict
SIZE_KEYS = ("backbone.patch_embed1.proj.weight", "patch_embed1.proj.weight")


def ckpt_first_conv(state):
    for k in SIZE_KEYS:
        if k in state:
            return k, tuple(state[k].shape)
    return None, None


def describe(shape):
    if shape is None:
        return "unknown (no patch_embed1.proj.weight)"
    c = shape[1]
    if c == 8:
        return "SAM3 v4 (8ch: RGB+target+aux x3+neg_flag) -> loadable by Network_SAM3"
    if c == 7:
        return ("SAM3 v3 (7ch: RGB+target+aux x3, pre-neg_flag) -> NOT loadable by the "
                "current Network_SAM3 (in_chans=8)")
    if c == 3:
        return "baseline (3ch RGB) -> loadable by Network_PVTv2"
    return "unexpected in_chans=%d" % c


def build_probes():
    """Return {name: state_dict} for the two network generations in this repo."""
    probes = {}
    try:
        from Network_PVTv2 import Network as BaseNet
        probes["Network_PVTv2 (baseline)"] = BaseNet(channel=64, imagenet_pretrained=False).state_dict()
    except Exception as exc:  # pragma: no cover
        print(">>> [ckpt-check] cannot build Network_PVTv2: %r" % (exc,))
    try:
        from Network_SAM3 import Network as S3Net
        probes["Network_SAM3 (8ch prior)"] = S3Net(channel=64, imagenet_pretrained=False).state_dict()
    except Exception as exc:  # pragma: no cover
        print(">>> [ckpt-check] cannot build Network_SAM3: %r" % (exc,))
    return probes


def compare(state, ref):
    """Return (n_size_mismatch, n_missing, n_unexpected, first_mismatch_key)."""
    bad = [k for k in state if k in ref and tuple(state[k].shape) != tuple(ref[k].shape)]
    missing = [k for k in ref if k not in state]
    unexpected = [k for k in state if k not in ref]
    return len(bad), len(missing), len(unexpected), (bad[0] if bad else None)


# ---------------------------------------------------------------------------
# importable helpers (used by MyTest_SAM3 / MyTrain_SAM3 / web_server_sam3)
# ---------------------------------------------------------------------------
def unwrap_state(obj):
    """state_dict from a bare save or from a Layer-1 ``{'state_dict': ...}`` one."""
    if isinstance(obj, dict):
        inner = obj.get("state_dict")
        if isinstance(inner, dict):
            return inner
    return obj


def inspect_ckpt(path):
    """Read a checkpoint and describe its first conv / prior_fusion layout."""
    state = unwrap_state(torch.load(path, map_location="cpu"))
    if not isinstance(state, dict):
        state = getattr(state, "state_dict", lambda: {})()
    key, shape = ckpt_first_conv(state)
    return {
        "path": path,
        "state": state,
        "key": key,
        "shape": shape,
        "in_chans": shape[1] if shape else None,
        "n_fusion_keys": sum(1 for k in state if k.startswith("prior_fusion")),
        "verdict": describe(shape),
    }


def assert_state_dict_channels(state, expected=8, require_fusion=True,
                               source="<state_dict>", verbose=True):
    """Same guard as ``assert_ckpt_channels`` for an already-loaded state_dict.

    Used when the caller had to ``torch.load`` anyway (resume needs the stored
    epoch): reading the file twice is slow and, for a 100 MB checkpoint on a
    network share, slow enough that it invites skipping the check.
    """
    key, shape = ckpt_first_conv(state)
    n_fusion = sum(1 for k in state if k.startswith("prior_fusion"))
    if verbose:
        print(">>> [ckpt-check] %s" % source)
        print("    first conv   : %s %s" % (key or "<missing>", shape))
        print("    verdict      : %s" % describe(shape))
        print("    prior_fusion : %d keys" % n_fusion)
    if (shape[1] if shape else None) != expected:
        raise ValueError("%s: in_chans=%s but this network needs %d -- %s"
                         % (source, (shape[1] if shape else None), expected,
                            describe(shape)))
    if require_fusion and n_fusion == 0:
        raise ValueError("%s: no prior_fusion.* keys -> not a Network_SAM3 "
                         "(8ch + fusion) checkpoint" % source)
    return state


def assert_ckpt_channels(path, expected=8, require_fusion=True, verbose=True):
    """Guard: raise ValueError when this file cannot be loaded properly.

    The historical failure mode is silent: a 7ch (pre-neg_flag) file loads with
    ``strict=False`` and quietly randomises ``backbone.patch_embed1``.  Any code
    that loads a checkpoint should call this FIRST.  Returns the state_dict.

    Works for both a bare state_dict and the Layer-1 ``{'state_dict': ...}``
    dict (``inspect_ckpt`` unwraps it), so old and new files are both accepted.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("checkpoint not found: %s" % path)
    info = inspect_ckpt(path)
    return assert_state_dict_channels(info["state"], expected=expected,
                                      require_fusion=require_fusion,
                                      source=info["path"], verbose=verbose)


def load_with_report(model, path, expected=8, require_fusion=True, verbose=True,
                     strict=False):
    """assert_ckpt_channels + load_state_dict(strict=False) + a key report.

    Returns the number of matched keys; raises when nothing could be matched, and
    prints missing/unexpected keys so a partial load can never be silent.

    ``strict=True`` additionally refuses *any* missing/unexpected key.  Even with
    the correct in_chans a checkpoint from another width/variant loads through
    ``strict=False`` while leaving part of the network random -- which is how a
    baseline number can be produced by an untrained decoder.  ``module.`` from a
    DataParallel save is stripped before the comparison.
    """
    state = assert_ckpt_channels(path, expected=expected,
                                 require_fusion=require_fusion, verbose=verbose)
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    own = model.state_dict()
    matched = sum(1 for k, v in state.items()
                  if k in own and tuple(own[k].shape) == tuple(v.shape))
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    model.load_state_dict(state, strict=False)
    if verbose:
        print(">>> [ckpt-check] matched %d/%d keys | missing %d | unexpected %d"
              % (matched, len(own), len(missing), len(unexpected)))
        if missing:
            print("    e.g. missing   : %s" % ", ".join(missing[:3]))
        if unexpected:
            print("    e.g. unexpected: %s" % ", ".join(unexpected[:3]))
    if matched == 0:
        raise ValueError("%s: no key matched the model -> wrong checkpoint" % path)
    if strict and (missing or unexpected):
        raise ValueError("%s: %d missing / %d unexpected key(s) -> not the same "
                         "architecture (pass strict=False to override)"
                         % (path, len(missing), len(unexpected)))
    return matched


def main():
    ap = argparse.ArgumentParser(description="checkpoint <-> network channel guard")
    ap.add_argument("paths", nargs="*", help="checkpoint files")
    ap.add_argument("--glob", dest="globs", action="append", default=None,
                    help="glob pattern (repeatable); default model_pth/**/*.pth")
    ap.add_argument("--expected", type=int, default=8,
                    help="in_chans required by --strict (default 8 = Network_SAM3 v4)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when a checkpoint does not have --expected channels")
    ap.add_argument("--no-probe-networks", action="store_true",
                    help="skip the per-network key/shape comparison")
    opt = ap.parse_args()

    files = list(opt.paths)
    patterns = opt.globs if opt.globs else ["model_pth/**/*.pth", "model_pth/*.pth"]
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    seen, ordered = set(), []
    for f in files:
        n = os.path.normpath(f)
        if os.path.isfile(n) and n not in seen:
            seen.add(n)
            ordered.append(n)
    if not ordered:
        print(">>> [ckpt-check] no checkpoint found (looked at: %s)" % ", ".join(patterns))
        return 1

    probes = {} if opt.no_probe_networks else build_probes()

    print(">>> [ckpt-check] %d checkpoint(s); --expected=%d" % (len(ordered), opt.expected))
    counts, bad_files = {}, []
    for path in ordered:
        try:
            state = torch.load(path, map_location="cpu")
            if not isinstance(state, dict):
                state = getattr(state, "state_dict", lambda: {})()
        except Exception as exc:
            print(">>> %s\n    UNREADABLE: %r" % (path, exc))
            bad_files.append(path)
            continue
        key, shape = ckpt_first_conv(state)
        n_ch = shape[1] if shape else None
        counts[n_ch] = counts.get(n_ch, 0) + 1
        n_fusion = sum(1 for k in state if k.startswith("prior_fusion"))

        print(">>> %s" % path)
        print("    first conv   : %s %s" % (key or "<missing>", shape))
        print("    verdict      : %s" % describe(shape))
        print("    prior_fusion : %d keys" % n_fusion)
        for name, ref in probes.items():
            n_bad, n_miss, n_unexp, example = compare(state, ref)
            if n_bad == 0 and n_miss == 0 and n_unexp == 0:
                print("    vs %-27s MATCH (strict-loadable, %d keys)" % (name, len(state)))
            else:
                extra = (" e.g. %s" % example) if example else ""
                print("    vs %-27s MISMATCH size=%d missing=%d unexpected=%d%s"
                      % (name, n_bad, n_miss, n_unexp, extra))
        if n_ch != opt.expected:
            bad_files.append(path)

    hist = ", ".join("%s:%d" % (k, v) for k, v in
                     sorted(counts.items(), key=lambda kv: -1 if kv[0] is None else kv[0]))
    print(">>> [ckpt-check] in_chans histogram: %s" % hist)
    if bad_files:
        print(">>> [ckpt-check] %d file(s) do NOT match in_chans=%d:" % (len(bad_files), opt.expected))
        for f in bad_files:
            print(">>>            %s" % f)
        if opt.strict:
            return 1
    else:
        print(">>> [ckpt-check] all checkpoints match in_chans=%d" % opt.expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
