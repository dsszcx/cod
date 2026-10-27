import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from thop import clever_format, profile




def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr



class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


# ---------------------------------------------------------------------------
# reproducibility / provenance (Layer-1, 2026-10)
# ---------------------------------------------------------------------------
# Three things every entry point needs and each one used to do differently (or
# not at all): seed the four RNGs, seed the DataLoader workers, and leave a
# machine-readable record of the run next to the checkpoint.

def set_seed(seed, deterministic=False, benchmark=None):
    """Seed random / numpy / torch / cuda and return the effective cudnn state.

    ``benchmark=None`` (default) means "on unless --deterministic".  The old code
    hard-coded ``cudnn.benchmark = True`` *and* claimed to be reproducible: the
    algorithm autotuner is free to pick a different kernel for the same shape,
    so two runs with the same seed do not produce the same loss.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if benchmark is None:
        benchmark = not deterministic
    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)
    return {"seed": int(seed),
            "cudnn.benchmark": bool(benchmark),
            "cudnn.deterministic": bool(deterministic)}


def seed_worker(worker_id):
    """DataLoader(worker_init_fn=seed_worker): seed torch, numpy AND random.

    torch seeds itself from ``base_seed + worker_id``; ``numpy`` / ``random`` are
    NOT seeded by DataLoader.  Under the default ``fork`` start method every
    worker inherits the parent's state, so all workers used to draw the same
    crop / rotation / flip sequence.
    """
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def sha256_file(path, chunk=1 << 20):
    """sha256 of a file (checkpoint provenance); None when the file is absent."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_state(cwd=None):
    """(sha, dirty) of the working tree; (None, None) when git is unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
            universal_newlines=True).strip()
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd,
            stderr=subprocess.DEVNULL, universal_newlines=True)
        return sha, bool(out.strip())
    except Exception:
        return None, None


def write_run_manifest(path, opt, extra=None):
    """Write a JSON manifest describing this run (never raises on missing git).

    The repo used to keep ``log.log`` only, opened in append mode: two runs in the
    same ``--save_root`` were indistinguishable afterwards, and nothing recorded
    the git commit / seed / dataset counts the number was produced from.
    """
    sha, dirty = git_state()
    manifest = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "args": {k: v for k, v in sorted(vars(opt).items())},
    }
    if extra:
        manifest.update(extra)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    # Resuming into the same --save_root must not destroy the first run's
    # provenance: run #1 keeps `run_manifest.json`, every later run gets its own
    # timestamped file next to it (previously the manifest was silently
    # overwritten, so "which config produced KKK_best_2.pth?" was unanswerable).
    if os.path.exists(path):
        stem, ext = os.path.splitext(path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cand = "%s_%s%s" % (stem, stamp, ext)
        i = 1
        while os.path.exists(cand):
            i += 1
            cand = "%s_%s_%d%s" % (stem, stamp, i, ext)
        path = cand
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(">>> [manifest] {}".format(path))
    return manifest


BEST_FILE = "best.json"


def write_best(save_root, epoch, mae, extra=None):
    """Persist the best epoch next to the checkpoints.

    Resume used to reset ``best_state`` to inf, so the first epoch after a
    resume always overwrote ``KKK_best_<n>.pth`` -- even when it was worse than
    the checkpoint the run was resumed from, which quietly replaced the model
    the reported number came from.
    """
    payload = {"epoch": int(epoch), "mae": float(mae),
               "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if extra:
        payload.update(extra)
    path = os.path.join(save_root, BEST_FILE)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


def read_best(save_root):
    """{'epoch','mae'} from best.json, or None when it is absent/unreadable."""
    path = os.path.join(save_root, BEST_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return {"epoch": int(d["epoch"]), "mae": float(d["mae"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_latest_checkpoint(save_root, pattern="*.pth"):
    """Checkpoint with the highest epoch in its name (mtime breaks ties).

    ``--resume auto`` used to take the newest *file*, so copying a checkpoint
    (which rewrites mtime) made an old epoch look like the latest one.
    """
    import glob
    import re
    best = None
    for p in glob.glob(os.path.join(save_root, pattern)):
        m = re.search(r"(\d+)", os.path.basename(p))
        key = (int(m.group(1)) if m else -1, os.path.getmtime(p))
        if best is None or key > best[0]:
            best = (key, p)
    return None if best is None else best[1]


def save_checkpoint(path, model, epoch, best_mae=None, opt=None, extra=None):
    """``torch.save`` a dict, not a bare state_dict.

    A bare state_dict carries no epoch, so resume had to parse the *file name*:
    renaming or copying a checkpoint silently restarted training at epoch 1.
    The weight key stays ``state_dict`` so every older loader still works.
    """
    net = model.module if hasattr(model, "module") else model
    payload = {"format": "cod-layer1",
               "epoch": int(epoch),
               "best_mae": None if best_mae is None else float(best_mae),
               "state_dict": net.state_dict(),
               "argv": list(sys.argv),
               "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if opt is not None:
        payload["args"] = {k: v for k, v in sorted(vars(opt).items())}
    if extra:
        payload.update(extra)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(path, map_location="cpu"):
    """-> (state_dict, meta).

    Accepts both the dict written by ``save_checkpoint`` and the bare
    state_dict produced by older runs.  ``meta`` is empty for the latter, and
    the caller must then fall back to the file name *and say so*.
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"], {k: v for k, v in obj.items() if k != "state_dict"}
    if not isinstance(obj, dict):
        raise TypeError("%s: expected a state_dict or a checkpoint dict, got %r"
                        % (path, type(obj)))
    return obj, {}


def write_eval_manifest(path, entries):
    """JSON list consumed by ``eval_metrics.py --manifest``."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(entries), f, indent=2, ensure_ascii=False)
    return path


def run_eval_metrics(manifest, csv=None, group_by="dataset", extra_args=None):
    """Run ``eval_metrics.py`` on `manifest`; returns its exit code.

    Launched through ``sys.executable``: a bare ``python`` twice resolved to an
    interpreter without torch, and the failure looked like a metrics bug.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "eval_metrics.py")
    cmd = [sys.executable, script, "--manifest", manifest]
    if csv:
        cmd += ["--csv", csv]
    if group_by:
        cmd += ["--group_by", group_by]
    if extra_args:
        cmd += [str(a) for a in extra_args]
    print(">>> [eval_metrics] " + " ".join(cmd))
    return subprocess.call(cmd)


def save_map(path, res01):
    """Write a [0,1] prediction map as an 8-bit PNG; abort instead of lying.

    ``cv2.imwrite`` returns False on a bad path and the old code printed
    "saving prediction" regardless, which surfaced later as a silently short
    prediction folder.  Values are clipped and rounded (the old ``res * 255``
    cast truncated every pixel by up to 1 grey level).
    """
    import cv2
    arr = np.clip(np.asarray(res01, dtype=np.float32) * 255.0, 0, 255)
    ok = cv2.imwrite(path, np.rint(arr).astype(np.uint8))
    if not ok:
        raise IOError("cv2.imwrite failed: %s" % path)
    return True
