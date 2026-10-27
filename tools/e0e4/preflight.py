"""tools/e0e4/preflight.py - Gate A/B environment audit; BLOCKED means do not run.

Every check returns ``{name, status, detail}`` with status OK / WARN / BLOCKED.
The exit code is 2 as soon as ONE check is BLOCKED, but the reports are still
written -- a blocked server should not look like a crashed script.

Written under ``--out``::

    RUN_LOGS/PREFLIGHT_REPORT.md   verdict + check table
    RUN_LOGS/ENVIRONMENT.md        python / torch / cuda / gpu / disk
    CHECKPOINT_SHA256.txt          E0 + SAM3 checkpoint hashes
    CODE_COMMIT.txt               git sha (dirty flag) + config sha
    PREFLIGHT.json                machine-readable copy

The checks deliberately mirror the §0 rules of docs/project/现行要求.md:

* the E0 checkpoint must *load as 3ch* (a 7/8ch file is not a drop-in baseline);
* Qwen2.5-VL-3B weights + transformers/qwen-vl-utils must be present;
* SAM3 must be importable AND the exact prompt API must exist;
* test data must pair Imgs/GT by stem, with counts reported (not silently short);
* every ``guards.*`` switch must be false (no GT, no filename context, no
  NonCAM skipping, no min-max, no stage-local tuning).

Usage::

    python tools/e0e4/preflight.py --config configs/e0e4_v1.yaml
"""
import argparse
import importlib.util
import os
import platform
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

OK, WARN, BLOCKED = "OK", "WARN", "BLOCKED"
DISK_WARN_GB = 5.0
VRAM_WARN_GB = 8.0


def check(name, status, detail):
    return {"name": name, "status": status, "detail": str(detail)}


def _module_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_guards(cfg):
    try:
        C.guard_config(cfg)
    except C.StageError as exc:
        return check("config guards", BLOCKED, exc)
    return check("config guards", OK, "all guards.* switches are false")


def check_config_file(cfg):
    required = ["version", "seed", "paths.e0_ckpt", "paths.qwen_model",
                "paths.sam3_ckpt", "manifest.p0_per_group", "manifest.p1",
                "e0.threshold", "e0.use_ca", "roi.max_roi", "e4.match_iou",
                "eval.bootstrap_n"]
    missing = [k for k in required if C.cfg_get(cfg, k) is None]
    if missing:
        return check("config keys", BLOCKED, "missing: %s" % ", ".join(missing))
    return check("config keys", OK, "%s (sha256 %s)"
                 % (os.path.basename(cfg.get("_path", "?")),
                    (cfg.get("_sha256") or "")[:16]))


def check_e0_checkpoint(path, use_ca=False):
    import torch

    if not os.path.isfile(path):
        return check("E0 checkpoint", BLOCKED, "not found: %s" % path)
    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            state = getattr(state, "state_dict", lambda: {})()
        from check_ckpt_channels import ckpt_first_conv, describe
        from Network_PVTv2 import Network

        key, shape = ckpt_first_conv(state)
        n_ch = shape[1] if shape else None
        if n_ch != 3:
            return check("E0 checkpoint", BLOCKED,
                         "%s is %sch (%s); the 3ch baseline needs 3ch"
                         % (path, n_ch, describe(shape)))
        ref = Network(channel=64, imagenet_pretrained=False,
                      use_ca=bool(use_ca)).state_dict()
        missing = [k for k in ref if k not in state]
        unexpected = [k for k in state if k not in ref]
        bad = [k for k in state if k in ref
               and tuple(state[k].shape) != tuple(ref[k].shape)]
        if missing or unexpected or bad:
            return check("E0 checkpoint", BLOCKED,
                         "%s: missing=%d unexpected=%d size_mismatch=%d "
                         "(first missing: %s) with e0.use_ca=%s"
                         % (os.path.basename(path), len(missing), len(unexpected),
                            len(bad), (missing or bad or ["-"])[0], bool(use_ca)))
        return check("E0 checkpoint", OK,
                     "%s | 3ch | %d/%d keys strict (use_ca=%s) | sha256 %s"
                     % (os.path.basename(path), len(state), len(ref),
                        bool(use_ca), (C.sha256_file(path) or "")[:16]))
    except Exception as exc:
        return check("E0 checkpoint", BLOCKED, "cannot read %s: %r" % (path, exc))


def check_qwen(model_dir):
    """Local Qwen2.5-VL-3B weights + the exact imports `llm_annotator` needs."""
    missing = [m for m in ("transformers", "accelerate", "qwen_vl_utils")
               if importlib.util.find_spec(m) is None]
    if missing:
        return check("Qwen runtime", BLOCKED,
                     "missing python package(s): %s -> pip install -r "
                     "requirements-server.txt (section [C])" % ", ".join(missing))
    if not os.path.isdir(model_dir):
        return check("Qwen weights", BLOCKED, "model dir not found: %s" % model_dir)
    files = set(os.listdir(model_dir))
    has_config = "config.json" in files
    has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
    if not (has_config and has_weights):
        return check("Qwen weights", BLOCKED,
                     "%s lacks config.json or *.safetensors/*.bin (found %d file(s))"
                     % (model_dir, len(files)))
    from transformers import AutoProcessor  # import check only

    del AutoProcessor
    return check("Qwen weights", OK, "%s (config.json + weights present)" % model_dir)


def check_sam3(ckpt):
    if importlib.util.find_spec("sam3") is None:
        return check("SAM3 package", BLOCKED,
                     "`import sam3` fails: install from sam3-main/ "
                     "(pip install -e sam3-main/sam3-main)")
    try:
        from sam3 import build_sam3_image_model  # noqa: F401
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        return check("SAM3 package", BLOCKED, "import failed: %r" % (exc,))
    needed = ("set_image", "add_geometric_prompt", "set_text_prompt")
    absent = [m for m in needed if not hasattr(Sam3Processor, m)]
    if absent:
        return check("SAM3 package", BLOCKED,
                     "Sam3Processor lacks %s" % ", ".join(absent))
    if not os.path.isfile(ckpt):
        return check("SAM3 checkpoint", BLOCKED, "not found: %s" % ckpt)
    size_gb = os.path.getsize(ckpt) / float(1 << 30)
    return check("SAM3 checkpoint", OK, "%s (%.2f GB, sha256 %s)"
                 % (os.path.basename(ckpt), size_gb,
                    (C.sha256_file(ckpt) or "")[:16]))


def check_env():
    detail = ("python %s | %s | torch %s | torchvision %s | numpy %s | cv2 %s | "
              "yaml %s" % (platform.python_version(), platform.platform(),
                           _module_version("torch") or "-",
                           _module_version("torchvision") or "-",
                           _module_version("numpy") or "-",
                           _module_version("cv2") or "-",
                           _module_version("yaml") or "-"))
    if sys.version_info < (3, 9):
        return check("python env", BLOCKED, detail + " (needs >= 3.9)")
    return check("python env", OK, detail)


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        return check("GPU", BLOCKED,
                     "torch.cuda.is_available() is False: Qwen2.5-VL-3B + SAM3 "
                     "cannot run on CPU in this round")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / float(1 << 30)
    free = torch.cuda.mem_get_info()[0] / float(1 << 30)
    detail = "%s | %.1f GB total | %.1f GB free | cuda %s" % (
        name, total, free, torch.version.cuda)
    if total < VRAM_WARN_GB:
        return check("GPU", WARN, detail + " | load Qwen/SAM3 sequentially")
    return check("GPU", OK, detail)


def check_disk(path):
    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    free_gb = shutil.disk_usage(probe).free / float(1 << 30)
    detail = "%s | %.1f GB free" % (probe, free_gb)
    if free_gb < DISK_WARN_GB:
        return check("disk", WARN, detail + " (composites + masks + sheets)")
    return check("disk", OK, detail)


def check_data(data_root, datasets):
    try:
        rows, problems = C.manifest_rows(data_root, datasets)
    except C.StageError as exc:
        return check("test data", BLOCKED, exc)
    if problems:
        return check("test data", BLOCKED, "; ".join(problems))
    if not rows:
        return check("test data", BLOCKED, "no image/GT pair under %s" % data_root)
    per_dataset = {d: sum(1 for r in rows if r["dataset"] == d) for d in datasets}
    per_group = {g: sum(1 for r in rows if r["group"] == g) for g in C.GROUPS}
    return check("test data", OK, "%d pairs | datasets %s | groups %s"
                 % (len(rows), per_dataset, per_group))


def check_metric_backend():
    missing = [m for m in ("py_sod_metrics", "cv2", "numpy")
               if importlib.util.find_spec(m) is None]
    metric_file = os.path.join(C.ROOT, "新metric.py")
    if missing:
        return check("metric backend", BLOCKED, "missing: %s" % ", ".join(missing))
    if not os.path.isfile(metric_file):
        return check("metric backend", BLOCKED, "missing %s" % metric_file)
    return check("metric backend", OK, "新metric.py + py_sod_metrics present")


def check_out_dir(out_dir):
    root = os.path.abspath(out_dir)
    if not os.path.isdir(root) or not os.listdir(root):
        return check("output dir", OK, "%s is empty/new" % root)
    return check("output dir", WARN,
                 "%s already has %d entr(ies); run_e0e4.py refuses without "
                 "--overwrite" % (root, len(os.listdir(root))))


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def _verdict(checks):
    return BLOCKED if any(c["status"] == BLOCKED for c in checks) else "READY"


def write_reports(cfg, out_dir, checks, data_root):
    root = os.path.abspath(out_dir)
    logs = os.path.join(root, "RUN_LOGS")
    C.ensure_dir(logs)
    verdict = _verdict(checks)
    blocked = [c for c in checks if c["status"] == BLOCKED]

    lines = ["# Preflight report (Gate A/B)", "",
             "- verdict: **%s**" % verdict,
             "- generated: %s" % C.utc_now(),
             "- config: `%s` (sha256 `%s`, version `%s`)"
             % (cfg.get("_path"), cfg.get("_sha256"), cfg.get("_version")),
             "- data root: `%s`" % data_root,
             "", "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        lines.append("| %s | %s | %s |" % (c["name"], c["status"],
                                           c["detail"].replace("|", "\\|")))
    if blocked:
        lines += ["", "## BLOCKED - do not run Gate C", ""]
        lines += ["- **%s**: %s" % (c["name"], c["detail"]) for c in blocked]
        lines += ["", "Fix every row above; WARN does not stop the run."]
    else:
        lines += ["", "All hard requirements satisfied.  Gate C may start with "
                  "`python tools/e0e4/run_e0e4.py --stage all --ids ...`."]
    C.write_json(os.path.join(root, "PREFLIGHT.json"),
                 {"verdict": verdict, "utc": C.utc_now(), "checks": checks,
                  "config": {"path": cfg.get("_path"),
                             "sha256": cfg.get("_sha256"),
                             "version": cfg.get("_version")},
                  "data_root": data_root})
    with open(os.path.join(logs, "PREFLIGHT_REPORT.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    torch_mod = sys.modules.get("torch")
    env = ["# Environment", "",
           "- utc: %s" % C.utc_now(),
           "- python: %s" % sys.version.split()[0],
           "- platform: %s" % platform.platform(),
           "- cwd: `%s`" % os.getcwd(),
           "- repo root: `%s`" % C.ROOT,
           "- torch: %s | torchvision: %s | cuda: %s"
           % (_module_version("torch"), _module_version("torchvision"),
              getattr(getattr(torch_mod, "version", None), "cuda", None)),
           "- numpy: %s | opencv: %s | PyYAML: %s | py_sod_metrics: %s"
           % (_module_version("numpy"), _module_version("cv2"),
              _module_version("yaml"), _module_version("py_sod_metrics")),
           "- transformers: %s | accelerate: %s | qwen_vl_utils: %s"
           % (_module_version("transformers"), _module_version("accelerate"),
              _module_version("qwen_vl_utils")),
           "- sam3: %s" % ("importable" if importlib.util.find_spec("sam3")
                           else "NOT importable"),
           "- config sha256: `%s`" % cfg.get("_sha256"),
           ""]
    with open(os.path.join(logs, "ENVIRONMENT.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(env) + "\n")

    ckpt_lines = []
    for label, path in (("E0", C.resolve_path(C.cfg_get(cfg, "paths.e0_ckpt"))),
                        ("SAM3", C.resolve_path(C.cfg_get(cfg, "paths.sam3_ckpt")))):
        ckpt_lines.append("%s  %s  (%s)"
                          % (C.sha256_file(path) or "MISSING", path, label))
    ckpt_lines.append("%s  %s  (CONFIG)" % (cfg.get("_sha256"), cfg.get("_path")))
    with open(os.path.join(root, "CHECKPOINT_SHA256.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(ckpt_lines) + "\n")

    sha, dirty = None, None
    try:
        from utils import git_state

        sha, dirty = git_state(C.ROOT)
    except Exception:
        pass
    with open(os.path.join(root, "CODE_COMMIT.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("git_sha: %s\ngit_dirty: %s\nconfig: %s\nconfig_sha256: %s\nutc: %s\n"
                 % (sha, dirty, cfg.get("_path"), cfg.get("_sha256"), C.utc_now()))
    return verdict


def run_preflight(cfg, out_dir, data_root=None):
    """Run every check, write the provenance files, return (verdict, checks)."""
    root = os.path.abspath(out_dir)
    C.ensure_dir(root)
    data_root = C.resolve_path(
        data_root or C.cfg_get(cfg, "paths.data_root", "Dataset/TestDataset"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    checks = [
        check_guards(cfg),
        check_config_file(cfg),
        check_env(),
        check_gpu(),
        check_disk(root),
        check_e0_checkpoint(C.resolve_path(C.cfg_get(cfg, "paths.e0_ckpt")),
                            use_ca=bool(C.cfg_get(cfg, "e0.use_ca", False))),
        check_qwen(C.resolve_path(C.cfg_get(cfg, "paths.qwen_model"))),
        check_sam3(C.resolve_path(C.cfg_get(cfg, "paths.sam3_ckpt"))),
        check_data(data_root, datasets),
        check_metric_backend(),
        check_out_dir(root),
    ]
    verdict = write_reports(cfg, root, checks, data_root)
    for c in checks:
        print("[%-7s] %-16s %s" % (c["status"], c["name"], c["detail"]))
    print(">>> VERDICT: %s -> %s/RUN_LOGS/PREFLIGHT_REPORT.md" % (verdict, root))
    return verdict, checks


def main(argv=None):
    ap = argparse.ArgumentParser(description="E0-E4 preflight (Gate A/B)")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data_root", default=None)
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        verdict, _checks = run_preflight(cfg, out_dir, opt.data_root)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 2 if verdict == BLOCKED else 0


if __name__ == "__main__":
    sys.exit(main())
