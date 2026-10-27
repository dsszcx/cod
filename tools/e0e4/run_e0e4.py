"""tools/e0e4/run_e0e4.py - orchestrate Gate C/D/E (preflight -> ... -> review).

Design points that matter for the report
---------------------------------------
* **One directory per split.**  ``--out`` defaults to
  ``<paths.out_dir>/<split>`` (``.../p0`` = Gate C, ``.../p1`` = Gate D/E), so a
  P0 prediction can never be overwritten by a P1 run and every run directory
  carries its own manifest, config copy and logs.
* **Models are loaded once per family and unloaded before the next.**  Qwen3B
  (~6.5 GB) and SAM3 cannot be co-resident on an 8 GB card, so ``qwen`` runs E2
  and E3 with one load, then the model is deleted, ``torch.cuda.empty_cache()``
  runs, and only then is SAM3 loaded.
* **Preflight BLOCKED stops the run** with exit code 2 and a RUN_SUMMARY that
  says which check failed -- it never starts E0 "just to see".
* Every stage is wrapped so its stdout goes both to the console and to
  ``RUN_LOGS/<stage>.log``; ``RUN_LOGS/stages.jsonl`` records elapsed seconds.

Usage (server, repo root)::

    python tools/e0e4/run_e0e4.py --split p0 --stage all --gpu_id 0
    python tools/e0e4/run_e0e4.py --split p1 --stage e4
    python tools/e0e4/run_e0e4.py --split p1 --stage eval --eval-stages E0,E1,E2,E3
"""
import argparse
import gc
import os
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

ALL_STAGES = ("preflight", "manifest", "e0", "roi", "qwen", "sam3", "e4",
              "eval", "review")
SUB_STAGES = ("qwen_e2", "qwen_e3", "sam3_e1", "sam3_e2", "sam3_e3")


class _Tee(object):
    """stdout -> console + one log file (stage logs are part of the evidence)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def _log_stage(root, name, fn):
    logs = os.path.join(root, "RUN_LOGS")
    C.ensure_dir(logs)
    path = os.path.join(logs, "%s.log" % name)
    t0 = time.time()
    print("\n=== [%s] start %s ===" % (name, C.utc_now()))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        old = sys.stdout
        sys.stdout = _Tee(old, fh)
        try:
            out = fn()
        finally:
            sys.stdout = old
    elapsed = round(time.time() - t0, 1)
    print("=== [%s] done in %.1f s ===" % (name, elapsed))
    with open(os.path.join(logs, "stages.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"stage": "%s", "elapsed_s": %s, "utc": "%s"}\n'
                 % (name, elapsed, C.utc_now()))
    return out, elapsed


def _free_vram():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_preflight(cfg, root):
    from tools.e0e4 import preflight

    verdict, checks = preflight.run_preflight(cfg, root)
    return {"verdict": verdict,
            "blocked": [c["name"] for c in checks if c["status"] == "BLOCKED"]}


def stage_manifest(cfg, root):
    from tools.e0e4 import build_manifest

    return build_manifest.run_stage(cfg, root, want_hash=True, overwrite=True)


def stage_e0(cfg, root, ids, limit, overwrite, gpu_id):
    from tools.e0e4 import infer_e0

    rows = infer_e0.run_stage(cfg, root, ids, limit=limit, overwrite=overwrite,
                              gpu_id=gpu_id)
    _free_vram()
    return {"n_images": len(rows)}


def stage_roi(cfg, root, ids, limit, overwrite):
    from tools.e0e4 import roi_from_mask

    rows = roi_from_mask.run_stage(cfg, root, ids, overwrite=overwrite,
                                   limit=limit)
    return {"n_images": len(rows)}


def stage_qwen(cfg, root, ids, branches, limit, overwrite):
    from tools.e0e4 import caption_qwen

    model_path = C.resolve_path(C.cfg_get(cfg, "paths.qwen_model"))
    captioner = caption_qwen.QwenCaptioner(
        model_path, max_tokens=int(C.cfg_get(cfg, "qwen.max_tokens", 192)))
    try:
        out = {}
        for branch in branches:
            recs = caption_qwen.run_stage(cfg, root, ids, branch, limit=limit,
                                          overwrite=overwrite, captioner=captioner)
            out[branch] = len(recs)
        caption_qwen.write_text_table(root)
        return out
    finally:
        del captioner
        _free_vram()


def stage_sam3(cfg, root, ids, branches, limit, overwrite):
    from tools.e0e4 import segment_sam3

    ckpt = C.resolve_path(C.cfg_get(cfg, "paths.sam3_ckpt"))
    conf = float(C.cfg_get(cfg, "sam3.confidence_threshold", 0.5))
    processor = segment_sam3.load_processor(ckpt, confidence_threshold=conf)
    try:
        out = {}
        for branch in branches:
            audit = segment_sam3.run_stage(cfg, root, ids, branch, limit=limit,
                                           overwrite=overwrite,
                                           processor=processor)
            out[branch] = audit.get("status_counts", {})
        return out
    finally:
        del processor
        _free_vram()


def stage_e4(cfg, root, ids, limit, overwrite):
    from tools.e0e4 import fuse_e4

    return fuse_e4.run_stage(cfg, root, ids, limit=limit, overwrite=overwrite)


def stage_eval(cfg, root, ids, eval_stages):
    from tools.e0e4 import eval_strict

    code = eval_strict.run(cfg, root, ids, stages=eval_stages)
    return {"exit_code": code}


def stage_review(cfg, root, ids, limit):
    from tools.e0e4 import make_review

    return make_review.run_stage(cfg, root, ids, limit=limit)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _lock_config(cfg, root):
    logs = os.path.join(root, "RUN_LOGS")
    C.ensure_dir(logs)
    dst = os.path.join(logs, "CONFIG_LOCKED.yaml")
    shutil.copyfile(cfg["_path"], dst)
    if C.sha256_file(dst) != cfg.get("_sha256"):
        raise C.StageError("config changed while locking it: %s" % cfg["_path"])
    return dst


def run(cfg, root, split="p0", stage="all", limit=None, overwrite=False,
        gpu_id="0", eval_stages=None, ids_file=None):
    """Execute the requested stage(s); returns a process exit code."""
    root = os.path.abspath(root)
    C.ensure_dir(root)
    C.guard_config(cfg)
    logs = os.path.join(root, "RUN_LOGS")
    C.ensure_dir(logs)
    _lock_config(cfg, root)

    stages = list(ALL_STAGES if stage == "all" else [stage])
    bad = [s for s in stages if s not in ALL_STAGES + SUB_STAGES]
    if bad:
        raise C.StageError("unknown stage(s): %s (choose from %s)"
                           % (", ".join(bad), ", ".join(ALL_STAGES + SUB_STAGES)))

    ids_path = ids_file or os.path.join(root, "manifest_%s.txt" % split)
    needs_ids = [s for s in stages if s not in ("preflight", "manifest")]
    # `--stage all` creates the id file in the manifest step, so it may not exist
    # yet; a run that starts at E0 must fail *before* touching the GPU.
    ids = (C.read_id_file(ids_path)
           if (needs_ids and os.path.isfile(ids_path)) else [])
    if needs_ids and not ids and "manifest" not in stages:
        raise C.StageError("missing id file %s: run --stage manifest first "
                           "(or pass --ids)" % ids_path)
    if eval_stages is None:
        eval_stages = ["E0", "E1", "E2", "E3", "E4"]

    summary = {"split": split, "out": root, "stages": stages,
               "n_ids": len(ids), "ids_file": ids_path if needs_ids else None,
               "config": {"path": cfg["_path"], "sha256": cfg.get("_sha256"),
                          "version": cfg.get("_version")},
               "started": C.utc_now(), "results": {}, "timings": {}}

    def write_summary(exit_code, error=None):
        summary["exit_code"] = exit_code
        summary["error"] = error
        summary["finished"] = C.utc_now()
        C.write_json(os.path.join(root, "RUN_SUMMARY.json"), summary)

    for name in stages:
        if name == "preflight":
            res, secs = _log_stage(root, name, lambda: stage_preflight(cfg, root))
            summary["results"][name] = res
            summary["timings"][name] = secs
            if res["verdict"] != "READY":
                write_summary(2, "preflight BLOCKED: %s" % ", ".join(res["blocked"]))
                print(">>> STOP: preflight BLOCKED (%s)" % ", ".join(res["blocked"]))
                return 2
            continue

        if name == "manifest":
            res, secs = _log_stage(root, name, lambda: stage_manifest(cfg, root))
            summary["results"][name] = {"counts": res.get("counts"),
                                        "seed": res.get("seed")}
            summary["timings"][name] = secs
            continue

        if not ids:
            ids = C.read_id_file(ids_path)
            summary["n_ids"] = len(ids)

        dispatch = {
            "e0": lambda: stage_e0(cfg, root, ids, limit, overwrite, gpu_id),
            "roi": lambda: stage_roi(cfg, root, ids, limit, overwrite),
            "qwen": lambda: stage_qwen(cfg, root, ids, ["e2", "e3"], limit, overwrite),
            "qwen_e2": lambda: stage_qwen(cfg, root, ids, ["e2"], limit, overwrite),
            "qwen_e3": lambda: stage_qwen(cfg, root, ids, ["e3"], limit, overwrite),
            "sam3": lambda: stage_sam3(cfg, root, ids, ["e1", "e2", "e3"],
                                       limit, overwrite),
            "sam3_e1": lambda: stage_sam3(cfg, root, ids, ["e1"], limit, overwrite),
            "sam3_e2": lambda: stage_sam3(cfg, root, ids, ["e2"], limit, overwrite),
            "sam3_e3": lambda: stage_sam3(cfg, root, ids, ["e3"], limit, overwrite),
            "e4": lambda: stage_e4(cfg, root, ids, limit, overwrite),
            "eval": lambda: stage_eval(cfg, root, ids, eval_stages),
            "review": lambda: stage_review(cfg, root, ids, limit),
        }
        res, secs = _log_stage(root, name, dispatch[name])
        summary["results"][name] = res
        summary["timings"][name] = secs
        if name == "eval" and res.get("exit_code") not in (0, None):
            write_summary(res["exit_code"], "eval returned %s" % res["exit_code"])
            return res["exit_code"]

    write_summary(0)
    print(">>> RUN SUMMARY: %s (%.1f s total)"
          % (os.path.join(root, "RUN_SUMMARY.json"),
             sum(summary["timings"].values())))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="E0-E4 orchestration")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None,
                    help="run dir (default: <paths.out_dir>/<split>)")
    ap.add_argument("--split", default="p0", choices=["p0", "p1"])
    ap.add_argument("--stage", default="preflight",
                    choices=list(ALL_STAGES + SUB_STAGES) + ["all"])
    ap.add_argument("--ids", default=None, help="explicit image-id file")
    ap.add_argument("--eval-stages", default="E0,E1,E2,E3,E4")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu_id", default="0")
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        base = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        root = base if opt.out else os.path.join(base, opt.split)
        eval_stages = [s.strip() for s in opt.eval_stages.split(",") if s.strip()]
        return run(cfg, root, split=opt.split, stage=opt.stage, limit=opt.limit,
                   overwrite=opt.overwrite, gpu_id=opt.gpu_id,
                   eval_stages=eval_stages, ids_file=opt.ids)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
