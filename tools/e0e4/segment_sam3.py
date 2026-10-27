"""tools/e0e4/segment_sam3.py - E1 (box) / E2/E3 (text) prompts -> SAM3 masks.

WHY A NEW SCRIPT INSTEAD OF `sam3_regions.py --prompt_mode hybrid`
------------------------------------------------------------------
`sam3_regions.py` is a *training-prior* generator for the whole dataset: it
resumes by file, reads a bbox from `prior_annotations.json`, and its `hybrid`
mode reuses one processor `state` for a whole frame, skipping text prompts when
a box already produced a mask.  E2/E3 need the exact opposite:

* **per-prompt isolation** -- every prompt calls `set_image` again, so a text
  prompt can never inherit the previous prompt's `geometric_prompt`, its boxes
  or its mask list;
* **no skipping** -- an E1 box that found nothing is recorded as EMPTY instead of
  quietly falling through to text;
* audit counters ``BOX_EXECUTED`` / ``TEXT_EXECUTED`` that the report quotes.

SAM3 always sees the **original RGB image**.  The Qwen composite (E3) is never
passed here -- that would change the pixels SAM3 segments.

Usage::

    python tools/e0e4/segment_sam3.py --config configs/e0e4_v1.yaml \
        --ids outputs/E0_E4_ZEROTRAIN_V1/manifest_p0.txt --branch e1
"""
import argparse
import contextlib
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tools.e0e4 import common as C  # noqa: E402

BRANCHES = ("e1", "e2", "e3")
BOX_EXECUTED = "BOX_EXECUTED"
TEXT_EXECUTED = "TEXT_EXECUTED"


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without sam3)
# ---------------------------------------------------------------------------
def rank_candidates(scores, max_objs=1):
    """Indices of the top-``max_objs`` scores, ties broken by lower index."""
    vals = [float(s) for s in scores]
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    return order[:max(1, int(max_objs))]


def pick_top_mask(masks, scores, max_objs=1):
    """Best single mask: the top-scoring candidate (no union of several objects).

    Returns ``(mask_bool_or_None, score, n_candidates)``.
    """
    if len(scores) == 0 or len(masks) == 0:
        return None, 0.0, 0
    idx = rank_candidates(scores, max_objs=1)[0]
    m = masks[idx]
    if hasattr(m, "detach"):          # torch tensor (possibly on cuda)
        m = m.detach().float().cpu().numpy()
    arr = np.asarray(m).squeeze()
    if arr.ndim == 3:            # (C,H,W) -> first channel
        arr = arr[0]
    return arr > (0.5 if arr.dtype.kind == "f" else 0), float(scores[idx]), len(scores)


def union_masks(masks):
    """OR of boolean masks (``None`` when the list is empty)."""
    out = None
    for m in masks:
        if m is None:
            continue
        out = np.asarray(m, bool) if out is None else (out | np.asarray(m, bool))
    return out


def resize_mask_nearest(mask, height, width):
    import cv2

    m = np.asarray(mask).astype(np.uint8)
    if m.shape[:2] == (int(height), int(width)):
        return m.astype(bool)
    return cv2.resize(m, (int(width), int(height)),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def write_mask_png(path, mask):
    import cv2

    C.ensure_dir(os.path.dirname(os.path.abspath(path)))
    arr = (np.asarray(mask).astype(np.uint8) * 255)
    if not cv2.imwrite(path, arr):
        raise C.StageError("cv2.imwrite failed: %s" % path)
    return path


# ---------------------------------------------------------------------------
# prompt construction (never from GT)
# ---------------------------------------------------------------------------
def prompts_from_rois(roi_rec):
    """E1: one box prompt per ROI (normalized cx,cy,w,h)."""
    if not roi_rec:
        raise C.StageError("E1 needs the ROI stage first")
    return [{"prompt_id": "roi%d" % r["roi_id"], "kind": "box",
             "box": list(r["normalized_xywh"]), "source": "M0_binary"}
            for r in roi_rec.get("rois", [])]


def prompts_from_captions(cap_rec, max_chars=80):
    """E2/E3: one text prompt per caption object (E3 ignores region_id here).

    Returns ``(prompts, skipped)``; ``skipped`` records objects whose text was
    too long/unusable so the audit can show them instead of dropping silently.
    """
    prompts, skipped = [], []
    for i, obj in enumerate((cap_rec or {}).get("objects", [])):
        text = str(obj.get("sam_text") or "").strip()
        rec = {"prompt_id": "obj%d" % (i + 1), "kind": "text", "text": text,
               "object_name": obj.get("name"), "region_id": obj.get("region_id"),
               "source": "qwen"}
        if not text:
            skipped.append(dict(rec, reason="empty_text"))
        elif len(text) > int(max_chars):
            skipped.append(dict(rec, reason="text_too_long"))
        else:
            prompts.append(rec)
    return prompts, skipped


# ---------------------------------------------------------------------------
# SAM3 execution
# ---------------------------------------------------------------------------
def load_processor(sam3_ckpt, confidence_threshold=0.5):
    """Mirror of ``sam3_regions.load_sam3`` with a configurable threshold."""
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        raise C.StageError(
            "cannot import sam3 (%r). Install it from sam3-main/ (pip install -e "
            "sam3-main/sam3-main) and provide paths.sam3_ckpt" % (exc,))
    if not os.path.isfile(sam3_ckpt):
        raise C.StageError("SAM3 checkpoint not found: %s" % sam3_ckpt)
    print(">>> Loading SAM3 from checkpoint: %s" % sam3_ckpt)
    model = build_sam3_image_model(checkpoint_path=sam3_ckpt, load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=float(confidence_threshold))


def execute_prompts(processor, image, prompts, torch, confidence_threshold=0.5):
    """Run each prompt with a FRESH state (``set_image`` per prompt).

    Returns a list of per-prompt dicts: ``{..., mask, score, n_candidates,
    status}``.  ``status`` is OK / EMPTY; the mask is ``None`` unless OK.
    """
    use_amp = bool(torch is not None and torch.cuda.is_available())
    ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
    results = []
    with ctx:                      # explicit: no autograd graph is ever built
        for prompt in prompts:
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Image encoding also runs BF16 kernels inside SAM3. Keep
                    # it in the same autocast scope as the prompt forward.
                    state = processor.set_image(image)
                    state = _forward_prompt(processor, prompt, state)
            else:
                state = processor.set_image(image)
                state = _forward_prompt(processor, prompt, state)
            mask, score, n_cand = pick_top_mask(state.get("masks", []),
                                                state.get("scores", []))
            ok = mask is not None and score >= float(confidence_threshold)
            results.append(dict(prompt, mask=mask if ok else None,
                                score=round(score, 6), n_candidates=int(n_cand),
                                status="OK" if ok else "EMPTY"))
    return results


def _forward_prompt(processor, prompt, state):
    if prompt["kind"] == "box":
        return processor.add_geometric_prompt(box=list(prompt["box"]), label=True,
                                              state=state)
    return processor.set_text_prompt(prompt=prompt["text"], state=state)


def run_stage(cfg, out_dir, ids, branch, limit=None, overwrite=False,
              processor=None):
    """E1/E2/E3 over ``ids``; one audit record per prompt in SAM3_AUDIT.json."""
    if branch not in BRANCHES:
        raise C.StageError("branch must be one of %s, got %r" % (BRANCHES, branch))
    C.guard_config(cfg)
    scfg = cfg.get("sam3") or {}
    ckpt = C.resolve_path(C.cfg_get(cfg, "paths.sam3_ckpt"))
    conf = float(scfg.get("confidence_threshold", 0.5))
    max_chars = int(scfg.get("max_prompt_chars", 80))

    data_root = C.resolve_path(C.cfg_get(cfg, "paths.data_root"))
    datasets = C.cfg_get(cfg, "paths.datasets", ["CAMO", "COD10K"])
    rows, problems = C.manifest_rows(data_root, datasets)
    if problems:
        raise C.StageError("dataset problems: %s" % "; ".join(problems))
    selected = C.rows_for_ids(rows, ids)
    if limit:
        selected = selected[:int(limit)]

    root = os.path.abspath(out_dir)
    stage_root = os.path.join(root, "sam3", branch)
    mask_root = os.path.join(stage_root, "masks")
    pred_root = os.path.join(root, "predictions", branch.upper())
    C.guard_stage_dir(stage_root, overwrite=overwrite)
    C.guard_stage_dir(pred_root, overwrite=overwrite)
    C.ensure_dir(mask_root)
    C.ensure_dir(pred_root)

    torch = None
    try:                              # needed for autocast/inference_mode even
        import torch as _torch        # when the caller already built the processor
        torch = _torch
    except Exception:
        torch = None
    if processor is None:
        if torch is None:
            raise C.StageError("torch is required to build the SAM3 processor")
        processor = load_processor(ckpt, confidence_threshold=conf)

    from PIL import Image

    status_rows = []
    audit = {"branch": branch, BOX_EXECUTED: 0, TEXT_EXECUTED: 0,
             "prompts": 0, "prompts_ok": 0, "images": 0,
             "status_counts": {}, "skipped_prompts": 0}
    t_start = time.time()
    for idx, row in enumerate(selected, 1):
        dataset, stem = row["dataset"], row["stem"]
        image_path, _gt = C.rel_id_to_paths(row, cfg)   # original RGB only
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size

        caption_status = None
        if branch == "e1":
            roi_json = os.path.join(root, "rois", dataset, stem + ".json")
            roi_rec = C.read_json(roi_json)
            if roi_rec is None:
                raise C.StageError("E1 needs the ROI stage first (missing %s)" % roi_json)
            prompts, skipped = prompts_from_rois(roi_rec), []
        else:
            cap_rec = C.read_json(os.path.join(root, "captions", branch,
                                               dataset, stem + ".json"))
            if cap_rec is None:
                raise C.StageError("%s needs the caption stage first" % branch.upper())
            caption_status = cap_rec.get("status")
            prompts, skipped = prompts_from_captions(cap_rec, max_chars=max_chars)

        results = execute_prompts(processor, image, prompts, torch,
                                  confidence_threshold=conf) if prompts else []
        mask_rel, union = {}, None
        for res in results:
            if res["mask"] is None:
                continue
            res["mask"] = resize_mask_nearest(res["mask"], height, width)
            path = os.path.join(mask_root, dataset,
                                "%s_%s.png" % (stem, res["prompt_id"]))
            write_mask_png(path, res["mask"])
            mask_rel[res["prompt_id"]] = os.path.relpath(path, C.ROOT).replace(os.sep, "/")
            union = res["mask"].copy() if union is None else (union | res["mask"])

        if union is not None:
            union_path = os.path.join(mask_root, dataset, stem + "_union.png")
            write_mask_png(union_path, union)
            union_rel = os.path.relpath(union_path, C.ROOT).replace(os.sep, "/")
            union_fraction = round(float(union.mean()), 6)
        else:
            union_rel, union_fraction = "", 0.0

        # eval_strict needs one PNG per (stage, image) so every row has the same
        # denominator; a stage that found nothing writes a black mask, it does
        # not write nothing at all.
        pred_mask = union if union is not None else np.zeros((height, width), bool)
        pred_path = os.path.join(pred_root, dataset, stem + ".png")
        C.ensure_dir(os.path.dirname(pred_path))
        write_mask_png(pred_path, pred_mask)

        if not prompts:
            status = "NO_ROI" if branch == "e1" else "NO_TEXT"
        else:
            status = "OK" if any(r["status"] == "OK" for r in results) else "EMPTY"
        nprompts_ok = sum(1 for r in results if r["status"] == "OK")
        audit["prompts"] += len(prompts)
        audit["prompts_ok"] += nprompts_ok
        audit["skipped_prompts"] += len(skipped)
        audit[BOX_EXECUTED if branch == "e1" else TEXT_EXECUTED] += len(prompts)
        audit["status_counts"][status] = audit["status_counts"].get(status, 0) + 1
        audit["images"] += 1

        rec = {"image_id": row["image_id"], "dataset": dataset, "group": row["group"],
               "branch": branch, "status": status,
               "caption_status": caption_status,
               "source_image": os.path.relpath(image_path, C.ROOT).replace(os.sep, "/"),
               "source_size": [height, width], "mask_source": "original RGB",
               "n_prompts": len(prompts), "n_prompts_ok": nprompts_ok,
               "prompts": [{k: v for k, v in r.items() if k != "mask"}
                           for r in results],
               "skipped": skipped, "mask_files": mask_rel,
               "union_mask": union_rel, "union_fraction": union_fraction,
               "config_version": cfg.get("_version")}
        C.write_json(os.path.join(stage_root, dataset, stem + ".json"), rec)
        status_rows.append({"image_id": row["image_id"], "group": row["group"],
                            "status": status, "n_prompts": len(prompts),
                            "n_prompts_ok": nprompts_ok,
                            "union_fraction": union_fraction})
        C.progress(idx, len(selected), "%s -> %s (%d/%d prompt)"
                   % (row["image_id"], status, nprompts_ok, len(prompts)))

    C.write_csv(os.path.join(stage_root, "SAM3_%s_STATUS.csv" % branch), status_rows)
    audit.update({
        "utc": C.utc_now(), "ckpt": ckpt, "ckpt_sha256": C.sha256_file(ckpt),
        "confidence_threshold": conf,
        "mean_prompts_per_image": round(audit["prompts"] / max(1, audit["images"]), 3),
        "elapsed_s": round(time.time() - t_start, 1),
        "config": {"version": cfg.get("_version"), "sha256": cfg.get("_sha256")},
    })
    C.write_json(os.path.join(root, "sam3", "SAM3_%s_AUDIT.json" % branch), audit)
    print(">>> SAM3 %s: %s" % (branch, audit["status_counts"]))
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser(description="E1/E2/E3: SAM3 segmentation")
    ap.add_argument("--config", default=os.path.join(C.ROOT, "configs", "e0e4_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--branch", required=True, choices=list(BRANCHES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    opt = ap.parse_args(argv)
    try:
        cfg = C.load_config(opt.config)
        out_dir = C.resolve_path(opt.out or C.cfg_get(cfg, "paths.out_dir"))
        run_stage(cfg, out_dir, C.read_id_file(opt.ids), opt.branch,
                  limit=opt.limit, overwrite=opt.overwrite)
    except C.StageError as exc:
        print(">>> BLOCKED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
