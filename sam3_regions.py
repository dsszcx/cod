"""sam3_regions.py - Server-side: SAM3 region generation (box-prompt / text-prompt).

For each image, read LLM annotations (tags + bbox) from prior_annotations.json,
then run SAM3 segmentation to produce candidate region masks.

prompt_mode:
  - box:    use bbox from LLM annotation as box-prompt (recommended)
  - text:   use tags as text-prompt (original behavior)
  - hybrid: box-prompt first, then text-prompt for additional regions

================================= 文件影响声明 =================================
本脚本对服务器文件系统的影响：
  1. 【读取】仅读取 --data_root 图片 + --ann 标注文件
  2. 【写入】仅写入 --out 指定输出目录（默认 ./sam3_regions/）
  3. 【绝不】删除/修改/覆盖服务器任何其他已有文件
==============================================================================

Usage (on server with SAM3 env):
  python sam3_regions.py --data_root ./Dataset/TrainValDataset/Imgs/ \
      --ann outputs/prior_annotations.json --split train \
      --ckpt ./checkpoints/sam3.pt --out ./sam3_regions/
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_sam3(checkpoint_path):
    """Load SAM3 image model with local checkpoint. bpe defaults to package assets."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f">>> Loading SAM3 from checkpoint: {checkpoint_path}")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


def _extract_top_mask(state, max_objs=1):
    """Extract the top-scoring mask(s) from the state produced by set_text_prompt.

    Returns (merged_mask, score) where merged_mask is a numpy (H,W) float32
    array (union of top-max_objs object masks), or (None, 0.0) if nothing found.
    """
    scores = state.get("scores", [])
    masks = state.get("masks", [])
    if len(scores) == 0:
        return None, 0.0
    n = min(len(scores), max_objs)
    idx = sorted(range(len(scores)), key=lambda i: scores[i].item(), reverse=True)[:n]
    H, W = masks[0].shape[-2:]
    merged = np.zeros((H, W), dtype=np.float32)
    top_score = 0.0
    for i in idx:
        m = masks[i].squeeze(0).cpu().numpy()
        merged = np.maximum(merged, m)
        top_score = max(top_score, float(scores[i].item()))
    return merged, top_score


def _bbox_to_normalized_xywh(bbox, img_w, img_h):
    """Convert [x1, y1, x2, y2] absolute bbox to normalized [cx, cy, w, h]."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))
    return [cx, cy, w, h]


def _bbox_is_valid(bbox):
    """Check if bbox [x1,y1,x2,y2] is valid (non-zero area)."""
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return (x2 - x1) > 1 and (y2 - y1) > 1


def process_split(data_root, annotations, split, processor, out_dir, resume=True,
                  filelist=None, prompt_mode="box"):
    """Process all images for one split.

    prompt_mode:
      - 'box':    use bbox as box-prompt (recommended, fast & accurate)
      - 'text':   use tags as text-prompt (original behavior)
      - 'hybrid': box-prompt first, then text-prompt for additional regions

    Tier-1 optimization: set_image is called once per image (not once per tag),
    so the image encoder runs only once even when an image has many tags.
    """
    items = annotations.get(split, {})
    if filelist:
        allowed = set()
        with open(filelist, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    allowed.add(line.split()[0])
        before = len(items)
        items = {k: v for k, v in items.items() if k in allowed}
        print(f">>> filelist {filelist}: {len(items)}/{before} images kept")
    print(f">>> Processing {split}: {len(items)} images")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for idx, (name, ann) in enumerate(items.items(), 1):
        npz_path = out_dir / f"{split}_{name.replace('.', '_')}.npz"
        if resume and npz_path.exists():
            n_ok += 1
            continue

        img_path = os.path.join(data_root, name)
        if not os.path.exists(img_path):
            print(f"[{idx}/{len(items)}] SKIP (no image): {name}")
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
            tags = ann.get("tags", [])
            bbox_raw = ann.get("bbox", []) or []

            # ---- Tier-1: encode image ONCE ----
            state = processor.set_image(image)
            processor.reset_all_prompts(state)

            tag_masks, tag_scores, valid_tags = [], [], []

            # ---- Box-prompt mode ----
            if prompt_mode in ("box", "hybrid") and _bbox_is_valid(bbox_raw):
                box_norm = _bbox_to_normalized_xywh(bbox_raw, img_w, img_h)
                state = processor.add_geometric_prompt(
                    box=box_norm, label=True, state=state)
                mask, score = _extract_top_mask(state, max_objs=1)
                if mask is not None:
                    primary_tag = tags[0] if tags else "object"
                    tag_masks.append(mask)
                    tag_scores.append(score)
                    valid_tags.append(primary_tag)
                    print(f"  [box-prompt] bbox={bbox_raw}, tag={primary_tag}, score={score:.3f}")

            # ---- Text-prompt mode (original or hybrid fallback) ----
            if prompt_mode == "text" or \
               (prompt_mode == "hybrid" and len(tag_masks) == 0) or \
               (prompt_mode == "box" and len(tag_masks) == 0):
                # Reset state for text-prompt (re-encode if needed)
                if prompt_mode in ("text",) or \
                   (prompt_mode == "box" and len(tag_masks) == 0):
                    state = processor.set_image(image)
                    processor.reset_all_prompts(state)
                for tag in tags:
                    state = processor.set_text_prompt(prompt=tag, state=state)
                    mask, score = _extract_top_mask(state, max_objs=1)
                    if mask is not None:
                        tag_masks.append(mask)
                        tag_scores.append(score)
                        valid_tags.append(tag)

            # In hybrid mode, also add text-prompt results on top of box-prompt
            if prompt_mode == "hybrid" and len(tag_masks) > 0 and tags:
                for tag in tags:
                    # Skip if tag is already the primary box tag
                    if tag in valid_tags:
                        continue
                    state = processor.set_text_prompt(prompt=tag, state=state)
                    mask, score = _extract_top_mask(state, max_objs=1)
                    if mask is not None and score > 0.5:
                        tag_masks.append(mask)
                        tag_scores.append(score)
                        valid_tags.append(tag)

            if len(tag_masks) == 0:
                print(f"[{idx}/{len(items)}] {name} -> NO regions")
                np.savez(
                    npz_path,
                    tags=np.array([], dtype=object),
                    masks=np.zeros((0, 1, 1), dtype=np.float32),
                    scores=np.array([], dtype=np.float32),
                    bbox=np.array(bbox_raw, dtype=np.float32),
                )
                continue

            masks = np.stack(tag_masks)
            scores = np.array(tag_scores)
            np.savez(
                npz_path,
                tags=np.array(valid_tags, dtype=object),
                masks=masks,
                scores=scores,
                bbox=np.array(bbox_raw, dtype=np.float32),
            )
            n_ok += 1
            print(f"[{idx}/{len(items)}] {name} -> {len(valid_tags)} regions, scores={scores.round(2)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{idx}/{len(items)}] {name} FAILED: {e}")

    print(f">>> Done {split}: {n_ok}/{len(items)} images processed -> {out_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="image dir")
    parser.add_argument("--ann", type=str, default="outputs/prior_annotations.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--ckpt", type=str, default="checkpoints/sam3.pt")
    parser.add_argument("--out", type=str, default="sam3_regions")
    parser.add_argument("--filelist", type=str, default=None,
                        help="txt file listing images to process (e.g. Info/CAM_train.txt)")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--prompt_mode", type=str, default="box",
                        choices=["box", "text", "hybrid"],
                        help="box: bbox box-prompt (recommended); "
                             "text: tag text-prompt (original); "
                             "hybrid: box first, text as supplement")
    opt = parser.parse_args()

    import torch
    # SAM3 runs under bfloat16 autocast (matching official example)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    annotations = json.load(open(opt.ann, encoding="utf-8"))
    processor = load_sam3(opt.ckpt)
    process_split(opt.data_root, annotations, opt.split, processor, opt.out,
                  resume=not opt.no_resume, filelist=opt.filelist,
                  prompt_mode=opt.prompt_mode)
