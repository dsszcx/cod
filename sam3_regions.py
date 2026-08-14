"""sam3_regions.py - Server-side: SAM3 text-prompt region generation.

For each image, read LLM tags from prior_annotations.json, then for each tag
run SAM3 text-prompt segmentation to produce candidate region masks.

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


def segment_by_tag(processor, image, tag, max_objs=1):
    """Run SAM3 text-prompt segmentation for one tag.
    Returns (mask, score) of the top-scoring object, or (None, 0)."""
    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt=tag, state=state)

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


def process_split(data_root, annotations, split, processor, out_dir, resume=True):
    items = annotations.get(split, {})
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
            tags = ann.get("tags", [])
            tag_masks, tag_scores, valid_tags = [], [], []
            for tag in tags:
                mask, score = segment_by_tag(processor, image, tag)
                if mask is not None:
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
                    bbox=np.array(ann.get("bbox", []) or [], dtype=np.float32),
                )
                continue

            masks = np.stack(tag_masks)
            scores = np.array(tag_scores)
            np.savez(
                npz_path,
                tags=np.array(valid_tags, dtype=object),
                masks=masks,
                scores=scores,
                bbox=np.array(ann.get("bbox", []) or [], dtype=np.float32),
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
    parser.add_argument("--no_resume", action="store_true")
    opt = parser.parse_args()

    import torch
    # SAM3 runs under bfloat16 autocast (matching official example)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    annotations = json.load(open(opt.ann, encoding="utf-8"))
    processor = load_sam3(opt.ckpt)
    process_split(opt.data_root, annotations, opt.split, processor, opt.out,
                  resume=not opt.no_resume)