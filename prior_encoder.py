"""prior_encoder.py - Convert Qwen LLM annotations into network input priors.

Inputs:  prior_annotations.json  (from llm_annotator.py)
Outputs: prior/{name}.npz        (pos_map: bbox gaussian heatmap 1xHxW,
                                 tag_bow: bag-of-words embedding vector,
                                 region_map: expanded attention region)

Usage:
  python prior_encoder.py --ann prior_annotations.json --out prior/
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path


def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab(annotations, min_freq=1):
    """Build a tag vocabulary from all annotations (train split)."""
    counter = {}
    for split_name, items in annotations.items():
        for ann in items.values():
            for tag in ann.get("tags", []):
                t = str(tag).strip()
                if t:
                    counter[t] = counter.get(t, 0) + 1
    vocab = sorted([t for t, c in counter.items() if c >= min_freq])
    return {t: i for i, t in enumerate(vocab)}, counter


def bbox_to_heatmap(bbox, size=352, sigma_scale=0.15):
    """bbox [x1,y1,x2,y2] -> 2D gaussian heatmap (1, size, size).
    If bbox invalid/empty, return a uniform low-value map."""
    H = W = size
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    heat = np.zeros((H, W), dtype=np.float32)
    if not bbox or len(bbox) != 4:
        return heat.reshape(1, H, W)
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sigma_x = max(w * sigma_scale, 8.0)
    sigma_y = max(h * sigma_scale, 8.0)
    exp = -(((xx - cx) ** 2) / (2 * sigma_x ** 2) +
            ((yy - cy) ** 2) / (2 * sigma_y ** 2))
    heat = np.exp(exp).astype(np.float32)
    return heat.reshape(1, H, W)


def bbox_to_region(bbox, size=352, expand_ratio=0.25):
    """bbox -> expanded binary region mask (1, size, size).
    Used as the 'auxiliary region' for attention guidance."""
    H = W = size
    region = np.zeros((H, W), dtype=np.float32)
    if not bbox or len(bbox) != 4:
        return region.reshape(1, H, W)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    ex1 = max(int(x1 - w * expand_ratio), 0)
    ey1 = max(int(y1 - h * expand_ratio), 0)
    ex2 = min(int(x2 + w * expand_ratio), W - 1)
    ey2 = min(int(y2 + h * expand_ratio), H - 1)
    region[ey1:ey2 + 1, ex1:ex2 + 1] = 1.0
    return region.reshape(1, H, W)


def tags_to_bow(tags, vocab):
    """tags list -> bag-of-words vector (len(vocab),) as float32."""
    bow = np.zeros(len(vocab), dtype=np.float32)
    for tag in tags:
        t = str(tag).strip()
        if t in vocab:
            bow[vocab[t]] += 1.0
    if bow.sum() > 0:
        bow = bow / (bow.sum() + 1e-8)  # normalize
    return bow


def build_prior_npz(ann_path, out_dir, size=352):
    annotations = load_annotations(ann_path)
    vocab, counter = build_vocab(annotations)
    print(f">>> Vocab size: {len(vocab)}")
    print(f">>> Top tags: {sorted(counter.items(), key=lambda x: -x[1])[:10]}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save vocab for the network's embedding layer
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    n_processed = 0
    for split_name, items in annotations.items():
        for name, ann in items.items():
            bbox = ann.get("bbox", [])
            tags = ann.get("tags", [])
            conf = float(ann.get("confidence", 0.0))

            pos_map = bbox_to_heatmap(bbox, size)
            region = bbox_to_region(bbox, size)
            tag_bow = tags_to_bow(tags, vocab)

            npz_path = out_dir / f"{split_name}_{name.replace('.', '_')}.npz"
            if npz_path.exists():
                continue  # resume: skip already generated
            np.savez(
                npz_path,
                pos_map=pos_map,          # (1, size, size)
                region=region,            # (1, size, size)
                tag_bow=tag_bow,          # (V,)
                confidence=np.float32(conf),
                bbox=np.array(bbox, dtype=np.float32) if bbox else np.zeros(4, np.float32),
            )
            n_processed += 1
    print(f">>> Done! {n_processed} prior files saved to {out_dir}")


def visualize_prior(img_path, npz_path, out_png="prior_viz.png"):
    """Visualize: original image + heatmap overlay (debugging tool)."""
    import cv2
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    data = np.load(npz_path)
    heat = data["pos_map"][0]
    heat = cv2.resize(heat, (w, h))
    heat = (heat / (heat.max() + 1e-8) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(out_png, overlay)
    print(f">>> Visualization saved to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, default="prior_annotations.json")
    parser.add_argument("--out", type=str, default="prior")
    parser.add_argument("--size", type=int, default=352)
    opt = parser.parse_args()
    build_prior_npz(opt.ann, opt.out, opt.size)