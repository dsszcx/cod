"""region_selector.py - Select target region + auxiliary regions from SAM3 candidates.

Two modes:
  --mode qwen      : Qwen2.5-VL judges which candidate is the camouflaged target
                     (requires server with Qwen env, image_root + model path)
  --mode heuristic : local heuristic (score = sam3_score * sqrt(area) * objectiveness)

Input : sam3_regions/*.npz  (candidates from SAM3)
Output: prior_sam3/*.npz    (target_mask + aux_masks)
"""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image

# ---------------- heuristic mode ----------------
BACKGROUND_HINTS = ["background", "ground", "rock", "sand", "sky", "water", "grass",
                    "soil", "tree bark", "leaf", "texture", "terrain", "snow", "cloud"]


def _objectiveness(tag: str) -> float:
    t = tag.lower().strip()
    for b in BACKGROUND_HINTS:
        if b in t:
            return 0.5
    return 1.0


def heuristic_select(tags, masks, scores, max_aux=3, target_area_range=(0.02, 0.95)):
    H, W = masks.shape[-2:]
    if len(masks) == 0:
        return np.zeros((1, H, W), np.float32), np.zeros((max_aux, H, W), np.float32), "", []
    bin_masks = (masks > 0.5).astype(np.float32)
    area = bin_masks.sum(axis=(1, 2)) / (H * W)
    rs = np.array(scores, np.float32) * np.sqrt(np.clip(area, 0, 1)) * \
         np.array([_objectiveness(t) for t in tags], np.float32)
    cand = [(i, rs[i]) for i in range(len(masks)) if target_area_range[0] <= area[i] <= target_area_range[1]]
    if not cand:
        cand = [(int(np.argmax(area)), float(np.max(rs)))]
    target_idx = max(cand, key=lambda x: x[1])[0]
    return _assemble(tags, bin_masks, target_idx, max_aux)


def _assemble(tags, bin_masks, target_idx, max_aux=3):
    H, W = bin_masks.shape[-2:]
    target_mask = bin_masks[target_idx:target_idx + 1]
    target_tag = tags[target_idx]
    others = [i for i in range(len(tags)) if i != target_idx]
    aux_masks, aux_tags = [], []
    for i in others[:max_aux]:
        inter = (bin_masks[i] * target_mask).sum()
        union = ((bin_masks[i] + target_mask) > 0).sum()
        if inter / (union + 1e-8) < 0.8:
            aux_masks.append(bin_masks[i])
            aux_tags.append(tags[i])
    while len(aux_masks) < max_aux:
        aux_masks.append(np.zeros((H, W), np.float32))
        aux_tags.append("")
    return target_mask, np.stack(aux_masks[:max_aux]), target_tag, aux_tags[:max_aux]


# ---------------- qwen mode ----------------
def qwen_select(image_path, tags, masks, scores, qwen, max_aux=3):
    """Qwen2.5-VL judges which candidate is the camouflaged target."""
    bin_masks = (masks > 0.5).astype(np.float32)
    if len(tags) == 0:
        return np.zeros((1, 1, 1), np.float32), np.zeros((max_aux, 1, 1), np.float32), "", []

    # build prompt: give qwen the tag list, ask which is the target
    tag_list = ", ".join(tags)
    prompt = (
        f"图像中的候选区域对应的语义描述有：{tag_list}。"
        f"请判断哪个是图中的伪装目标主体（与背景相似、需要特别分辨的目标物体），"
        f"只回答候选描述中的某一个，不要解释。"
    )
    try:
        ann = qwen.generate_with_text(image_path, prompt)
        answer = str(ann.get("answer", "")).strip()
    except Exception as e:
        print(f">>> qwen select failed ({e}), fallback to first tag")
        answer = tags[0]

    # find the tag matching the answer (fuzzy contains)
    target_idx = 0
    for i, t in enumerate(tags):
        if t.lower() in answer.lower() or answer.lower() in t.lower():
            target_idx = i
            break
    return _assemble(tags, bin_masks, target_idx, max_aux)


# ---------------- pipeline ----------------
def process_split(regions_dir, out_dir, split, mode, qwen=None, image_root=None,
                  size=352, resume=True):
    regions_dir = Path(regions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(regions_dir.glob(f"{split}_*.npz"))
    print(f">>> Selecting regions ({mode}) for {split}: {len(files)} files")

    n_ok = 0
    for idx, f in enumerate(files, 1):
        out_path = out_dir / f.name
        if resume and out_path.exists():
            n_ok += 1
            continue
        d = np.load(f, allow_pickle=True)
        tags = [str(t) for t in d["tags"]]
        masks = d["masks"]
        scores = d["scores"]
        bbox = d["bbox"]

        if len(masks) == 0:
            H = W = size
            target = np.zeros((1, H, W), np.float32)
            aux = np.zeros((3, H, W), np.float32)
            target_tag = ""
        else:
            if mode == "qwen" and qwen is not None and image_root is not None:
                rest = f.name[len(f"{split}_"):]          # camourflage_00001_jpg.npz
                rest = rest[:-4]                              # drop .npz
                _base, _ext = rest.rsplit("_", 1)            # (camourflage_00001, jpg)
                img_name = f"{_base}.{_ext}"                 # camourflage_00001.jpg
                img_path = os.path.join(image_root, img_name)
                target, aux, target_tag, _ = qwen_select(
                    img_path, tags, masks, scores, qwen)
            else:
                target, aux, target_tag, _ = heuristic_select(tags, masks, scores)

        np.savez(out_path, target_mask=target, aux_masks=aux,
                 target_tag=np.array([target_tag], dtype=object), bbox=bbox)
        n_ok += 1
        if idx % 20 == 0:
            print(f"[{idx}/{len(files)}] done, target_tag={target_tag}")
    print(f">>> Done {mode}: {n_ok}/{len(files)} -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, default="sam3_regions")
    parser.add_argument("--out", type=str, default="prior_sam3")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--mode", type=str, default="heuristic", choices=["qwen", "heuristic"])
    parser.add_argument("--image_root", type=str, default=None, help="image dir (needed for qwen mode)")
    parser.add_argument("--qwen_model", type=str, default="/home/16t/cr/sr/chatsr/Qwen2.5-VL-3B-Instruct/",
                        help="Qwen model path (needed for qwen mode)")
    parser.add_argument("--no_resume", action="store_true")
    opt = parser.parse_args()

    qwen = None
    if opt.mode == "qwen":
        assert opt.image_root is not None, "--image_root required for qwen mode"
        from llm_annotator import QwenVLAnnotator
        qwen = QwenVLAnnotator(model_path=opt.qwen_model)

    process_split(opt.regions, opt.out, opt.split, opt.mode, qwen=qwen,
                  image_root=opt.image_root, resume=not opt.no_resume)