"""region_selector.py - Select target region + auxiliary regions from SAM3 candidates.

Two modes:
  --mode qwen      : Qwen2.5-VL judges which candidate is the camouflaged target
  --mode heuristic : local heuristic (score = sam3_score * sqrt(area) * objectiveness)

Optimizations (per region_selector优化方案, 2026-08):
  1. Qwen visual prompting: candidate mask boundaries are drawn (numbered) on the
     original image, and Qwen answers with a number instead of a free-text tag.
  2. aux hard-negative mining: candidates ranked by IoU with the target (excluding
     near-duplicates IoU >= 0.8), the top `max_aux` are used as auxiliary regions.
  3. prior quality gate: target area < min_target_ratio of the image OR target SAM3
     score < conf_thresh  ->  zero prior (treated as "no target").
  4. Qwen failure fallback: on error / unparseable answer, fall back to
     heuristic_select instead of blindly picking tags[0].

Input : sam3_regions/*.npz  (candidates from SAM3)
Output: prior_sam3/*.npz    (target_mask + aux_masks)
"""
import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:      # pragma: no cover - server may lack cv2
    cv2 = None
    _CV2_OK = False

# ---------------- heuristic mode ----------------
BACKGROUND_HINTS = ["background", "ground", "rock", "sand", "sky", "water", "grass",
                    "soil", "tree bark", "leaf", "texture", "terrain", "snow", "cloud"]


def _objectiveness(tag: str) -> float:
    t = tag.lower().strip()
    for b in BACKGROUND_HINTS:
        if b in t:
            return 0.5
    return 1.0


def _scores_array(scores, n):
    """Safely convert SAM3 scores to a float array of length n."""
    s = np.asarray(scores, np.float32) if scores is not None and len(scores) else \
        np.zeros(n, np.float32)
    if s.shape[0] != n:
        s = np.pad(s, (0, max(n - s.shape[0], 0)))[:n]
    return s



def heuristic_select(tags, masks, scores, max_aux=3, target_area_range=(0.02, 0.95)):
    """Pick target by heuristic score; returns (target, aux, target_tag, aux_tags, score)."""
    H, W = masks.shape[-2:]
    if len(masks) == 0:
        return (np.zeros((1, H, W), np.float32), np.zeros((max_aux, H, W), np.float32),
                "", [], 0.0)
    bin_masks = (masks > 0.5).astype(np.float32)
    area = bin_masks.sum(axis=(1, 2)) / (H * W)
    scores = _scores_array(scores, len(masks))
    rs = scores * np.sqrt(np.clip(area, 0, 1)) * \
         np.array([_objectiveness(t) for t in tags], np.float32)
    cand = [(i, rs[i]) for i in range(len(masks))
            if target_area_range[0] <= area[i] <= target_area_range[1]]
    if not cand:
        cand = [(int(np.argmax(area)), float(np.max(rs)))]
    target_idx = max(cand, key=lambda x: x[1])[0]
    target, aux, target_tag, aux_tags = _assemble(tags, bin_masks, target_idx, max_aux)
    return target, aux, target_tag, aux_tags, float(scores[target_idx])


def _assemble(tags, bin_masks, target_idx, max_aux=3):
    """Build target + aux masks.

    Aux candidates are ranked by IoU with the target (descending) so the network
    is trained against the "hardest" confusable regions; near-duplicates of the
    target (IoU >= 0.8) are excluded. Zero-padded when fewer than max_aux.
    """
    H, W = bin_masks.shape[-2:]
    target_mask = bin_masks[target_idx:target_idx + 1]
    target_tag = tags[target_idx]
    others = [i for i in range(len(tags)) if i != target_idx]

    ious = []
    for i in others:
        inter = (bin_masks[i] * target_mask).sum()
        union = ((bin_masks[i] + target_mask) > 0).sum()
        ious.append((float(inter / (union + 1e-8)), i))
    ious.sort(key=lambda x: x[0], reverse=True)   # hardest negatives first

    aux_masks, aux_tags = [], []
    for iou, i in ious:
        if iou >= 0.8:                 # near-duplicate of the target -> skip
            continue
        aux_masks.append(bin_masks[i])
        aux_tags.append(tags[i])
        if len(aux_masks) >= max_aux:
            break
    while len(aux_masks) < max_aux:
        aux_masks.append(np.zeros((H, W), np.float32))
        aux_tags.append("")
    return target_mask, np.stack(aux_masks[:max_aux]), target_tag, aux_tags[:max_aux]


# ---------------- visual prompting (Qwen) ----------------
def draw_candidates(image_path, bin_masks, tags, draw_dir, max_cand=5):
    """Draw candidate mask contours (numbered) on the image; save to draw_dir.

    Returns the path of the annotated image. Requires OpenCV.
    """
    assert _CV2_OK, "OpenCV (cv2) is required for visual prompting"
    os.makedirs(draw_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 255, 255),
              (255, 0, 255), (0, 255, 255)]
    n = min(len(bin_masks), max_cand)
    for i in range(n):
        m = np.asarray(bin_masks[i] > 0.5, np.uint8) * 255
        if m.shape[-2:] != (ih, iw):
            m = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = colors[i % len(colors)]
        # Draw thick bounding rectangle (3px) instead of thin contours
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        # Large number label with filled background for visibility
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(img, (x, max(y - th - 10, 0)), (x + tw + 10, y), color, -1)
        cv2.putText(img, label, (x + 5, max(y - 5, th + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(draw_dir, f"{stem}_cand.png")
    cv2.imwrite(out_path, img)
    return out_path


def _match_tag(tags, answer):
    """Fuzzy tag match; returns index or None."""
    a = answer.lower().strip()
    for i, t in enumerate(tags):
        t = t.lower()
        if t in a or a in t:
            return i
    return None



# ---------------- qwen mode ----------------
def qwen_select(image_path, tags, masks, scores, qwen, max_aux=3, draw_dir=None,
                max_cand=5):
    """Qwen2.5-VL judges which candidate is the camouflaged target.

    With visual prompting (draw_dir given and OpenCV available), candidate mask
    boundaries are drawn (numbered) on the image and Qwen answers with a NUMBER.
    Otherwise the original tag-list prompt is used. Any failure / unparseable
    answer falls back to heuristic_select (never blindly tags[0]).

    Returns (target, aux, target_tag, aux_tags, target_score).
    """
    bin_masks = (masks > 0.5).astype(np.float32)
    if len(tags) == 0:
        return (np.zeros((1, 1, 1), np.float32), np.zeros((max_aux, 1, 1), np.float32),
                "", [], 0.0)
    n = min(len(tags), max_cand)

    # ---- n==1: only one candidate, no need to ask Qwen ----
    if n == 1:
        target, aux, target_tag, aux_tags = _assemble(tags, bin_masks, 0, max_aux)
        return target, aux, target_tag, aux_tags, (float(scores[0]) if len(scores) else 0.0)

    # ---- visual prompting: draw numbered candidates, Qwen answers a number ----
    annotated = None
    if _CV2_OK and draw_dir is not None:
        try:
            annotated = draw_candidates(image_path, bin_masks[:n], tags[:n],
                                        draw_dir, max_cand=max_cand)
        except Exception as e:
            print(f">>> draw_candidates failed ({e}), fall back to raw-image prompt")

    # System prompt for visual selection (NOT "Output only JSON"!)
    _VISUAL_SYS = ("You are a visual analysis assistant. "
                   "Answer with a single number only. No JSON, no explanation.")

    if annotated is not None:
        prompt = (
            f"The image shows {n} candidate regions with colored bounding boxes, "
            f"each labeled with a number (1 to {n}). "
            f"Which numbered region is the camouflaged target "
            f"(most similar to background, hardest to spot)? "
            f"Answer with the number only."
        )
        try:
            ann = qwen.generate_with_text(annotated, prompt,
                                          system_prompt=_VISUAL_SYS)
            answer = str(ann.get("answer", "")).strip()
        except Exception as e:
            print(f">>> qwen visual select failed ({e}), fallback to heuristic")
            return heuristic_select(tags, masks, scores, max_aux=max_aux)
        m = re.search(r"(\d+)", answer)
        if m:
            k = int(m.group(1))
            if 1 <= k <= n:
                target_idx = k - 1
                target, aux, target_tag, aux_tags = _assemble(
                    tags, bin_masks, target_idx, max_aux)
                target_score = float(scores[target_idx]) if target_idx < len(scores) else 0.0
                return target, aux, target_tag, aux_tags, target_score
        print(f">>> qwen answered {answer!r} but expected 1-{n}, fallback to heuristic")
        return heuristic_select(tags, masks, scores, max_aux=max_aux)

    # ---- fallback: original tag-list prompt (no OpenCV / no draw_dir) ----
    _TAG_SYS = ("You are a visual analysis assistant. "
                "Answer with one region description only. No JSON, no explanation.")
    tag_list = ", ".join(tags[:n])
    prompt = (
        f"Candidate region descriptions: {tag_list}. "
        f"Which one is the camouflaged target (similar to background, hard to spot)? "
        f"Answer with ONE description only."
    )
    try:
        ann = qwen.generate_with_text(image_path, prompt, system_prompt=_TAG_SYS)
        answer = str(ann.get("answer", "")).strip()
    except Exception as e:
        print(f">>> qwen select failed ({e}), fallback to heuristic")
        return heuristic_select(tags, masks, scores, max_aux=max_aux)

    target_idx = _match_tag(tags, answer)
    if target_idx is None:
        print(f">>> qwen answer did not match any tag ({answer!r}), fallback to heuristic")
        return heuristic_select(tags, masks, scores, max_aux=max_aux)
    target, aux, target_tag, aux_tags = _assemble(tags, bin_masks, target_idx, max_aux)
    target_score = float(scores[target_idx]) if target_idx < len(scores) else 0.0
    return target, aux, target_tag, aux_tags, target_score



# ---------------- pipeline ----------------
def process_split(regions_dir, out_dir, split, mode, qwen=None, image_root=None,
                  size=352, resume=True, min_target_ratio=0.005, conf_thresh=0.6,
                  max_aux=3, draw_dir=None, max_cand=5, file_list=None):
    """Process sam3_regions → prior_sam3.

    file_list: optional list of specific npz Paths to process (for multi-GPU shards).
               If None, auto-discovers all {split}_*.npz in regions_dir.
    """
    regions_dir = Path(regions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_list is not None:
        files = [Path(f) for f in file_list]
    else:
        files = sorted(regions_dir.glob(f"{split}_*.npz"))
    print(f">>> Selecting regions ({mode}) for {split}: {len(files)} files")
    print(f">>> quality gate: min_target_ratio={min_target_ratio}, "
          f"conf_thresh={conf_thresh}, max_aux={max_aux}")

    tmp_dir = None
    if draw_dir is None and mode == "qwen":
        tmp_dir = tempfile.mkdtemp(prefix="rs_draw_")   # auto-clean below

    n_ok = 0
    n_gated = 0
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
            aux = np.zeros((max_aux, H, W), np.float32)
            target_tag = ""
            target_score = 0.0
        else:
            if mode == "qwen" and qwen is not None and image_root is not None:
                rest = f.name[len(f"{split}_"):]          # camourflage_00001_jpg.npz
                rest = rest[:-4]                              # drop .npz
                _base, _ext = rest.rsplit("_", 1)            # (camourflage_00001, jpg)
                img_name = f"{_base}.{_ext}"                 # camourflage_00001.jpg
                img_path = os.path.join(image_root, img_name)
                target, aux, target_tag, _, target_score = qwen_select(
                    img_path, tags, masks, scores, qwen, max_aux=max_aux,
                    draw_dir=(draw_dir or tmp_dir), max_cand=max_cand)
            else:
                target, aux, target_tag, _, target_score = heuristic_select(
                    tags, masks, scores, max_aux=max_aux)

            # ---- prior quality gate: drop tiny / low-confidence targets ----
            H, W = target.shape[-2:]
            area_ratio = float(target.sum()) / (H * W)
            if area_ratio < min_target_ratio or target_score < conf_thresh:
                target = np.zeros((1, H, W), np.float32)
                aux = np.zeros((max_aux, H, W), np.float32)
                target_tag = ""
                target_score = 0.0
                n_gated += 1

        np.savez(out_path, target_mask=target, aux_masks=aux,
                 target_tag=np.array([target_tag], dtype=object), bbox=bbox)
        n_ok += 1
        if idx % 20 == 0:
            print(f"[{idx}/{len(files)}] done, target_tag={target_tag}")

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f">>> Done {mode}: {n_ok}/{len(files)} -> {out_dir}, "
          f"zero-prior by quality gate: {n_gated}")


def _run_shard_worker_rs(opt):
    """Subprocess worker: load Qwen on one GPU, process a shard of npz files."""
    from llm_annotator import QwenVLAnnotator
    qwen = QwenVLAnnotator(model_path=opt.qwen_model)
    with open(opt.shard_files, "r", encoding="utf-8") as f:
        file_list = json.load(f)
    print(f"[GPU {opt.gpu_id}] Qwen loaded, {len(file_list)} npz files")
    process_split(opt.regions, opt.out, opt.split, opt.mode, qwen=qwen,
                  image_root=opt.image_root, resume=not opt.no_resume,
                  min_target_ratio=opt.min_target_ratio, conf_thresh=opt.conf_thresh,
                  max_aux=opt.max_aux, draw_dir=opt.draw_dir, max_cand=opt.max_cand,
                  file_list=file_list)


def _run_multi_gpu_rs(gpus, regions_dir, split, out_dir, image_root,
                       qwen_model, resume, min_target_ratio, conf_thresh,
                       max_aux, draw_dir, max_cand):
    """Split npz files across GPUs, run each as a subprocess."""
    import subprocess, sys, tempfile

    regions_dir = Path(regions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_files = sorted(str(f) for f in regions_dir.glob(f"{split}_*.npz"))
    pending = [fp for fp in all_files if not (resume and (out_dir / Path(fp).name).exists())]

    shards = [[] for _ in range(len(gpus))]
    for i, fp in enumerate(pending):
        shards[i % len(gpus)].append(fp)

    tmp_dir = tempfile.mkdtemp(prefix="rs_shard_")
    procs = []
    print(f">>> Multi-GPU: {len(gpus)} GPUs, {len(all_files)} total, {len(pending)} pending")
    for i, gid in enumerate(gpus):
        shard_path = os.path.join(tmp_dir, f"shard_gpu{gid}.json")
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(shards[i], f)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--shard_mode", "--gpu_id", str(gid), "--shard_files", shard_path,
               "--regions", str(regions_dir), "--out", str(out_dir),
               "--split", split, "--mode", "qwen",
               "--image_root", image_root, "--qwen_model", qwen_model,
               "--min_target_ratio", str(min_target_ratio),
               "--conf_thresh", str(conf_thresh),
               "--max_aux", str(max_aux), "--max_cand", str(max_cand)]
        if draw_dir:
            cmd += ["--draw_dir", draw_dir]
        if not resume:
            cmd.append("--no_resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"  GPU {gid}: {len(shards[i])} npz files")
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        p.wait()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    n_done = len(list(out_dir.glob(f"{split}_*.npz")))
    print(f">>> Multi-GPU done! {n_done} npz files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, default="sam3_regions")
    parser.add_argument("--out", type=str, default="prior_sam3")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--mode", type=str, default="heuristic", choices=["qwen", "heuristic"])
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--qwen_model", type=str, default="/home/16t/cr/sr/chatsr/Qwen2.5-VL-3B-Instruct/")
    parser.add_argument("--min_target_ratio", type=float, default=0.005)
    parser.add_argument("--conf_thresh", type=float, default=0.6)
    parser.add_argument("--max_aux", type=int, default=3)
    parser.add_argument("--max_cand", type=int, default=5)
    parser.add_argument("--draw_dir", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0",
                        help="comma-separated GPU IDs for parallel qwen mode")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--shard_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_files", type=str, default=None, help=argparse.SUPPRESS)
    opt = parser.parse_args()

    if opt.shard_mode:
        _run_shard_worker_rs(opt)
    else:
        gpu_list = [int(x.strip()) for x in opt.gpus.split(",") if x.strip()]
        if opt.mode == "qwen" and len(gpu_list) > 1:
            assert opt.image_root is not None, "--image_root required for qwen mode"
            _run_multi_gpu_rs(gpu_list, opt.regions, opt.split, opt.out, opt.image_root,
                              opt.qwen_model, not opt.no_resume, opt.min_target_ratio,
                              opt.conf_thresh, opt.max_aux, opt.draw_dir, opt.max_cand)
        else:
            qwen = None
            if opt.mode == "qwen":
                assert opt.image_root is not None, "--image_root required for qwen mode"
                from llm_annotator import QwenVLAnnotator
                qwen = QwenVLAnnotator(model_path=opt.qwen_model)
            process_split(opt.regions, opt.out, opt.split, opt.mode, qwen=qwen,
                          image_root=opt.image_root, resume=not opt.no_resume,
                          min_target_ratio=opt.min_target_ratio,
                          conf_thresh=opt.conf_thresh, max_aux=opt.max_aux,
                          draw_dir=opt.draw_dir, max_cand=opt.max_cand)
