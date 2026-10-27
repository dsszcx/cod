"""llm_annotator.py - Qwen prior annotator (simplified Step1).
Annotations: tags + bbox for sam3_regions.py (Step2).
Supports: filename context, multi-GPU (--gpus), resume, --num_runs voting.

Optimizations (2026-09):
  - OpenCV saliency pre-screening: detect candidate regions before Qwen,
    so Qwen only needs to NAME regions instead of finding + describing + locating.
  - Simplified prompts: reduce Qwen 3B cognitive load.
  - Image enhancement: improve Qwen's visual perception.
  - Dual-mode: prescreen mode (default) + legacy open-ended mode.
"""
import argparse, json, os, re
import numpy as np
from pathlib import Path

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    from qwen_vl_utils import process_vision_info
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False

# --------------- prompts (English, simplified for 3B model) ---------------

# Legacy open-ended prompt (fallback)
SYSTEM_PROMPT = (
    "You are a camouflage expert. "
    "Identify 3-5 objects in this image that could be camouflaged. "
    "For each, give a short description and approximate bounding box."
)

USER_PROMPT = (
    "List 3-5 objects that might be camouflaged in this image. "
    "For each, give: description, [x1,y1,x2,y2]. "
    "Example: fox [68,73,279,274], lizard [200,100,500,400]"
)

# Prescreen prompt: Qwen sees numbered boxes, just names each region
PRESCREEN_SYSTEM_PROMPT = (
    "You are a camouflage expert. The image shows numbered candidate regions. "
    "For each region, give a short English noun phrase describing the object inside."
)

PRESCREEN_USER_PROMPT_TEMPLATE = (
    "The image has {n} numbered bounding boxes drawn on it. "
    "For each box, briefly describe what object is inside using an English noun phrase. "
    "Reply in this exact format (one per line):\n"
    "1: description\n"
    "2: description\n"
    "...\n"
    "Do NOT include coordinates. Example:\n"
    "1: white bird\n"
    "2: green leaf\n"
    "3: rock"
)

def extract_context(filename):
    if not filename: return ""
    if filename.startswith("COD10K"):
        parts = filename.replace("COD10K-", "").split("-")
        cat = parts[4] if len(parts) > 4 else ""
        return f" This is likely a {cat}." if cat else ""
    elif filename.startswith("camourflage"):
        return " This is a camouflage detection image."
    return ""

BACKGROUND_TAGS = {"rock","ground","texture","soil","tree bark","leaf",
    "grass","water","sky","sand","terrain","snow","cloud","mud","bark",
    "moss","dirt","stone","pebble","earth","floor","wall","branch",
    "trunk","stem","petal","flower"}

# Words a 3B model uses instead of naming an object ("background" was actually
# seen in the prescreen output).  They are filtered by _normalize_items on top
# of BACKGROUND_TAGS: a tag here would otherwise be pushed into SAM3 as a text
# prompt (Step2 --prompt_mode hybrid) and simply waste a forward pass.
NON_OBJECT_TAGS = {"background", "foreground", "none", "unknown", "nothing",
    "object", "something", "camouflage", "camouflaged object"}

# Prose/refusal patterns.  They only guard the free-text branch of
# _parse_multi_target: answering the legacy open-ended prompt with
# "I cannot see any camouflaged object" used to store that whole sentence as a
# tag.  \b keeps real nouns safe (hare/pig/wasp/spider do not match).
_JUNK_DESC_RE = re.compile(
    r"\b(cannot|can't|can not|unable|sorry|please|not sure|no camouflag|"
    r"there is no|there are no|is|are|was|were|appears|looks|seems|shown|"
    r"visible|image|photo|picture|brightness|color)\b", re.I)

# --------------- OpenCV saliency / pre-screening ---------------

def opencv_detect_candidates(image_path, max_cand=5, min_area_ratio=0.005,
                              max_area_ratio=0.85):
    """Use OpenCV to detect candidate salient regions before Qwen.

    Pipeline:
      1. Saliency detection (spectral residual or fine-grained if available)
      2. Threshold + morphological cleanup
      3. Connected components with area filtering
      4. Merge overlapping boxes, keep top-K by area

    Returns: list of [x1, y1, x2, y2] boxes (int), or empty list on failure.
    """
    if not _CV2_OK:
        return []

    img = cv2.imread(image_path)
    if img is None:
        return []
    ih, iw = img.shape[:2]
    img_area = ih * iw
    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)

    # --- Step 1: saliency map ---
    saliency_map = None
    try:
        # Try spectral residual saliency (OpenCV 3.4.2+)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img)
        if not success:
            saliency_map = None
    except AttributeError:
        pass

    if saliency_map is None:
        # Fallback: multi-scale edge + gradient based saliency
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gaussian blur at multiple scales for center-surround contrast
        blur_small = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_large = cv2.GaussianBlur(gray, (21, 21), 0)
        saliency_map = cv2.absdiff(blur_small, blur_large).astype(np.float32)
        # Also add edge information
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        saliency_map = saliency_map + edges * 0.5

    # Normalize to [0, 255]
    saliency_map = (saliency_map - saliency_map.min()) / \
                   (saliency_map.max() - saliency_map.min() + 1e-8)
    sal_8bit = (saliency_map * 255).astype(np.uint8)

    # --- Step 2: threshold + morphological cleanup ---
    _, binary = cv2.threshold(sal_8bit, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # --- Step 3: connected components + area filtering ---
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for i in range(1, num_labels):  # skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        # Expand bbox slightly for context (10% padding)
        pad_w, pad_h = int(w * 0.10), int(h * 0.10)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(iw, x + w + pad_w)
        y2 = min(ih, y + h + pad_h)
        candidates.append({
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "cx": centroids[i][0],
            "cy": centroids[i][1],
        })

    # --- Step 4: merge overlapping boxes, keep top-K ---
    candidates.sort(key=lambda c: c["area"], reverse=True)
    merged = []
    for c in candidates:
        overlap = False
        for m in merged:
            iou = _bbox_iou(c["bbox"], m["bbox"])
            if iou > 0.5:
                overlap = True
                # Keep the larger one (already sorted, so m is larger)
                break
        if not overlap:
            merged.append(c)
        if len(merged) >= max_cand:
            break

    return [c["bbox"] for c in merged]


def fallback_boxes(image_path, max_cand=5, mode="quad"):
    """Boxes used when opencv_detect_candidates() returns nothing.

    The 3B model cannot find + locate + name a camouflaged object in one
    open-ended shot -- that path produced the 2026-10-15 "SKIP (no tags)" flood
    and the "[1, 2]" junk that crashed the parser.  It CAN name an object inside
    a *given* box, and SAM3 turns a loose box into a mask, so when the saliency
    proposer comes up empty we hand over the whole image (plus overlapping
    quadrants, for context) instead of dropping to the open-ended prompt.

    mode: "quad"  -> whole image + 4 quadrants (12% overlap, so an object sitting
                     on a quadrant border is not cut in half)
          "whole" -> whole image only
          "off"   -> [] (caller falls back to legacy)

    Returns [[x1,y1,x2,y2], ...] with the whole-image box first, or [] when the
    image cannot be read at all.
    """
    if mode == "off":
        return []
    w = h = 0
    if _CV2_OK:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
    if not w or not h:
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            return []
    if not w or not h:
        return []
    whole = [0, 0, int(w), int(h)]
    if mode == "whole" or max_cand <= 1:
        return [whole]

    mx, my = int(w * 0.12), int(h * 0.12)          # quadrant overlap margin
    hw, hh = int(w / 2), int(h / 2)
    quads = [
        [0, 0, min(int(w), hw + mx), min(int(h), hh + my)],    # top-left
        [max(0, hw - mx), 0, int(w), min(int(h), hh + my)],    # top-right
        [0, max(0, hh - my), min(int(w), hw + mx), int(h)],    # bottom-left
        [max(0, hw - mx), max(0, hh - my), int(w), int(h)],    # bottom-right
    ]
    boxes = [whole]
    for b in quads:
        if len(boxes) >= max_cand:
            break
        if (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0:
            boxes.append([int(v) for v in b])
    return boxes


def _bbox_iou(b1, b2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-8)


def draw_numbered_boxes(image_path, boxes, out_path):
    """Draw numbered bounding boxes on the image for Qwen visual prompting.

    Each box is drawn with a distinct color and a large numbered label.
    Returns out_path on success, None on failure.
    """
    if not _CV2_OK:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None

    colors = [
        (0, 0, 255),    # red
        (255, 0, 0),    # blue
        (0, 200, 0),    # green
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
    ]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors[i % len(colors)]
        # Draw filled background for number
        label = str(i + 1)
        # Thick rectangle border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # Number label with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(img, (x1, max(y1 - th - 10, 0)), (x1 + tw + 10, y1),
                      color, -1)
        cv2.putText(img, label, (x1 + 5, max(y1 - 5, th + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path


def enhance_image_for_qwen(image_path, out_path, target_size=800):
    """Enhance image quality before sending to Qwen (subtle, non-destructive).

    - Resize small images up for better Qwen perception
    - Slight contrast enhancement (CLAHE on L channel)
    - Denoise if image is very noisy

    Returns out_path on success, original image_path if no enhancement needed.
    """
    if not _CV2_OK:
        return image_path
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    h, w = img.shape[:2]
    enhanced = False

    # Resize small images
    if max(h, w) < target_size:
        scale = target_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
        enhanced = True

    # CLAHE contrast enhancement on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    # Only enhance if image is low-contrast
    if l_channel.std() < 50:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced = True

    if enhanced:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img)
        return out_path
    return image_path


# --------------- parse prescreen Qwen output ---------------

def _parse_prescreen_output(text, boxes):
    """Parse Qwen's prescreen response: '1: description\\n2: description\\n...'

    Each line should be 'N: description'. Returns list of
    {"tags": [description], "bbox": [x1,y1,x2,y2]}.
    """
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\s*[:：]\s*(.+)", line)
        if m:
            idx = int(m.group(1)) - 1  # 1-indexed to 0-indexed
            desc = m.group(2).strip().strip("\"'.")
            # Clean up: remove coordinates if model still included them
            desc = re.sub(r"\s*\[.*?\]\s*", "", desc).strip()
            if desc and 0 <= idx < len(boxes):
                tags = [t.strip() for t in re.split(r"[,/]", desc) if t.strip()]
                tags = [t for t in tags if t.lower() not in BACKGROUND_TAGS]
                if tags:
                    results.append({"tags": tags, "bbox": _validate_bbox(boxes[idx])})

    # If parsing failed entirely, try legacy parser (normalised: a bare index list
    # such as "[1, 2]" is junk, not an object description -> [] instead of a crash)
    if not results:
        return _normalize_items(_parse_multi_target(text))

    return _normalize_items(results)


def _validate_bbox(bbox, img_w=0, img_h=0):
    """Validate and clamp bbox to image bounds.

    When img_w/img_h are provided and >0, checks minimum area (0.1% of image).
    When img_w/img_h are 0 (unknown), only does basic clamping.
    """
    x1,y1,x2,y2 = [max(0,int(v)) for v in bbox]
    if img_w > 0 and img_h > 0:
        x1,y1 = max(0,min(x1,img_w)), max(0,min(y1,img_h))
        x2,y2 = max(x1+1,min(x2,img_w)), max(y1+1,min(y2,img_h))
        if (x2-x1)*(y2-y1) < img_w*img_h*0.001:
            return [0,0,img_w,img_h]
    else:
        # Unknown image size: ensure positive area
        x2,y2 = max(x1+1,x2), max(y1+1,y2)
    return [x1,y1,x2,y2]

def _is_junk_desc(desc):
    """True when a free-text fragment is prose/refusal, not an object name.

    Guards the free-text branch of _parse_multi_target only.  The legacy
    open-ended prompt made the 3B model answer things like "I cannot see any
    camouflaged object in this image" or "brightness changes"; storing those as
    tags then pushed a full sentence into SAM3 as a text prompt (Step2).
    """
    return bool(_JUNK_DESC_RE.search(desc)) or len(desc.split()) > 6


def _normalize_items(items):
    """Coerce parser output into [{"tags": [str], "bbox": [x1,y1,x2,y2]}].

    Root cause of the 2026-10-15 server crash: Qwen answered with a bare JSON
    array of indices ("[1, 2]"), _parse_multi_target returned it verbatim and
    prescreen_generate_multi called r.get("tags") on an int ->
    AttributeError: 'int' object has no attribute 'get'.

    Anything that is not a usable object description (int / None / nested list /
    empty or background-only tags) is dropped here, so a junk reply degrades to
    an empty list -- which every caller already handles -- instead of raising.
    """
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, (list, tuple)):
        return []
    out = []
    for it in items:
        if isinstance(it, dict):
            tags, bbox = it.get("tags"), it.get("bbox")
        elif isinstance(it, str):
            tags, bbox = [it], None
        else:
            continue                      # int / None / nested list: not an object
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, (list, tuple)):
            continue
        clean = []
        for t in tags:
            if not isinstance(t, str):
                continue
            t = t.strip().strip("\"'.").strip()
            if not t or len(t) >= 60:
                continue
            if t.isdigit():                     # a bare index ("1") is not a name
                continue
            if t.lower() in BACKGROUND_TAGS or t.lower() in NON_OBJECT_TAGS:
                continue
            if t not in clean:
                clean.append(t)
        if not clean:
            continue
        try:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                bbox = [0, 0, 0, 0]
            bbox = _validate_bbox([int(v) for v in bbox])
        except (TypeError, ValueError):
            bbox = [0, 0, 0, 0]
        out.append({"tags": clean, "bbox": bbox})
    return out


def _parse_multi_target(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text).strip()
    try:
        d = json.loads(text)
        if isinstance(d, list) and d: return _normalize_items(d)
        if isinstance(d, dict): return _normalize_items([d])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Split between items WITHOUT consuming the closing bracket: the old
    # r"\]\s*,\s*" ate the "]" of everything but the last item, so
    # "fox [68,73,279,274], lizard [...]" lost the fox box (it silently became
    # a tag without coordinates).  Lookbehind keeps the "]" inside the part.
    parts = re.split(r"(?<=\])\s*,\s*|\)\s*,\s*|\n", text)
    results = []
    for part in parts:
        part = part.strip().strip(",").strip()
        if not part: continue
        m = re.search(r"(.+?)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", part)
        if m:
            tags = [t.strip().strip("\"'") for t in m.group(1).split(",") if t.strip()]
            bbox = [int(m.group(i)) for i in range(2,6)]
            if tags: results.append({"tags":tags,"bbox":_validate_bbox(bbox)})
        else:
            desc = re.sub(r"[\[\]\d,;]", "", part).strip().strip("\'")
            desc = re.sub(r"^[\s\)\]\.\:]+", "", desc).strip()   # leftover "1)" numbering
            if desc and len(desc) < 50 and not _is_junk_desc(desc):
                results.append({"tags":[desc],"bbox":[0,0,0,0]})
    # Single normalisation point for the regex/line-parsed branch as well.
    return _normalize_items(results)

class QwenVLAnnotator:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct", device="auto",
                 max_tokens=256, temperature=0.2):
        if not _LLM_AVAILABLE:
            raise RuntimeError("pip install transformers qwen-vl-utils accelerate")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device; self.max_tokens = max_tokens; self.temperature = temperature
        print(f">>> Loading Qwen from {model_path} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=352*352, max_pixels=1280*28*28)

    def _call_model(self, image_path, system_prompt, user_prompt, max_tokens=None):
        """Common model call. Returns raw output text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        mt = max_tokens or self.max_tokens
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=mt, do_sample=False)
        gen = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    # ---- legacy open-ended mode ----
    def generate(self, image_path, filename=""):
        prompt = USER_PROMPT + (f"\n\n{extract_context(filename)}" if extract_context(filename) else "")
        output_text = self._call_model(image_path, SYSTEM_PROMPT, prompt)
        return _parse_multi_target(output_text)

    # ---- prescreen mode: OpenCV detects, Qwen names ----
    def prescreen_generate(self, image_path, max_cand=5, tmp_dir=None,
                           box_fallback="quad"):
        """OpenCV detects candidate boxes → draw numbered boxes → Qwen names each.

        Returns list of {"tags": [...], "bbox": [x1,y1,x2,y2]}, same format
        as legacy generate().

        box_fallback (2026-10-15) decides what happens when the proposer finds
        nothing or the reply cannot be parsed:
          "quad"/"whole" -> keep asking about GIVEN boxes (fallback_boxes); the
                            open-ended prompt is then only used when the image
                            itself is unreadable -- it is the path that produced
                            the SKIP flood, so it must not be the first resort.
          "off"          -> legacy behaviour (open-ended prompt, old bugs).
        """
        # Step 1: OpenCV saliency detection, then the box fallback
        boxes = opencv_detect_candidates(image_path, max_cand=max_cand)
        if not boxes and box_fallback != "off":
            boxes = fallback_boxes(image_path, max_cand=max_cand,
                                   mode=box_fallback)
            if boxes:
                print(f"  [prescreen] OpenCV empty -> {len(boxes)} "
                      f"{box_fallback} fallback box(es)")

        if not boxes:
            # Image unreadable (or --box_fallback off): legacy open-ended mode
            print(f"  [prescreen] no usable box, fallback to legacy")
            return self.generate(image_path)

        # Step 2: Enhance image for Qwen
        enhanced_path = image_path
        if tmp_dir:
            enhanced_path = enhance_image_for_qwen(
                image_path, os.path.join(tmp_dir, "enhanced.jpg"))

        # Steps 3-5: draw numbered boxes -> Qwen names each region -> parse
        def _ask(boxes_, tag):
            draw_path = enhanced_path
            if _CV2_OK and tmp_dir:
                drawn = draw_numbered_boxes(
                    enhanced_path, boxes_,
                    os.path.join(tmp_dir, f"numbered_{tag}.jpg"))
                if drawn:
                    draw_path = drawn
            user_prompt = PRESCREEN_USER_PROMPT_TEMPLATE.format(n=len(boxes_))
            output_text = self._call_model(
                draw_path, PRESCREEN_SYSTEM_PROMPT, user_prompt, max_tokens=150)
            print(f"  [prescreen] Qwen raw ({tag}): {output_text!r}")
            return _parse_prescreen_output(output_text, boxes_)

        results = _ask(boxes, "boxes")

        if not results and box_fallback != "off":
            # Unusable reply (e.g. the bare "[1, 2]"): ask once more with a
            # single whole-image box -- still a question the 3B model can
            # answer, unlike the open-ended prompt.
            whole = fallback_boxes(image_path, max_cand=1, mode="whole")
            if whole:
                print("  [prescreen] Parse failed -> retry with whole-image box")
                results = _ask(whole, "whole")

        if not results:
            # Fallback to legacy mode if parsing fails
            print(f"  [prescreen] Parse failed, fallback to legacy")
            return self.generate(image_path)

        return results

    def prescreen_generate_multi(self, image_path, n_runs=1, max_cand=5, tmp_dir=None,
                                 box_fallback="quad"):
        """Multi-run prescreen with voting. Same interface as generate_multi."""
        from collections import Counter

        if n_runs <= 1:
            results = self.prescreen_generate(image_path, max_cand=max_cand,
                                              tmp_dir=tmp_dir,
                                              box_fallback=box_fallback)
            # Guard: only dict entries carrying tags are usable.  A parser
            # regression used to hand a bare int here -> AttributeError on
            # r.get("tags") (the 2026-10-15 crash).
            results = [r for r in results
                       if isinstance(r, dict) and r.get("tags")]
            if results:
                # Return the first result with highest tag count
                best = max(results, key=lambda r: len(r.get("tags", [])))
                return best
            return {"tags": [], "bbox": [0, 0, 0, 0]}

        all_tags = []
        all_bboxes = []
        for _ in range(n_runs):
            results = self.prescreen_generate(image_path, max_cand=max_cand,
                                              tmp_dir=tmp_dir,
                                              box_fallback=box_fallback)
            for r in results:
                if not isinstance(r, dict):
                    continue
                all_tags.extend(r.get("tags", []))
                all_bboxes.append(r.get("bbox", [0, 0, 0, 0]))

        if not all_tags:
            return {"tags": [], "bbox": [0, 0, 0, 0]}

        tc = Counter(all_tags)
        top = [t for t, _ in tc.most_common(5)]
        avg = [round(sum(b[i] for b in all_bboxes) / len(all_bboxes)) for i in range(4)]
        return {"tags": top, "bbox": _validate_bbox(avg),
                "consistency": round(tc.most_common(1)[0][1] / max(1, len(all_tags)), 3)}

    def generate_multi(self, image_path, n_runs=1, filename=""):
        from collections import Counter
        all_results = [self.generate(image_path, filename=filename) for _ in range(max(1, n_runs))]
        if n_runs <= 1:
            for objs in all_results:
                for o in objs:
                    if isinstance(o, dict) and o.get("tags"):
                        return o
            return {"tags":[], "bbox":[0,0,0,0]}
        all_tags, all_bboxes = [], []
        for objs in all_results:
            for o in objs:
                if not isinstance(o, dict):
                    continue                   # junk entry (see _normalize_items)
                all_tags.extend(o.get("tags",[])); all_bboxes.append(o.get("bbox",[0,0,0,0]))
        if not all_tags: return {"tags":[], "bbox":[0,0,0,0]}
        tc = Counter(all_tags); top = [t for t,_ in tc.most_common(5)]
        avg = [round(sum(b[i] for b in all_bboxes)/len(all_bboxes)) for i in range(4)]
        return {"tags":top, "bbox":_validate_bbox(avg),
                "consistency":round(tc.most_common(1)[0][1]/max(1,len(all_results)),3)}


    def generate_with_text(self, image_path, prompt, system_prompt=None):
        sys_prompt = system_prompt or "You are a helpful assistant. Output only JSON."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        gen = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
            elif isinstance(result, list) and result and isinstance(result[0], dict):
                return result[0]
            else:
                return {"answer": str(result)}
        except (json.JSONDecodeError, ValueError):
            return {"answer": raw}


def is_noncam_name(name):
    """COD10K "NonCAM" rule -- SINGLE SOURCE OF TRUTH: dataloader_sam3.is_noncam.

    Those 2960 train images contain no camouflaged object: the prepared GT is
    all-zero and the 8th input channel (neg_flag) is 1 *because of this file-name
    rule*.  A prior for them must therefore stay empty -- asking Qwen would only
    invent objects.  Imported lazily so this module stays importable without
    torch.  Any change to the rule must be made in dataloader_sam3, not here.
    """
    name = os.path.basename(str(name))
    try:
        from dataloader_sam3 import is_noncam
        return bool(is_noncam(name))
    except Exception:
        return "NonCAM" in name


def _annotate_loop(image_files, data_root, split, annotations, out_path, annotator,
                   num_runs=1, mode="legacy", tmp_dir=None, box_fallback="quad"):
    """Main annotation loop. mode='legacy' or 'prescreen'."""
    import tempfile as _tf
    _tmp = tmp_dir or _tf.mkdtemp(prefix="qwen_prescreen_")
    for idx, img_path in enumerate(image_files, 1):
        name = img_path.name
        if name in annotations.get(split, {}): continue
        try:
            if mode == "prescreen":
                ann = annotator.prescreen_generate_multi(
                    str(img_path), n_runs=num_runs, tmp_dir=_tmp,
                    box_fallback=box_fallback)
            else:
                ann = annotator.generate_multi(str(img_path), n_runs=num_runs, filename=img_path.name)
            if not isinstance(ann, dict) or not ann.get("tags"):
                print(f"[{idx}/{len(image_files)}] {name} -> SKIP (no tags)"); continue
            annotations.setdefault(split, {})[name] = ann
            print(f"[{idx}/{len(image_files)}] {name} -> tags={ann['tags']} bbox={ann.get('bbox')}")
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {name} FAILED: {e}")
        if idx % 20 == 0:
            with open(out_path, "w", encoding="utf-8") as f: json.dump(annotations, f, ensure_ascii=False, indent=2)
    with open(out_path, "w", encoding="utf-8") as f: json.dump(annotations, f, ensure_ascii=False, indent=2)
    n = sum(len(v) for v in annotations.values())
    nv = sum(1 for v in annotations.values() for a in v.values() if a.get("tags"))
    print(f">>> Done! Total={n}, valid={nv}")


def batch_annotate(data_root, out_path, split, model_path, max_images=-1, resume=True,
                   filelist=None, gpus=None, num_runs=1, mode="legacy",
                   box_fallback="quad", noncam_policy="skip_train"):
    if gpus is None: gpus = [0]
    data_root = Path(data_root)
    image_files = sorted(p for p in data_root.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if filelist:
        allowed = set()
        with open(filelist, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line: allowed.add(line.split()[0])
        image_files = [p for p in image_files if p.name in allowed]
        print(f">>> filelist: {len(image_files)} images")
    if max_images > 0: image_files = image_files[:max_images]

    # COD10K "NonCAM" images have no camouflaged object (prepared GT is all-zero
    # and the 8th input channel neg_flag=1 follows from the file name, see
    # dataloader_sam3.is_noncam).  Their prior MUST stay empty, so there is
    # nothing to ask Qwen.  Only the train split is skipped by default, because
    # this json also feeds MyTest_SAM3's prior-coverage check.
    if noncam_policy != "annotate" and (split == "train" or noncam_policy == "skip_all"):
        n_before = len(image_files)
        image_files = [p for p in image_files if not is_noncam_name(p.name)]
        print(f">>> skip COD10K 'NonCAM': {n_before - len(image_files)} image(s) "
              f"(no target in GT, neg_flag=1) -> {len(image_files)} left")

    print(f">>> Images: {len(image_files)}, gpus: {gpus}, num_runs: {num_runs}, "
          f"mode: {mode}, box_fallback: {box_fallback}")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    annotations = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f: annotations = json.load(f)
        print(f">>> Resume: {sum(len(v) for v in annotations.values())} existing")
        if noncam_policy != "annotate":
            # Housekeeping: drop stale NonCAM entries of this split so the json
            # only describes images that really contain a target.
            stale = [k for k in annotations.get(split, {}) if is_noncam_name(k)]
            if stale:
                for k in stale:
                    annotations[split].pop(k, None)
                print(f">>> Purged {len(stale)} NonCAM entry/entries from {split}")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)
    if len(gpus) > 1 and len(image_files) > 1:
        _run_multi_gpu(gpus, image_files, data_root, split, model_path, out_path,
                       annotations, num_runs, mode=mode, box_fallback=box_fallback)
    else:
        annotator = QwenVLAnnotator(model_path=model_path)
        _annotate_loop(image_files, data_root, split, annotations, out_path, annotator,
                       num_runs=num_runs, mode=mode, box_fallback=box_fallback)


def _run_shard_worker(opt, num_runs=1):
    from llm_annotator import QwenVLAnnotator
    annotator = QwenVLAnnotator(model_path=opt.model)
    with open(opt.shard_files, "r", encoding="utf-8") as f: file_list = json.load(f)
    data_root = Path(opt.data_root)
    shard_files = [data_root / n for n in file_list]
    print(f"[GPU {opt.gpu_id}] {len(shard_files)} images, mode={opt.mode}")
    annotations = {}
    if opt.out_shard and os.path.exists(opt.out_shard):
        with open(opt.out_shard, "r", encoding="utf-8") as f: annotations = json.load(f)
    _annotate_loop(shard_files, data_root, opt.split, annotations, Path(opt.out_shard),
                   annotator, num_runs=num_runs, mode=opt.mode,
                   box_fallback=getattr(opt, "box_fallback", "quad"))


def _run_multi_gpu(gpus, image_files, data_root, split, model_path, out_path,
                   base_annotations, num_runs=1, mode="legacy", box_fallback="quad"):
    import subprocess, sys, tempfile, shutil
    shards = [[] for _ in range(len(gpus))]
    pending = [f for f in image_files if f.name not in base_annotations.get(split, {})]
    for i, f in enumerate(pending): shards[i % len(gpus)].append(f.name)
    tmp_dir = tempfile.mkdtemp(prefix="qwen_shards_")
    procs, shard_paths = [], []
    print(f">>> Multi-GPU: {len(gpus)} GPUs, {len(pending)} pending, mode={mode}")
    for i, gid in enumerate(gpus):
        sp = os.path.join(tmp_dir, f"shard_gpu{gid}.json"); shard_paths.append(sp)
        lp = os.path.join(tmp_dir, f"list_gpu{gid}.json")
        with open(lp, "w", encoding="utf-8") as f: json.dump(shards[i], f)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--shard_mode",
               "--gpu_id", str(gid),
               "--shard_files", lp, "--data_root", str(data_root), "--split", split,
               "--model", model_path, "--out_shard", sp, "--num_runs", str(num_runs),
               "--mode", mode, "--box_fallback", box_fallback]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gid)
        print(f"  GPU {gid}: {len(shards[i])} images")
        procs.append((gid, len(shards[i]), subprocess.Popen(cmd, env=env)))

    failed = []
    for gid, n_shard, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((gid, rc, n_shard))

    # ---- merge FIRST, clean up AFTER -------------------------------------
    # The old code ran shutil.rmtree(tmp_dir) *before* reading the per-GPU
    # shard json files, so every multi-GPU run silently merged zero shards and
    # only kept whatever was already in base_annotations.  Step1 still looked
    # successful (it just re-wrote the previous json), which is exactly how a
    # 2-GPU run can lose all of its new annotations without a single warning.
    annotations = dict(base_annotations)
    n_merged = n_have = 0
    for sp in shard_paths:
        if not os.path.exists(sp):
            print(f">>> WARNING: shard output missing: {sp}")
            continue
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_have += 1
        for s, entries in data.items():
            annotations.setdefault(s, {}).update(entries)
            n_merged += len(entries)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f">>> merged {n_merged} shard entries from {n_have}/{len(shard_paths)} "
          f"shards; Total={sum(len(v) for v in annotations.values())}")

    if failed or n_have < len(shard_paths):
        # A dead shard used to look identical to a successful run: its images
        # simply had no annotation and fell back to an all-zero prior.
        for gid, rc, n_shard in failed:
            print(f">>> ERROR: GPU {gid} shard ({n_shard} images) exited with "
                  f"code {rc}")
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/prior_annotations.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--filelist", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="prescreen",
                        choices=["legacy", "prescreen"],
                        help="prescreen: OpenCV detects + Qwen names (recommended for 3B). "
                             "legacy: Qwen open-ended (original)")
    parser.add_argument("--box_fallback", type=str, default="quad",
                        choices=["quad", "whole", "off"],
                        help="What to do when the saliency proposer finds no candidate "
                             "(or the reply is unparseable): quad=whole image + 4 "
                             "quadrants (default), whole=whole image only, off=old "
                             "behaviour (fall back to the open-ended prompt)")
    parser.add_argument("--noncam_policy", type=str, default="skip_train",
                        choices=["skip_train", "skip_all", "annotate"],
                        help="COD10K 'NonCAM' images (GT all-zero, neg_flag=1, no target): "
                             "skip_train=skip them in the train split only (default), "
                             "skip_all=skip in both splits, annotate=keep old behaviour")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--shard_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_files", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out_shard", type=str, default=None, help=argparse.SUPPRESS)
    opt = parser.parse_args()
    if opt.shard_mode:
        _run_shard_worker(opt, num_runs=opt.num_runs)
    else:
        gpu_list = [int(x.strip()) for x in opt.gpus.split(",") if x.strip()]
        batch_annotate(opt.data_root, opt.out, opt.split, opt.model,
                       max_images=opt.max_images, resume=not opt.no_resume,
                       filelist=opt.filelist, gpus=gpu_list, num_runs=opt.num_runs,
                       mode=opt.mode, box_fallback=opt.box_fallback,
                       noncam_policy=opt.noncam_policy)
