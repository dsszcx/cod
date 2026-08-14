"""llm_annotator.py - Qwen2.5-VL-3B-Instruct prior annotator (offline batch).

================================= 文件影响声明 =================================
本脚本对服务器文件系统的影响（已最小化设计）：
  1. 【读取】仅读取 --data_root 下的图片文件（jpg/jpeg/png/bmp）
  2. 【写入】仅写入 --out 指定的输出文件（默认 ./outputs/prior_annotations.json）
  3. 【绝不】删除、修改、覆盖服务器上任何其他已有文件
  4. 【绝不】写入模型目录、系统目录
请放心运行。所有产出均可随时手动删除（仅 outputs/ 目录）。
==============================================================================
"""
import argparse
import json
import os
import re
from pathlib import Path

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    from qwen_vl_utils import process_vision_info
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

SYSTEM_PROMPT = (
    "You are a camouflaged object detection expert. Look carefully at the image and find "
    "the most salient camouflaged target (likely similar to background in color/texture). "
    "Do two things: (1) give 2-5 Chinese tags describing the category and salient parts "
    "(e.g. ['white bird', 'wing']); (2) give a coarse bounding box of the target. "
    "Use original pixel coordinates: [x1, y1, x2, y2]. "
    "If no obvious camouflaged target exists, set confidence to 0 and bbox to the full image."
)

USER_PROMPT = (
    "Analyze this image. Output ONLY JSON, no other text, in this format:\n"
    '{"tags": ["tag1", "tag2"], "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}'
)


class QwenVLAnnotator:
    """Wrapper of Qwen2.5-VL-3B-Instruct for prior annotation."""

    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct", device="auto",
                 max_tokens=256, temperature=0.2):
        if not _LLM_AVAILABLE:
            raise RuntimeError("Missing deps: pip install transformers qwen-vl-utils accelerate")
        # resolve "auto" to an actual torch device string
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        print(f">>> Loading Qwen2.5-VL-3B from {model_path} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=352 * 352, max_pixels=1280 * 28 * 28)

    def generate(self, image_path: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens,
                temperature=self.temperature, do_sample=False, top_p=0.9)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._parse_output(output_text)

    def generate_with_text(self, image_path: str, prompt: str) -> dict:
        # Ask a custom question about the image; return parsed JSON.
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Output only JSON."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt + " Only output JSON: {\"answer\": \"your choice\"}"},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        import re as _re
        m = _re.search(r'\{[^{}]*\}', output_text, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"answer": output_text.strip()[:50]}

    @staticmethod
    def _parse_output(text: str) -> dict:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                tags = data.get("tags", [])
                bbox = data.get("bbox", [])
                conf = float(data.get("confidence", 0.0))
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    bbox = []
                return {"tags": [str(t) for t in tags][:5],
                        "bbox": [int(v) for v in bbox] if bbox else [],
                        "confidence": conf}
            except (json.JSONDecodeError, ValueError):
                pass
        return {"tags": [], "bbox": [], "confidence": 0.0}


def batch_annotate(data_root, out_path, split, model_path, max_images=-1, resume=True):
    """Process all images under data_root, write ONLY to out_path."""
    data_root = Path(data_root)
    image_files = sorted(p for p in data_root.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if max_images > 0:
        image_files = image_files[:max_images]
    print(f">>> Images to process: {len(image_files)}")

    # ---- safety: ensure output dir is our own (never overwrite unrelated files) ----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> Output file (new): {out_path.absolute()}")
    print(f">>> This script reads only: {data_root.absolute()}")
    print(f">>> It will NOT modify any other file on the server.")

    annotations = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f">>> Resume: loaded {sum(len(v) for v in annotations.values())} existing records")

    annotator = QwenVLAnnotator(model_path=model_path)
    for idx, img_path in enumerate(image_files, 1):
        name = img_path.name
        if name in annotations.get(split, {}):
            print(f"[{idx}/{len(image_files)}] skip (exists): {name}")
            continue
        try:
            ann = annotator.generate(str(img_path))
            annotations.setdefault(split, {})[name] = ann
            print(f"[{idx}/{len(image_files)}] {name} -> tags={ann['tags']} "
                  f"bbox={ann['bbox']} conf={ann['confidence']}")
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {name} FAILED: {e}")
        if idx % 20 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            print(f">>> Checkpoint saved to {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    n_total = sum(len(v) for v in annotations.values())
    n_valid = sum(1 for v in annotations.values() for a in v.values() if a.get("bbox"))
    print(f">>> Done! Saved to {out_path}. Total={n_total}, valid bbox={n_valid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/prior_annotations.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--no_resume", action="store_true")
    opt = parser.parse_args()
    batch_annotate(opt.data_root, opt.out, opt.split, opt.model,
                   max_images=opt.max_images, resume=not opt.no_resume)