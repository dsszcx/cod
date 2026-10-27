"""Probe the 16 E3 EMPTY cases without changing the completed P0 run."""
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tools.e0e4.segment_sam3 import load_processor, write_mask_png

    source = ROOT / "outputs/MASK_PROMPT_P0_20261022"
    out = ROOT / "outputs/EMPTY_DIAG_20261022"
    out.mkdir(exist_ok=True)
    cfg = C.load_config(str(ROOT / "configs/mask_prompt_20261022.yaml"))
    status = C.read_csv(str(source / "sam3/e3/SAM3_e3_STATUS.csv"))
    selected = [r for r in status if r["status"] == "EMPTY"]
    print("Diagnostic EMPTY images:", len(selected), flush=True)
    processor = load_processor(C.resolve_path(cfg["paths"]["sam3_ckpt"]), 0.5)
    original_forward = processor.model.forward_grounding
    raw = {}

    def capture(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        cls = outputs["pred_logits"].sigmoid()
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        scores = (cls * presence).squeeze(-1).reshape(-1)
        best = int(scores.argmax().item())
        masks = outputs["pred_masks"]
        raw.clear()
        raw.update(scores=scores.detach().float().cpu().numpy(),
                   presence=presence.detach().float().cpu().reshape(-1).tolist(),
                   top_logits=masks.reshape(-1, *masks.shape[-2:])[best].detach().float().cpu())
        return outputs

    processor.model.forward_grounding = capture
    records = []
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, row in enumerate(selected, 1):
            ident = row["image_id"]
            dataset, stem = ident.split("/", 1)
            caption = C.read_json(str(source / "captions/e3" / (ident + ".json")))
            roi = C.read_json(str(source / "rois" / (ident + ".json")))
            image_path = Path(cfg["paths"]["data_root"]) / dataset / "Imgs" / (stem + ".jpg")
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            for obj_idx, obj in enumerate(caption["objects"], 1):
                category = obj["name"].split()[-1].strip(".,;:").lower()
                matching = [r for r in roi["rois"] if r["roi_id"] == obj.get("region_id")]
                chosen = (matching or roi["rois"])[0] if roi["rois"] else None
                for mode, text in (("original_text", obj["sam_text"]),
                                   ("category_word", category),
                                   ("roi_plus_text", obj["sam_text"])):
                    state = processor.set_image(image)
                    if mode == "roi_plus_text" and chosen:
                        state = processor.add_geometric_prompt(
                            box=chosen["normalized_xywh"], label=True, state=state)
                    raw.clear()
                    state = processor.set_text_prompt(prompt=text, state=state)
                    scores = raw["scores"]
                    best_score = float(scores.max())
                    logits = raw["top_logits"][None, None]
                    prob = F.interpolate(logits, size=(image.height, image.width),
                                         mode="bilinear", align_corners=False).sigmoid()[0, 0].numpy()
                    top_mask = prob > 0.5
                    path = out / "top_candidate_masks" / mode / dataset / (stem + ".png")
                    write_mask_png(str(path), top_mask)
                    record = {"image_id": ident, "group": row["group"],
                              "object_index": obj_idx, "mode": mode, "text": text,
                              "top_score": round(best_score, 8),
                              "n_above_05": int((scores > 0.5).sum()),
                              "n_above_03": int((scores > 0.3).sum()),
                              "n_above_01": int((scores > 0.1).sum()),
                              "presence": raw["presence"],
                              "top5_scores": sorted(scores.tolist(), reverse=True)[:5],
                              "top_mask_fraction": round(float(top_mask.mean()), 6),
                              "roi_id": chosen["roi_id"] if mode == "roi_plus_text" and chosen else None,
                              "mask_note": "Top candidate saved even below confidence threshold; diagnostic only."}
                    records.append(record)
                    with open(out / "PROBES.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print("[%d/%d] %s %s score=%.5f >0.5=%d" %
                          (idx, len(selected), ident, mode, best_score, record["n_above_05"]), flush=True)
                    del state
    C.write_json(str(out / "DIAGNOSTIC_RESULTS.json"), {
        "n_images": len(selected), "n_probes": len(records),
        "elapsed_s": round(time.time() - started, 1), "records": records,
        "source_config_sha256": cfg["_sha256"],
        "selection": "All E3 EMPTY cases from completed P0; no GT used in prompting.",
        "category_rule": "Last word of Qwen object name; not a label from filenames or GT.",
    })
    print("EMPTY_DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
