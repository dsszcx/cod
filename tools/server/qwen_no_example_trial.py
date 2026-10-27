"""Repeat Qwen on the same P0 inputs, changing only the concrete example."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C
from tools.e0e4 import caption_qwen as Q
from tools.e0e4 import prompt_no_category_example as P


def stats(records):
    texts = [o["sam_text"] for r in records for o in r.get("objects", [])]
    return {"n_images": len(records), "statuses": dict(Counter(r["status"] for r in records)),
            "frog_example_images": sum(any(o["sam_text"].strip().lower() == "a mottled brown frog"
                                              for o in r.get("objects", [])) for r in records),
            "placeholder_images": sum("<category>" in r["raw_output"] or "<noun phrase>" in r["raw_output"]
                                      for r in records),
            "most_common_texts": Counter(texts).most_common(8)}


def main():
    import yaml

    source = ROOT / "outputs/MASK_PROMPT_P0_20261022"
    out = ROOT / "outputs/QWEN_NO_EXAMPLE_20261022"
    out.mkdir(exist_ok=True)
    cfg_data = yaml.safe_load((ROOT / "configs/mask_prompt_20261022.yaml").read_text(encoding="utf-8"))
    cfg_data["version"] = "qwen_no_category_example_20261022_v2"
    cfg_data["qwen"]["prompt_version"] = P.PROMPT_VERSION
    cfg_data["paths"]["out_dir"] = str(out)
    config_path = out / "CONFIG.yaml"
    config_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cfg = C.load_config(str(config_path))
    shutil.copytree(source / "rois", out / "rois")
    for name in ("manifest_p0.txt", "INPUT_MANIFEST.csv", "ROI_STATUS.csv"):
        shutil.copy2(source / name, out / name)
    ids = C.read_id_file(str(out / "manifest_p0.txt"))
    # These JSON files still point to the exact original composite PNGs.
    # No image, crop, text length limit, sampling setting or parsing rule changes.
    Q.build_prompts = P.build_prompts
    print("Changed only concrete JSON example; same 30 original/composite inputs.", flush=True)
    started = time.time()
    captioner = Q.QwenCaptioner(cfg["paths"]["qwen_model"], max_tokens=cfg["qwen"]["max_tokens"])
    summary = {"source": str(source), "out": str(out), "config_sha256": cfg["_sha256"], "branches": {}}
    comparison = []
    for branch in ("e2", "e3"):
        fresh = Q.run_stage(cfg, str(out), ids, branch, captioner=captioner)
        old = C.read_jsonl(str(source / "captions" / (branch + ".jsonl")))
        old_by_id = {r["image_id"]: r for r in old}
        summary["branches"][branch] = {"old": stats(old), "new": stats(fresh)}
        for rec in fresh:
            prev = old_by_id[rec["image_id"]]
            comparison.append({"image_id": rec["image_id"], "group": rec["group"], "branch": branch,
                               "old_status": prev["status"], "new_status": rec["status"],
                               "old_texts": " | ".join(o["sam_text"] for o in prev.get("objects", [])),
                               "new_texts": " | ".join(o["sam_text"] for o in rec.get("objects", [])),
                               "new_raw_output": rec["raw_output"]})
        print(branch, json.dumps(summary["branches"][branch]), flush=True)
    Q.write_text_table(str(out))
    C.write_csv(str(out / "OLD_NEW_COMPARISON.csv"), comparison)
    summary["elapsed_s"] = round(time.time() - started, 1)
    C.write_json(str(out / "COMPARISON_SUMMARY.json"), summary)
    print("QWEN_NO_EXAMPLE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
