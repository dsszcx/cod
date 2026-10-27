"""Evaluate downloaded server predictions against the matching local dataset."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C
from tools.e0e4 import eval_strict, make_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="outputs/mask_prompt_p0_results_20261022.tar.gz")
    parser.add_argument("--out", default="outputs/MASK_PROMPT_P0_20261022")
    parser.add_argument("--config-rel", default="RUN_LOGS/CONFIG_LOCKED.yaml")
    args = parser.parse_args()
    archive_path = ROOT / args.archive
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (out / member.name).resolve()
            if out.resolve() not in target.parents and target != out.resolve():
                raise ValueError(member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError("Unexpected archive entry: " + member.name)
        archive.extractall(out)
    cfg = C.load_config(str(out / args.config_rel))
    remote_data = cfg["paths"]["data_root"]
    cfg["paths"]["data_root"] = str(ROOT / "Dataset/TestDataset")
    cfg["paths"]["out_dir"] = str(out)
    ids = C.read_id_file(str(out / "manifest_p0.txt"))
    rows, problems = C.manifest_rows(cfg["paths"]["data_root"], ["CAMO", "COD10K"])
    if problems:
        raise ValueError(problems)
    selected = C.rows_for_ids(rows, ids)
    original = {r["image_id"]: r for r in C.read_csv(str(out / "INPUT_MANIFEST.csv"))}
    for row in selected:
        image_path, _ = C.rel_id_to_paths(row, cfg)
        if C.sha256_file(image_path) != original[row["image_id"]]["image_sha256"]:
            raise ValueError("Local/server image mismatch: " + row["image_id"])
    C.write_json(str(out / "LOCAL_EVAL_PATHS.json"), {
        "server_data_root": remote_data,
        "local_data_root": cfg["paths"]["data_root"],
        "inference_config_sha256": cfg["_sha256"],
        "image_hashes_matched": len(selected),
        "note": "Only evaluation paths changed; inference predictions unchanged.",
    })
    print("Local images match server SHA256:", len(selected), flush=True)
    code = eval_strict.run(cfg, str(out), ids)
    if code:
        return code
    make_review.run_stage(cfg, str(out), ids)
    print((out / "metrics/metrics_table.md").read_text(encoding="utf-8"))
    print("LOCAL_EVAL_DONE:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
