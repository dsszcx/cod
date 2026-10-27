"""Package the current inference code and baseline for the server trial."""
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from make_server_bundle import collect


def main():
    server_root = "/srv/cod-project"
    cfg = yaml.safe_load((ROOT / "configs/e0e4_v1.yaml").read_text(encoding="utf-8"))
    cfg["version"] = "baseline_mask_prompt_20261022_v1"
    cfg["paths"]["data_root"] = "/srv/cod-workspace/Dataset/TestDataset"
    cfg["paths"]["sam3_ckpt"] = "/srv/cod-workspace/sam3-main/sam3-main/checkpoints/sam3.pt"
    cfg["paths"]["out_dir"] = "outputs/MASK_PROMPT_P0_20261022"
    files, missing = collect(str(ROOT), with_weights=False)
    if missing:
        raise FileNotFoundError(missing)
    files += ["PROJECT_STRUCTURE.md", "model_pth/OUR/Net_epoch_40.pth",
              "tools/server/mask_prompt_part1.sh"]
    out = ROOT / "releases/mask_prompt_trial_20261022.zip"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for rel in sorted(set(files)):
            if rel.endswith(".sh"):
                archive.writestr(rel, (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n"))
            else:
                archive.write(ROOT / rel, rel)
        archive.writestr("configs/mask_prompt_20261022.yaml",
                         yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print("archive:", out)
    print("bytes:", out.stat().st_size)
    print("server_root:", server_root)


if __name__ == "__main__":
    main()
