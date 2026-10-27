"""Segment the revised Qwen captions, reusing fixed E0/E1 from P0."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C
from tools.e0e4 import fuse_e4
from tools.e0e4.run_e0e4 import stage_sam3


def main():
    source = ROOT / "outputs/MASK_PROMPT_P0_20261022"
    out = ROOT / "outputs/QWEN_NO_EXAMPLE_20261022"
    cfg = C.load_config(str(out / "CONFIG.yaml"))
    ids = C.read_id_file(str(out / "manifest_p0.txt"))
    # Reuse exactly the original baseline/box outputs, without new inference.
    for stage in ("E0", "E1"):
        shutil.copytree(source / "predictions" / stage, out / "predictions" / stage)
    shutil.copytree(source / "e0", out / "e0",
                    ignore=shutil.ignore_patterns("*.npy", "M0_soft.png"))
    shutil.copytree(source / "sam3/e1", out / "sam3/e1")
    shutil.copy2(source / "sam3/SAM3_e1_AUDIT.json", out / "sam3/SAM3_e1_AUDIT.json")
    for name in ("E0_INDEX.csv", "E0_RUN.json"):
        shutil.copy2(source / name, out / name)
    C.write_json(str(out / "REUSED_OUTPUTS.json"), {
        "source": str(source), "reused_stages": ["E0", "E1", "ROI"],
        "source_config_sha256": C.read_json(str(source / "E0_RUN.json"))["config"]["sha256"],
        "new_config_sha256": cfg["_sha256"],
        "note": "E1 mask references retain original paths; they are read unchanged by E4.",
    })
    print("Reused E0/E1/ROI. Running only revised E2/E3 text branches.", flush=True)
    stage_sam3(cfg, str(out), ids, ["e2", "e3"], limit=None, overwrite=False)
    fuse_e4.run_stage(cfg, str(out), ids)
    print("QWEN_V2_SAM3_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
