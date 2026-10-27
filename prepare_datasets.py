"""prepare_datasets.py - Organize COD10K-v2 data into the standard Dataset/ layout.

Merges COD10K-v2 images into the project's data folders so the existing pipeline
(dataloader_sam3 / MyTrain_SAM3) can train on CAMO + COD10K.

Two modes:
  * default (CAM-only) : copies only camouflaged images (Info/CAM_*.txt).
      Train = CAMO(1000) + COD10K-CAM-train(3038) = 4038
      Test  = COD10K-CAM-test(2026)
  * --full             : copies ALL images with a matching GT_Object (file
      intersection, i.e. the complete 6000-train / 4000-test official split).
      Train = CAMO(1000) + COD10K-train-all(6000) = 7000
      Test  = COD10K-test-all(4000)

Source (raw COD10K-v2, unmodified):
  COD10K-v2/COD10K-v2/
    Info/CAM_train.txt, Info/CAM_test.txt   # official CAM file lists
    Info/NonCAM_train.txt, Info/NonCAM_test.txt
    Train/Image + Train/GT_Object
    Test/Image  + Test/GT_Object

Output (what the training/inference code consumes):
  Dataset/TrainValDataset/Imgs + GT
  Dataset/TestDataset/COD10K/Imgs + GT
  Dataset/COD10K_report.json               # data integrity report

Known COD10K-v2 quirks handled automatically (verified 2026-08):
  * COD10K-CAM-2-Terrestrial-23-Cat-1506.jpg is listed in CAM_train.txt but physically
    moved to Test/Image (it IS in CAM_test.txt) -> CAM-only mode uses the 3038 present
    files; --full mode simply takes the physical folder contents.
  * COD10K-CAM-1-Aquatic-3-Crab-39.jpg / COD10K-CAM-3-Flying-59-Grasshopper-3638.jpg
    exist in Train/Image but are in no official list -> included by --full (they have
    GT_Object), skipped by CAM-only mode.

Usage:
  python prepare_datasets.py [--cod10k_root ./COD10K-v2/COD10K-v2]
                             [--train_out ./Dataset/TrainValDataset]
                             [--test_out  ./Dataset/TestDataset/COD10K]
                             [--report    ./Dataset/COD10K_report.json]
                             [--full]                # all 6000 train + 4000 test
                             [--include_noncamo]     # CAM-mode + NONCAM-train
                             [--check_sizes]         # verify img.size == gt.size
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def parse_filelist(path):
    """Read a COD10K Info txt (lines: 'filename.jpg 1') -> set of filenames."""
    if not os.path.exists(path):
        return set()
    names = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.add(line.split()[0])
    return names


def copy_file(src, dst_dir, stats):
    """Copy src into dst_dir, skip if an identical file already exists."""
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst):
        if os.path.getsize(dst) == os.path.getsize(src):
            stats["existing"] += 1
            return dst
        # size mismatch -> overwrite (safer than keeping corrupt file)
        shutil.copy2(src, dst)
        stats["overwritten"] += 1
        return dst
    shutil.copy2(src, dst)
    stats["copied"] += 1
    return dst


def merge_cam_split(img_src, gt_src, cam_list, out_dir, report_key, stats):
    """Copy camouflaged images + GT_Object from a raw COD10K split.

    Returns (copied_imgs, missing) where missing are CAM names without a
    physical image or GT file.
    """
    img_out = os.path.join(out_dir, "Imgs")
    gt_out = os.path.join(out_dir, "GT")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

    copied = 0
    missing = []
    for name in sorted(cam_list):
        base = name[:-4] if name.lower().endswith(".jpg") else os.path.splitext(name)[0]
        img_path = os.path.join(img_src, f"{base}.jpg")
        gt_path = os.path.join(gt_src, f"{base}.png")
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            missing.append(name)
            continue
        copy_file(img_path, img_out, stats)
        copy_file(gt_path, gt_out, stats)
        copied += 1
        if copied % 500 == 0:
            print(f"    ... {copied} images copied")
    report_key["listed"] = len(cam_list)
    report_key["copied"] = copied
    report_key["missing"] = missing
    return copied, missing


def merge_all_split(img_src, gt_src, out_dir, report_key, stats):
    """Copy ALL images that have a matching GT_Object (file intersection).

    Unlike merge_cam_split (which follows the official CAM lists), this takes the
    complete physical folder contents - i.e. CAM + NONCAM. Used by --full.
    Returns (copied, missing).
    """
    img_out = os.path.join(out_dir, "Imgs")
    gt_out = os.path.join(out_dir, "GT")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

    gt_by_name = {os.path.splitext(g)[0] for g in os.listdir(gt_src)}
    img_names = sorted(f for f in os.listdir(img_src)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    copied = 0
    missing = []
    for name in img_names:
        base = os.path.splitext(name)[0]
        img_path = os.path.join(img_src, f"{base}.jpg")
        gt_path = os.path.join(gt_src, f"{base}.png")
        if base not in gt_by_name or not os.path.exists(img_path) or not os.path.exists(gt_path):
            missing.append(name)
            continue
        copy_file(img_path, img_out, stats)
        copy_file(gt_path, gt_out, stats)
        copied += 1
        if copied % 1000 == 0:
            print(f"    ... {copied} images copied")
    report_key["listed"] = len(img_names)
    report_key["copied"] = copied
    report_key["missing"] = missing
    return copied, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cod10k_root", type=str, default="./COD10K-v2/COD10K-v2")
    parser.add_argument("--train_out", type=str, default="./Dataset/TrainValDataset")
    parser.add_argument("--test_out", type=str, default="./Dataset/TestDataset/COD10K")
    parser.add_argument("--report", type=str, default="./Dataset/COD10K_report.json")
    parser.add_argument("--include_noncamo", action="store_true",
                        help="also copy NONCAM images (all-black GT) into train")
    parser.add_argument("--full", action="store_true",
                        help="copy ALL COD10K images (CAM+NONCAM): 6000 train + 4000 test")
    parser.add_argument("--check_sizes", action="store_true",
                        help="open every image+GT in output folders and verify size match")
    opt = parser.parse_args()

    root = Path(opt.cod10k_root)
    info = root / "Info"
    train_img_src = root / "Train" / "Image"
    train_gt_src = root / "Train" / "GT_Object"
    test_img_src = root / "Test" / "Image"
    test_gt_src = root / "Test" / "GT_Object"

    for p in (info, train_img_src, train_gt_src, test_img_src, test_gt_src):
        if not p.is_dir():
            print(f"[FATAL] source path not found: {p}")
            sys.exit(1)

    cam_train = parse_filelist(info / "CAM_train.txt")
    cam_test = parse_filelist(info / "CAM_test.txt")
    noncam_train = parse_filelist(info / "NonCAM_train.txt")
    noncam_test = parse_filelist(info / "NonCAM_test.txt")
    print(f"[Info] CAM_train={len(cam_train)} CAM_test={len(cam_test)} "
          f"NonCAM_train={len(noncam_train)} NonCAM_test={len(noncam_test)}")

    report = {
        "cod10k_root": str(root),
        "mode": "full" if opt.full else "cam-only",
        "cod10k_train_cam": {"listed": len(cam_train)},
        "cod10k_test_cam": {"listed": len(cam_test)},
        "cod10k_train_all": {},
        "cod10k_test_all": {},
    }

    # ---- train: merge COD10K into Dataset/TrainValDataset ----
    print("\n[1/2] Merging COD10K train ({} mode) into {}".format(
        "full" if opt.full else "CAM-only", opt.train_out))
    stats_train = {"copied": 0, "existing": 0, "overwritten": 0}
    if opt.full:
        n_cod10k_train, miss_train = merge_all_split(
            train_img_src, train_gt_src, opt.train_out,
            report["cod10k_train_all"], stats_train)
        print(f"    COD10K train ALL copied: {n_cod10k_train}, "
              f"missing: {len(miss_train)} -> {miss_train[:5]}")
    else:
        n_cod10k_train, miss_train = merge_cam_split(
            train_img_src, train_gt_src, cam_train, opt.train_out,
            report["cod10k_train_cam"], stats_train)
        print(f"    COD10K train CAM copied: {n_cod10k_train}, "
              f"missing: {len(miss_train)} -> {miss_train[:5]}")

        if opt.include_noncamo:
            print("\n[1b] Adding NONCAM-train (empty GT) into", opt.train_out)
            stats_nc = {"copied": 0, "existing": 0, "overwritten": 0}
            report["cod10k_train_noncamo"] = {"listed": len(noncam_train)}
            n_nc, miss_nc = merge_cam_split(
                train_img_src, train_gt_src, noncam_train, opt.train_out,
                report["cod10k_train_noncamo"], stats_nc)
            print(f"    NONCAM train copied: {n_nc}, missing: {len(miss_nc)}")

    # ---- test: COD10K into Dataset/TestDataset/COD10K ----
    print("\n[2/2] Building COD10K test split into", opt.test_out)
    stats_test = {"copied": 0, "existing": 0, "overwritten": 0}
    if opt.full:
        n_cod10k_test, miss_test = merge_all_split(
            test_img_src, test_gt_src, opt.test_out,
            report["cod10k_test_all"], stats_test)
        print(f"    COD10K test ALL copied: {n_cod10k_test}, "
              f"missing: {len(miss_test)} -> {miss_test[:5]}")
    else:
        n_cod10k_test, miss_test = merge_cam_split(
            test_img_src, test_gt_src, cam_test, opt.test_out,
            report["cod10k_test_cam"], stats_test)
        print(f"    COD10K test CAM copied: {n_cod10k_test}, "
              f"missing: {len(miss_test)} -> {miss_test[:5]}")

    # ---- verify existing merged train folder integrity ----
    camo_imgs = Path(opt.train_out) / "Imgs"
    camo_gts = Path(opt.train_out) / "GT"
    n_img = len([f for f in os.listdir(camo_imgs) if f.lower().endswith((".jpg", ".png"))])
    n_gt = len([f for f in os.listdir(camo_gts) if f.lower().endswith((".png", ".jpg"))])
    report["train_total"] = {"imgs": n_img, "gts": n_gt}
    report["test_cod10k_total"] = {
        "imgs": len(os.listdir(Path(opt.test_out) / "Imgs")),
        "gts": len(os.listdir(Path(opt.test_out) / "GT")),
    }
    print(f"\n[Summary] merged train Imgs={n_img} GT={n_gt} | "
          f"COD10K-test Imgs={report['test_cod10k_total']['imgs']}")

    # ---- size consistency check (optional, slower) ----
    if opt.check_sizes:
        print("\n[check_sizes] verifying img.size == gt.size ...")
        bad = []
        for folder, name in ((camo_imgs, "train"), (Path(opt.test_out) / "Imgs", "test")):
            gt_dir = Path(str(folder).replace("Imgs", "GT"))
            imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
            for i, fn in enumerate(imgs, 1):
                stem = os.path.splitext(fn)[0]
                gt_fn = stem + ".png"
                img_p = folder / fn
                gt_p = gt_dir / gt_fn
                if not gt_p.exists():
                    bad.append(f"{name}:{fn} (no GT)")
                    continue
                try:
                    if Image.open(img_p).size != Image.open(gt_p).size:
                        bad.append(f"{name}:{fn} (size mismatch)")
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{name}:{fn} (open failed: {e})")
                if i % 1000 == 0:
                    print(f"    ... {name}: {i} checked")
        report["size_mismatches"] = bad
        print(f"    size mismatches: {len(bad)} -> {bad[:10]}")

    with open(opt.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] report written to {opt.report}")


if __name__ == "__main__":
    main()
