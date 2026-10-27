#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_server_bundle.py - 打服务器部署包（zip，含 MANIFEST 与 sha256）

包内容 = 跑通 Step1-5 所需的最小文件集：
    代码（根目录 .py/.sh）+ lib/ + templates/ + tests/ + docs/ + 依赖清单
    （可选）pretrained/pvt_v2_b2.pth —— 训练用的 ImageNet 预训练骨干

用法::

    python make_server_bundle.py                 # -> releases/server_bundle_sam3_<日期>.zip
    python make_server_bundle.py --list          # 只列清单，不写包
    python make_server_bundle.py --no_weights    # 不带 pvt_v2_b2.pth（服务器已有）
    python make_server_bundle.py --out xx.zip --prefix pkg/    # 包内加一层目录

服务器端解包::

    unzip -o server_bundle_sam3_YYYYMMDD.zip -d /srv/cod-workspace
    # 或（避免中文文件名乱码）
    python3 -m zipfile -e server_bundle_sam3_YYYYMMDD.zip /srv/cod-workspace

不含：Dataset/、sam3-main/、model_pth/、prior*/、results*/、outputs/（数据与产出，
服务器上自己生成或已存在）。
"""
import argparse
import glob as globlib
import hashlib
import os
import sys
import zipfile
from datetime import datetime

# 根目录下要带的文件（不存在则记入 missing）
ROOT_FILES = [
    # 核心流程
    "run_server_pipeline.py", "run_server_pipeline.sh", "run_local_train.sh",
    "llm_annotator.py", "sam3_regions.py", "region_selector.py",
    "MyTrain_SAM3.py", "MyTest_SAM3.py",
    # 网络 / 数据 / 工具
    "Network_SAM3.py", "Network_PVTv2.py", "dataloader.py", "dataloader_sam3.py",
    "utils.py",
    "check_ckpt_channels.py", "prepare_datasets.py",
    "prior_quality_report.py", "eval_metrics.py", "新metric.py",
    # 自检入口（详细回归测试由 tests/ 递归收集）
    "verify_pipeline.py",
    # Web 演示（可选，但不占空间）
    "web_server.py", "web_server_sam3.py", "GUI.py",
    # 老入口（保留，供对照）
    "MyTrain.py", "MyTest.py",
    # 文档 / 依赖 / 打包脚本
    "README.md", "requirements.txt", "requirements-server.txt",
    "make_server_bundle.py",
]

ROOT_GLOBS = ["requirements*.txt"]

# 目录 -> 收集哪些后缀（递归）
DIR_GLOBS = {
    "lib": (".py",),
    "tools": (".py",),
    "tests": (".py",),
    "configs": (".yaml", ".yml"),
    "templates": (".html", ".css", ".js"),
    "docs": (".md", ".txt"),
}

# 可选权重（--no_weights 时跳过）
WEIGHT_FILES = ["pretrained/pvt_v2_b2.pth"]

# 无论如何都不打包
EXCLUDE = {".git", ".venv", "__pycache__", ".idea", "Dataset.zip",
           "cod_improve_server.tar.gz"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n):
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024.0 or unit == "GB":
            return "%.1f %s" % (x, unit)
        x /= 1024.0


def collect(root, with_weights=True):
    keep, missing = [], []

    def add(rel):
        rel = rel.replace(os.sep, "/")
        p = os.path.join(root, rel)
        (keep if os.path.isfile(p) else missing).append(rel)

    for rel in ROOT_FILES:
        add(rel)
    for pat in ROOT_GLOBS:
        for p in globlib.glob(os.path.join(root, pat)):
            add(os.path.relpath(p, root))
    for dname, exts in DIR_GLOBS.items():
        base = os.path.join(root, dname)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                add(os.path.relpath(os.path.join(dirpath, fn), root))
    if with_weights:
        for rel in WEIGHT_FILES:
            add(rel)

    keep = sorted(set(keep))
    keep = [r for r in keep if not any(part in EXCLUDE for part in r.split("/"))]
    return keep, sorted(set(missing))


def build_manifest(entries, with_weights, stamp):
    lines = [
        "SAM3 camouflaged-object-detection 服务器部署包 MANIFEST",
        "生成时间 : %s" % stamp,
        "含权重   : %s" % ("yes (pretrained/pvt_v2_b2.pth)" if with_weights else "no"),
        "条目数   : %d" % len(entries),
        "总字节   : %d" % sum(e[1] for e in entries),
        "",
        "解包: unzip -o <zip> -d /srv/cod-workspace   （或 python3 -m zipfile -e <zip> <dir>）",
        "先看: README.md",
        "自检: cd /srv/cod-workspace && python verify_pipeline.py",
        "再跑: cd /srv/cod-workspace && bash run_server_pipeline.sh --dry-run --yes",
        "再跑: cd /srv/cod-workspace && bash run_server_pipeline.sh --fresh --yes",
        "说明: docs/deployment/服务器部署说明.md",
        "",
        "%-58s %12s  %s" % ("rel_path", "bytes", "sha256"),
        "-" * 100,
    ]
    for rel, size, digest in entries:
        lines.append("%-58s %12d  %s" % (rel, size, digest))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="打 SAM3 服务器部署包")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="源码根目录（默认脚本所在目录）")
    ap.add_argument("--out", default=None,
                    help="输出 zip（默认 releases/server_bundle_sam3_<日期>.zip）")
    ap.add_argument("--prefix", default="", help="包内前缀目录（默认无）")
    ap.add_argument("--no_weights", action="store_true", help="不带 pretrained/pvt_v2_b2.pth")
    ap.add_argument("--list", action="store_true", help="只列清单，不写 zip")
    opt = ap.parse_args(argv)

    root = os.path.abspath(opt.root)
    with_weights = not opt.no_weights
    files, missing = collect(root, with_weights)
    if not files:
        print(">>> 没有收集到任何文件，检查 --root：%s" % root)
        return 1

    entries = [(rel, os.path.getsize(os.path.join(root, rel)),
                sha256_file(os.path.join(root, rel))) for rel in files]

    total = sum(e[1] for e in entries)
    print(">>> 源目录 : %s" % root)
    print(">>> 条目   : %d 个文件, %s" % (len(entries), human(total)))
    for rel, size, _d in entries:
        print("      %-56s %10s" % (rel, human(size)))
    if missing:
        print(">>> 未找到（跳过 %d 个）：%s" % (len(missing), ", ".join(missing)))
    if opt.list:
        print(">>> --list：不写 zip。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d")
    out = os.path.abspath(opt.out or os.path.join(
        root, "releases", "server_bundle_sam3_%s.zip" % stamp))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prefix = opt.prefix.strip("/")
    prefix = (prefix + "/") if prefix else ""

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, _size, _d in entries:
            zf.write(os.path.join(root, rel), prefix + rel)
        zf.writestr(prefix + "MANIFEST-server-bundle.txt",
                    build_manifest(entries, with_weights, stamp))

    digest = sha256_file(out)
    # newline="\n": on Windows the default text mode would write CRLF and
    # `sha256sum -c` on the Linux box then fails on the trailing '\r'
    with open(out + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%s  %s\n" % (digest, os.path.basename(out)))

    print(">>> 打包完成")
    print("    zip      : %s" % out)
    print("    zip 大小 : %s（未压缩 %s）"
          % (human(os.path.getsize(out)), human(total)))
    print("    sha256   : %s" % digest)
    print("    校验文件 : %s.sha256" % out)
    print("    解包     : unzip -o %s -d /srv/cod-workspace" % os.path.basename(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
