#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_server_pipeline.py - SAM3 伪装目标检测：服务器一键全流程（含“先删旧产出”）

    Step 1  Qwen 预筛标注        -> outputs/prior_annotations.json
    Step 2  SAM3 区域生成        -> sam3_regions/{split}_*.npz
    Step 3  目标/辅助区域选择    -> prior_sam3/{split}_*.npz   （默认 qwen 判别）
    Step 4  训练                 -> model_pth/SAM3/*.pth
    Step 5  推理                 -> results_sam3/<run>/<dataset>/*.png

为什么要用这个脚本而不是旧的 run_server_pipeline.sh
---------------------------------------------------
旧版只跑 train split、Step3 写死 heuristic、不清理上一次的产出，也没有任何
覆盖率检查。于是“先验目录不存在 -> 先验恒 0 -> 静默训练出一条死分支”可以毫无
提示地发生（见 docs/experiments/Layer0证据报告.md）。

本脚本做了三件旧版没做的事：
  1. ``--fresh``（默认开）：跑之前先把上一次的产出删干净（删什么逐条打印）；
  2. **train 和 test 两个 split 都生成先验**（训练读 train_*.npz，验证/推理读
     test_*.npz，缺一不可）；
  3. 每一步之后都有覆盖率闸门，不达标直接 exit 2，绝不带着死先验进训练。

用法（服务器）::

    python run_server_pipeline.py --dry-run              # 只打印：会删什么、会跑什么
    python run_server_pipeline.py --purge_only --yes     # 只删旧产出
    python run_server_pipeline.py --fresh --yes          # 删旧产出 + 全流程 Step1-5
    python run_server_pipeline.py --step 1,2,3 --yes     # 只生成先验
    python run_server_pipeline.py --step 1,3 --yes       # Qwen 环境只跑 1/3（2 在另一台）
    python run_server_pipeline.py --fresh --max_images 50 --yes    # 50 张冒烟

注意：``--fresh`` 会删掉 outputs/prior_annotations.json，所以 **分多次执行时第二步
不要再带 --fresh**（脚本会对这种组合给出警告，见 check_fresh_conflict()）。

磁盘提示：全量 7000 训练 + 4250 测试的中间产物约 45 GB（sam3_regions/）+ 95 GB
（prior_sam3/，每张图 4 个 float32 原图尺寸 mask）。跑之前先 df -h 确认空间。
"""
import argparse
import glob as globlib
import json
import os
import shutil
import subprocess
import sys
import time

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --fresh 默认清理清单（相对 --root；支持 * 通配；全部是“上一次跑出来的东西”）
DEFAULT_PURGE = [
    "outputs",              # prior_annotations.json / metrics / prior_quality 报告
    "sam3_regions*",        # Step2 候选区域
    "prior_sam3*",          # Step3 最终先验（含 prior_sam3_0814/ 等历史目录）
    "results_sam3*",        # Step5 预测图
    "model_pth/SAM3/log.log",
    "_pipeline_tmp",        # 本脚本的合并图片视图
]

# 绝不允许被 --fresh 删掉的目录名（防手滑）
PROTECTED_BASENAMES = {"Dataset", "TestDataset", "TrainValDataset", "pretrained",
                       "sam3-main", ".git", ".venv", "lib", "templates"}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def log(msg=""):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def die(msg, code=1):
    log("ERROR: %s" % msg)
    sys.exit(code)


def human(nbytes):
    x = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or unit == "TB":
            return "%.1f %s" % (x, unit)
        x /= 1024.0


def path_stats(path):
    """(文件数, 字节数)；文件或目录（递归）都支持。"""
    if os.path.isfile(path):
        try:
            return 1, os.path.getsize(path)
        except OSError:
            return 1, 0
    n_files = n_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if os.path.isfile(fp):
                n_files += 1
                try:
                    n_bytes += os.path.getsize(fp)
                except OSError:
                    pass
    return n_files, n_bytes


def run(cmd, cwd=None, env_extra=None, dry=False):
    """执行一条流水线命令；先打印完整命令（可复现），非 0 直接退出。"""
    printable = " ".join(('"%s"' % c if " " in c else c) for c in cmd)
    log("RUN  %s" % printable)
    if dry:
        return 0
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=cwd, env=env)
    log("END  rc=%d  (%.1f s)" % (rc, time.time() - t0))
    if rc != 0:
        die("命令失败 rc=%d：%s" % (rc, printable), rc)
    return rc


def run_soft(cmd, cwd=None, dry=False):
    """执行“只用于报告”的命令：rc 非 0 只警告，不中断主流程。"""
    log("RUN* %s" % " ".join(cmd))
    if dry:
        return 0
    rc = subprocess.call(cmd, cwd=cwd, env=os.environ.copy())
    if rc != 0:
        log("WARN rc=%d（不影响主流程）" % rc)
    return rc


def confirm(question, assume_yes, dry):
    if dry:
        log("[dry-run] 需要确认：%s -> 视为 yes" % question)
        return True
    if assume_yes:
        log("%s -> yes (--yes)" % question)
        return True
    if not sys.stdin.isatty():
        die("stdin 不是交互终端；请加 --yes 表示同意「%s」" % question)
    ans = input("%s [y/N] " % question).strip().lower()
    return ans in ("y", "yes")


def count_images(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if os.path.splitext(f)[1].lower() in IMAGE_EXT)


def is_noncam_name(name):
    """COD10K 'NonCAM' 图片判断——单一真相源：dataloader_sam3.is_noncam。

    这类图没有伪装目标（GT 全黑、第 8 通道 neg_flag=1 就是由文件名规则推出来的），
    Step1 不标注它们。这里只在导入不到 dataloader_sam3（缺 torch 的精简环境）时
    才退回同一条本地字符串规则，避免两处规则漂移。
    """
    base = os.path.basename(str(name))
    try:
        from dataloader_sam3 import is_noncam
        return bool(is_noncam(base))
    except Exception:
        return "NonCAM" in base


def skips_noncam(opts, split):
    """当前 split 是否要跳过 NonCAM（与 llm_annotator.batch_annotate 同口径）。"""
    return (opts.noncam_policy != "annotate"
            and (split == "train" or opts.noncam_policy == "skip_all"))


def count_cam_images(d):
    """图片数，排除 COD10K 'NonCAM'。用于 Gate1 的分母。

    NonCAM 图按设计就该是「全 0 先验 + neg_flag=1」，把它们算进分母会让覆盖率
    的及格线被 2960 张注定拿不到标签的图抬高（0.30×7000≈2100 已接近整个 CAM
    子集 4040 的一半），所以分母必须是 CAM 子集。
    """
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if os.path.splitext(f)[1].lower() in IMAGE_EXT
               and not is_noncam_name(f))


def expected_ann_count(opts, split, imgs_dir, skip_noncam):
    """Step1 实际会标注的图片数。口径与 llm_annotator.batch_annotate 一致：
    先按 --max_images 截断，再剔除 NonCAM（截断发生在剔除之前）。"""
    if not os.path.isdir(imgs_dir):
        return 0
    names = sorted(f for f in os.listdir(imgs_dir)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXT)
    if opts.max_images > 0:
        names = names[:opts.max_images]
    if skip_noncam:
        names = [n for n in names if not is_noncam_name(n)]
    return len(names)


def count_npz(d, split):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if f.startswith(split + "_") and f.endswith(".npz"))


def ann_count(ann_path, split):
    if not os.path.isfile(ann_path):
        return 0
    with open(ann_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return len(data.get(split, {}))


def resolve(root, p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def parse_steps(text):
    """'all' | '1,2,3' | '1-3' | '4' -> {'1','2','3',...}"""
    text = str(text).strip()
    if text.lower() == "all":
        return {"1", "2", "3", "4", "5"}
    steps = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            for i in range(int(a), int(b) + 1):
                steps.add(str(i))
        else:
            steps.add(str(int(chunk)))
    bad = steps - {"1", "2", "3", "4", "5"}
    if bad:
        die("无法识别的 --step：%s" % ",".join(sorted(bad)))
    return steps


def check_fresh_conflict(opts):
    """--fresh 与“不从 Step1 开始的局部 step”组合会误删上一轮的中间产物。"""
    if not opts.fresh:
        return
    if opts.steps <= {"4", "5"}:
        die("--fresh 会删掉 prior_sam3/ 和 outputs/，但 --step 里没有 1/2/3；"
            "只想跑训练/推理请改用 --keep。")
    if not ({"1", "2", "3"} <= opts.steps):
        log("WARN: --fresh 与局部 --step（%s）同时使用。如果这是分多次执行的第二步，"
            "上一轮生成的中间产物会被删掉！确认无误可忽略。" % ",".join(sorted(opts.steps)))


def split_imgs(opts, split):
    """train -> 单个目录；test -> CAMO/COD10K 两个目录。"""
    if split == "train":
        return [opts.train_imgs_dir]
    return [os.path.join(opts.test_root, name, "Imgs").replace(os.sep, "/")
            for name in opts.test_dataset_list]


def build_image_view(opts, split, sources, dry):
    """把多个图片目录合并成一个“只扫一层”的目录（优先 symlink，失败退化为复制）。

    llm_annotator / sam3_regions / region_selector 都只扫一层目录，而先验文件名
    里不含数据集名，所以 test 集（CAMO + COD10K）必须先合并再统一处理，否则
    region_selector 的 qwen 模式按 npz 名还原图片路径时会找不到图。
    单来源（train）时直接返回原目录，不做多余拷贝。
    """
    if len(sources) == 1:
        return sources[0]
    view = os.path.join(opts.view_dir, "imgs_%s" % split)
    if dry:
        log("[dry-run] 会合并图片视图：%s <- %s" % (view, ", ".join(sources)))
        return view
    names, plan = {}, []
    for src in sources:
        if not os.path.isdir(src):
            die("图片目录不存在：%s" % src)
        for fn in sorted(os.listdir(src)):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXT:
                continue
            if fn in names:
                die("图片重名：%s 同时出现在 %s 和 %s，合并视图无法区分，请先改名。"
                    % (fn, names[fn], src))
            names[fn] = src
            plan.append((os.path.join(src, fn), os.path.join(view, fn)))
    log("合并图片视图：%s <- %s（%d 个来源 / %d 张图）"
        % (view, ", ".join(sources), len(sources), len(plan)))
    shutil.rmtree(view, ignore_errors=True)
    os.makedirs(view, exist_ok=True)
    n_link = n_copy = 0
    for src_fp, dst_fp in plan:
        if os.path.exists(dst_fp):
            continue
        try:
            os.symlink(os.path.abspath(src_fp), dst_fp)
            n_link += 1
        except OSError:
            shutil.copy2(src_fp, dst_fp)
            n_copy += 1
    log("合并图片视图就绪：symlink=%d copy=%d" % (n_link, n_copy))
    return view


# ---------------------------------------------------------------------------
# --fresh：删除上一次的产出
# ---------------------------------------------------------------------------
def _safe_to_delete(path, root):
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    if rp == rr:
        return False, "就是 --root 本身"
    if not rp.startswith(rr + os.sep):
        return False, "不在 --root 之下"
    if os.path.basename(rp) in PROTECTED_BASENAMES:
        return False, "受保护目录（数据集/权重/代码）"
    return True, ""


def expire_targets(opts, patterns):
    """展开 patterns -> 真实存在的路径（去重，并去掉被更外层目录包含的项）。

    安全规则：
      * 带通配符的模式 **只匹配目录**（否则 ``sam3_regions*`` 会把源码
        ``sam3_regions.py`` 一起匹配进来）；
      * 显式路径（无通配）文件/目录都算；
      * ``_safe_to_delete`` 再兜一层：不许删 --root 本身、不许越界、不许碰数据集/权重/代码。
    """
    found = []
    for pat in patterns:
        full = resolve(opts.root, pat)
        if any(ch in pat for ch in "*?["):
            found.extend(p for p in globlib.glob(full) if os.path.isdir(p))
        elif os.path.exists(full):
            found.append(full)
    keep, seen = [], set()
    for p in sorted(set(found), key=lambda s: (len(s), s)):
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        if any(rp == k or rp.startswith(k + os.sep) for k in seen):
            continue
        ok, why = _safe_to_delete(rp, opts.root)
        if not ok:
            log("跳过（%s）：%s" % (why, rp))
            continue
        seen.add(rp)
        keep.append(p)
    return keep


def purge(opts, patterns=None, tag="fresh"):
    patterns = list(patterns if patterns is not None else opts.purge_patterns)
    targets = expire_targets(opts, patterns)
    log("=" * 74)
    log("[%s] 待删除清单（%d 项；模式：%s）" % (tag, len(targets), ", ".join(patterns)))
    log("=" * 74)
    if not targets:
        log("没有需要删除的旧产出（已是干净的）。")
        return []
    total_files = total_bytes = 0
    for t in targets:
        n, b = path_stats(t)
        total_files += n
        total_bytes += b
        log("  - %-56s %7d files %11s" % (t, n, human(b)))
    log("  合计 %d files / %s" % (total_files, human(total_bytes)))
    if not confirm("确认删除以上 %d 项（%s）？" % (len(targets), human(total_bytes)),
                   opts.yes, opts.dry_run):
        die("用户取消删除。")
    if opts.dry_run:
        log("[dry-run] 未真正删除。")
        return targets
    for t in targets:
        try:
            if os.path.isdir(t) and not os.path.islink(t):
                shutil.rmtree(t)
            else:
                os.remove(t)
            log("  已删除 %s" % t)
        except OSError as exc:
            log("  WARN 删除失败 %s：%r" % (t, exc))
    log("[%s] 清理完成，释放约 %s" % (tag, human(total_bytes)))
    return targets


# ---------------------------------------------------------------------------
# 闸门（Gate）与先验质量扫描
# ---------------------------------------------------------------------------
def gate(opts, label, got, expect, min_ratio, detail=""):
    """覆盖率闸门：不达标直接终止（exit 2），杜绝“静默的死先验”。"""
    if expect <= 0:
        if opts.dry_run:
            log("GATE %-16s 跳过（分母为 0：%s）" % (label, detail))
            return 0.0
        die("%s：分母为 0（%s）——上游步骤没有任何产出。" % (label, detail))
    ratio = got / float(expect)
    log("GATE %-16s %d/%d = %6.2f%%  (要求 >= %.2f%%)  %s"
        % (label, got, expect, 100.0 * ratio, 100.0 * min_ratio, detail))
    if ratio < min_ratio:
        if opts.dry_run:
            log("  [dry-run] 闸门未通过也不中止（dry-run 读的是上一轮的旧文件，"
                "真实执行时会先删掉它们）。")
            return ratio
        if opts.force:
            log("WARN 闸门未通过，但指定了 --force，继续执行。")
            return ratio
        log("GATE FAILED：%s 覆盖率不足 -> 终止（exit 2）。"
            "先修上游，或确认无误后用 --force。" % label)
        sys.exit(2)
    return ratio


def scan_zero_priors(prior_dir, split, every=500):
    """统计 {split}_*.npz 中 target_mask 全 0 的比例（全 0 = 该样本先验是死的）。"""
    import numpy as np          # 只在这里用 numpy，避免无 numpy 的环境直接崩
    files = sorted(f for f in os.listdir(prior_dir)
                   if f.startswith(split + "_") and f.endswith(".npz"))
    n_zero = 0
    for i, fn in enumerate(files, 1):
        try:
            with np.load(os.path.join(prior_dir, fn), allow_pickle=True) as d:
                n_fg = int(np.count_nonzero(d["target_mask"] > 0.5))
            if n_fg == 0:
                n_zero += 1
        except Exception:
            n_zero += 1          # 读不出来 == 用不了
        if every and i % every == 0:
            log("  zero-scan %d/%d, zero=%d" % (i, len(files), n_zero))
    return len(files), n_zero


# ---------------------------------------------------------------------------
# Step 1-5
# ---------------------------------------------------------------------------
def step1_annotate(opts, split, imgs_dir):
    cmd = [opts.python, "llm_annotator.py",
           "--data_root", imgs_dir,
           "--split", split,
           "--mode", opts.ann_mode,
           "--model", opts.qwen_model,
           "--out", opts.ann,
           "--gpus", opts.gpus,
           "--num_runs", str(opts.num_runs),
           "--box_fallback", opts.ann_box_fallback,
           "--noncam_policy", opts.noncam_policy]
    if opts.max_images > 0:
        cmd += ["--max_images", str(opts.max_images)]
    if not opts.resume:
        cmd += ["--no_resume"]
    run(cmd, cwd=opts.root, dry=opts.dry_run)


def step2_regions(opts, split, imgs_dir):
    cmd = [opts.python, "sam3_regions.py",
           "--data_root", imgs_dir,
           "--ann", opts.ann,
           "--split", split,
           "--ckpt", opts.sam3_ckpt,
           "--out", opts.regions,
           "--prompt_mode", opts.prompt_mode]
    if not opts.resume:
        cmd += ["--no_resume"]
    run(cmd, cwd=opts.root, env_extra={"CUDA_VISIBLE_DEVICES": opts.gpus.split(",")[0]},
        dry=opts.dry_run)


def step3_select(opts, split, imgs_dir):
    # qwen 模式必须给 --image_root（region_selector 按 npz 名反推图片路径）
    cmd = [opts.python, "region_selector.py",
           "--regions", opts.regions,
           "--out", opts.prior,
           "--split", split,
           "--mode", opts.sel_mode,
           "--image_root", imgs_dir,
           "--qwen_model", opts.qwen_model,
           "--min_target_ratio", str(opts.min_target_ratio),
           "--conf_thresh", str(opts.conf_thresh),
           "--max_aux", str(opts.max_aux),
           "--max_cand", str(opts.max_cand),
           "--gpus", opts.gpus,
           "--prior_size", str(opts.prior_size)]
    if opts.draw_dir:
        cmd += ["--draw_dir", opts.draw_dir]
    if not opts.resume:
        cmd += ["--no_resume"]
    run(cmd, cwd=opts.root, dry=opts.dry_run)


def step4_train(opts):
    cmd = [opts.python, "MyTrain_SAM3.py",
           "--epoch", str(opts.epoch),
           "--lr", str(opts.lr),
           "--batchsize", str(opts.batch),
           "--trainsize", str(opts.trainsize),
           "--gpus", opts.gpus,
           "--seed", str(opts.seed),
           "--train_root", opts.train_root,
           "--test_root", opts.test_root,
           "--test_datasets", opts.test_datasets,
           "--prior_root", opts.prior,
           "--prior_mode", opts.prior_mode,
           "--workers", str(opts.workers),
           "--aug_mode", opts.aug_mode,
           "--size_rates", opts.size_rates,
           "--save_root", opts.save_root]
    if opts.dp:
        cmd += ["--dp"]
    if opts.amp:
        cmd += ["--amp"]
    if opts.img_filter:
        cmd += ["--img_filter", opts.img_filter]
    if opts.load:
        cmd += ["--load", opts.load]
    if opts.minmax:
        cmd += ["--minmax"]
    if opts.strict_prior:
        cmd += ["--strict_prior"]
    run(cmd, cwd=opts.root, dry=opts.dry_run)


def step5_test(opts):
    pth = opts.pth
    if not pth:
        cands = globlib.glob(os.path.join(resolve(opts.root, opts.save_root), "*.pth"))
        if not cands:
            die("在 %s 下找不到任何 .pth，请用 --pth 指定权重。" % opts.save_root)
        pth = max(cands, key=os.path.getmtime)
        log("自动选择最新权重：%s" % pth)
    if not os.path.isfile(pth):
        die("权重不存在：%s" % pth)

    # Layer-0 守卫：7ch（neg_flag 之前）的老权重在这里就被拦下，而不是崩在 GPU 上
    log("校验权重通道数（要求 8ch = RGB+target+aux x3+neg_flag）")
    run([opts.python, "check_ckpt_channels.py", pth,
         "--expected", "8", "--strict", "--no-probe-networks"],
        cwd=opts.root, dry=opts.dry_run)

    cmd = [opts.python, "MyTest_SAM3.py",
           "--pth_path", pth,
           "--testsize", str(opts.trainsize),
           "--data_root", opts.test_root,
           "--datasets", opts.test_datasets,
           "--prior_root", opts.prior,
           "--prior_mode", opts.prior_mode,
           "--save_root", opts.res_root,
           "--gpu_id", opts.gpus.split(",")[0]]
    if opts.save_name:
        cmd += ["--save_name", opts.save_name]
    if opts.minmax:
        cmd += ["--minmax"]
    if opts.strict_prior:
        cmd += ["--strict_prior"]
    run(cmd, cwd=opts.root, dry=opts.dry_run)


ARCHIVE_CODE = (
    "import glob, os, shutil, sys, torch\n"
    "from check_ckpt_channels import ckpt_first_conv\n"
    "save_root = sys.argv[1]\n"
    "dst = os.path.join(save_root, '_stale_pre8ch')\n"
    "moved = []\n"
    "for p in sorted(glob.glob(os.path.join(save_root, '*.pth'))):\n"
    "    try:\n"
    "        st = torch.load(p, map_location='cpu')\n"
    "    except Exception as e:\n"
    "        print('>>> skip(unreadable): %s (%r)' % (p, e)); continue\n"
    "    _k, shape = ckpt_first_conv(st if isinstance(st, dict) else {})\n"
    "    ch = shape[1] if shape else None\n"
    "    if ch != 8:\n"
    "        os.makedirs(dst, exist_ok=True)\n"
    "        shutil.move(p, os.path.join(dst, os.path.basename(p)))\n"
    "        moved.append((os.path.basename(p), ch))\n"
    "print('>>> archived %d non-8ch checkpoint(s) -> %s' % (len(moved), dst))\n"
    "for nm, ch in moved:\n"
    "    print('    %s (in_chans=%s)' % (nm, ch))\n"
)


def archive_stale_ckpts(opts):
    """把 save_root 下非 8ch 的老权重移到 _stale_pre8ch/，避免被自动挑中。"""
    save_root = resolve(opts.root, opts.save_root)
    log("检查 %s 下的权重通道数（非 8ch 的移到 _stale_pre8ch/）" % save_root)
    if opts.dry_run:
        log("[dry-run] 跳过权重归档")
        return
    cmd = [opts.python, "check_ckpt_channels.py", "--glob",
           os.path.join(globlib.escape(save_root), "*.pth"), "--no-probe-networks"]
    run_soft(cmd, cwd=opts.root)
    rc = subprocess.call([opts.python, "-c", ARCHIVE_CODE, save_root], cwd=opts.root)
    if rc != 0:
        log("WARN 权重归档失败（可能环境里没有 torch）；Step5 仍会逐个权重校验通道。")


def quality_report(opts, split, imgs_dir, gt_dir, tag):
    """可选：为某个 split/数据集生成先验质量报告（覆盖/候选/上界/实际 IoU）。"""
    if not os.path.isdir(gt_dir):
        log("跳过质量报告（GT 目录不存在）：%s" % gt_dir)
        return
    out = os.path.join(opts.root, "outputs", "prior_quality", "%s_%s.json" % (split, tag))
    cmd = [opts.python, "prior_quality_report.py",
           "--regions", opts.regions,
           "--prior", opts.prior,
           "--splits", split,
           "--img_dir", imgs_dir,
           "--gt_dir", gt_dir,
           "--out", out,
           "--min_target_ratio", str(opts.min_target_ratio),
           "--conf_thresh", str(opts.conf_thresh),
           "--max_aux", str(opts.max_aux)]
    run_soft(cmd, cwd=opts.root, dry=opts.dry_run)


# ---------------------------------------------------------------------------
# 预检 / 单个 split 的 Step1-3 / 汇总
# ---------------------------------------------------------------------------
def preflight(opts):
    log("=" * 74)
    log("环境预检")
    log("=" * 74)
    log("root       : %s" % opts.root)
    log("python     : %s" % opts.python)
    log("steps      : %s" % ",".join(sorted(opts.steps)))
    log("splits     : %s" % ",".join(opts.split_list))
    log("fresh      : %s    dry_run: %s" % (opts.fresh, opts.dry_run))
    log("train imgs : %s  (%d images)" % (opts.train_imgs_dir, count_images(opts.train_imgs_dir)))
    for name in opts.test_dataset_list:
        d = os.path.join(opts.test_root, name, "Imgs")
        log("test  imgs : %s  (%d images)" % (d, count_images(d)))
    log("ann json   : %s" % opts.ann)
    log("regions    : %s" % opts.regions)
    log("prior      : %s" % opts.prior)
    log("prompt_mode: %s    sel_mode: %s" % (opts.prompt_mode, opts.sel_mode))
    log("qwen model : %s" % opts.qwen_model)
    log("sam3 ckpt  : %s" % opts.sam3_ckpt)
    log("gpus       : %s    epoch: %d  batch: %d  lr: %s"
        % (opts.gpus, opts.epoch, opts.batch, opts.lr))

    problems = []
    if not os.path.isdir(opts.root):
        problems.append("--root 不存在：%s" % opts.root)
    for fn in ("llm_annotator.py", "sam3_regions.py", "region_selector.py",
               "MyTrain_SAM3.py", "MyTest_SAM3.py", "check_ckpt_channels.py",
               "dataloader_sam3.py", "Network_SAM3.py", "dataloader.py"):
        if not os.path.isfile(os.path.join(opts.root, fn)):
            problems.append("缺少代码文件：%s" % fn)
    if opts.dry_run:
        if problems:
            for p in problems:
                log("  PREFLIGHT(dry-run, 忽略)：%s" % p)
            log("（dry-run 不因为缺文件而中止）")
        return
    if ("1" in opts.steps or "3" in opts.steps) and not os.path.isdir(opts.qwen_model):
        problems.append("Qwen 模型目录不存在：%s" % opts.qwen_model)
    if "2" in opts.steps and not os.path.isfile(opts.sam3_ckpt):
        problems.append("SAM3 权重不存在：%s" % opts.sam3_ckpt)
    if "3" in opts.steps and opts.sel_mode == "qwen" and not os.path.isdir(opts.qwen_model):
        problems.append("Step3 qwen 模式需要 Qwen 模型：%s" % opts.qwen_model)

    # 磁盘空间：全量 sam3_regions/ ≈45 GB + prior_sam3/ ≈95 GB
    if opts.steps & {"2", "3"}:
        need = opts.min_free_gb
        if opts.max_images > 0:
            need = min(10.0, need)
        try:
            free_gb = shutil.disk_usage(opts.root).free / float(1024 ** 3)
            log("磁盘剩余   : %.1f GB（要求 >= %.1f GB%s）"
                % (free_gb, need, "，冒烟模式已放宽" if opts.max_images > 0 else ""))
            if free_gb < need:
                problems.append("磁盘剩余 %.1f GB < 要求的 %.1f GB（--min_free_gb）"
                                % (free_gb, need))
        except OSError as exc:
            log("WARN 无法读取 %s 的磁盘剩余空间：%r" % (opts.root, exc))
    if problems:
        for p in problems:
            log("  PREFLIGHT: %s" % p)
        die("预检失败，共 %d 项（修好后重跑；确认可忽略时用 --force 也救不了预检，"
            "请用 --keep/--dry-run 绕开）。" % len(problems))


def run_split(opts, split, imgs_dir):
    n_img = count_images(imgs_dir)
    n_cam = count_cam_images(imgs_dir)
    skip_ncam = skips_noncam(opts, split)
    log("-" * 74)
    if n_img and n_cam < n_img:
        log("split=%s  图片=%d（其中 NonCAM=%d，%s）  标注分母=%d  目录=%s"
            % (split, n_img, n_img - n_cam,
               "本 split 已跳过" if skip_ncam else "本 split 仍标注",
               n_cam if skip_ncam else n_img, imgs_dir))
    else:
        log("split=%s  图片=%d  目录=%s" % (split, n_img, imgs_dir))
    log("-" * 74)

    if "1" in opts.steps:
        step1_annotate(opts, split, imgs_dir)
        # 分母 = Step1 真正会标注的图（NonCAM 已剔除），否则 Gate1 的及格线会被
        # 2960 张注定无标签的 NonCAM 图抬高。
        expect = expected_ann_count(opts, split, imgs_dir, skip_ncam)
        ratio = gate(opts, "1 标注覆盖", ann_count(opts.ann, split), expect,
                     opts.min_ann_ratio, "prior_annotations.json[%s]" % split)
        if not opts.dry_run and ratio < 0.7:
            log("WARN: %s 只有 %.1f%% 的图片拿到标签（分母已剔除 COD10K NonCAM）；"
                "没标签的图在训练/验证时会用全 0 先验。如果偏差太大，考虑 "
                "--ann_box_fallback whole、--ann_mode legacy，或调低 region_selector 的 "
                "--conf_thresh/--min_target_ratio（真实先验覆盖率以 Gate3/4 为准）。"
                % (split, 100.0 * ratio))

    if "2" in opts.steps:
        step2_regions(opts, split, imgs_dir)
        gate(opts, "2 区域覆盖", count_npz(opts.regions, split), ann_count(opts.ann, split),
             opts.min_region_ratio, "sam3_regions/%s_*.npz" % split)

    if "3" in opts.steps:
        step3_select(opts, split, imgs_dir)
        n_prior = count_npz(opts.prior, split)
        gate(opts, "3 先验覆盖", n_prior, count_npz(opts.regions, split),
             opts.min_prior_ratio, "prior_sam3/%s_*.npz" % split)
        # Gate 3b: image-level coverage.  Gate3 divides npz by npz, so it cannot
        # see an image that never got an annotation/region at all (2026-10: 3
        # CAM images were silently missing from a 4040-image split).
        n_cam = count_cam_images(imgs_dir)
        if n_cam:
            gate(opts, "3b 图片级先验", n_prior, n_cam, 0.99,
                 "prior npz / annotatable images")
        if not opts.skip_zero_scan and not opts.dry_run:
            n_scan, n_zero = scan_zero_priors(opts.prior, split)
            if n_scan == 0:
                die("先验目录里没有 %s_*.npz：%s" % (split, opts.prior))
            ratio = n_zero / float(n_scan)
            log("GATE %-16s 全 0 先验 %d/%d = %5.2f%%  (允许 <= %.2f%%)"
                % ("4 先验非空", n_zero, n_scan, 100 * ratio,
                   100 * opts.max_zero_prior_ratio))
            if ratio > opts.max_zero_prior_ratio and not opts.force:
                log("GATE FAILED：全 0 先验占比过高 -> exit 2 "
                    "（放宽请调大 --max_zero_prior_ratio，或用 --force）")
                sys.exit(2)
            if ratio > 0.30:
                log("WARN: 全 0 先验偏高（本地历史实测 0.21~0.32）；"
                    "这些样本在高斯初始化下等于没有先验分支。"
                    "注意分母是「已生成 npz 的图」：NonCAM 已在 Step1 跳过，"
                    "所以它约等于 CAM 子集（4040 张），阈值比按 7000 张算更严。")


def summary(opts):
    log("=" * 74)
    log("产出汇总（root=%s）" % opts.root)
    log("=" * 74)
    for tag, rel in (("annotations", opts.ann), ("regions", opts.regions),
                     ("prior", opts.prior), ("weights", opts.save_root),
                     ("results", opts.res_root)):
        fp = resolve(opts.root, rel)
        if os.path.exists(fp):
            n, b = path_stats(fp)
            log("  %-12s %-34s %7d files %11s" % (tag, rel, n, human(b)))
        else:
            log("  %-12s %-34s %s" % (tag, rel, "(not created)"))
    for split in opts.split_list:
        log("  npz 计数     %-12s regions=%-6d prior=%-6d"
            % (split, count_npz(opts.regions, split), count_npz(opts.prior, split)))
    log("  ann 计数     train=%d test=%d"
        % (ann_count(opts.ann, "train"), ann_count(opts.ann, "test")))
    log("=" * 74)
    log("下一步（评估）——默认协议（sigmoid x255，不要加 --minmax）：")
    log("  python eval_metrics.py --pred_dir %s<run>/%s --gt_dir %s --name <run>"
        % (opts.res_root.rstrip("/") + "/", opts.test_dataset_list[0],
           os.path.join(opts.test_root, opts.test_dataset_list[0], "GT")))
    log("  # 老预测图（min-max 拉伸过的）才需要加 --minmax")
    if "3" in opts.steps:
        log("  先验质量报告：python prior_quality_report.py --regions %s --prior %s \\"
            % (os.path.basename(opts.regions), os.path.basename(opts.prior)))
        log("      --splits test --gt_dir %s --img_dir %s"
            % (os.path.join(opts.test_root, opts.test_dataset_list[0], "GT"),
               os.path.join(opts.test_root, opts.test_dataset_list[0], "Imgs")))


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def _normalize_argv(argv):
    """允许 kebab-case：--dry-run / --min-target-ratio 等同于 --dry_run / --min_target_ratio。"""
    raw = list(sys.argv[1:] if argv is None else argv)
    out = []
    for tok in raw:
        if tok.startswith("--") and len(tok) > 2:
            head, sep, val = tok.partition("=")
            out.append("--" + head[2:].replace("-", "_") + (sep + val if sep else ""))
        else:
            out.append(tok)
    return out


class _Fmt(argparse.ArgumentDefaultsHelpFormatter):
    """布尔开关不显示 “(default: ...)”（否则 --keep 会显示成默认 True，很误导）。"""

    def _get_help_string(self, action):
        if isinstance(action.default, bool):
            return action.help
        return super()._get_help_string(action)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SAM3 伪装目标检测：服务器一键全流程（先删旧产出，再生成先验并训练）",
        formatter_class=_Fmt)

    g = p.add_argument_group("路径（相对 --root，或绝对路径）")
    g.add_argument("--root", default="/srv/cod-workspace", help="服务器工作根目录")
    g.add_argument("--python", default=sys.executable, help="各步骤使用的 python 解释器")
    g.add_argument("--train_root", default="Dataset/TrainValDataset/", help="训练集根目录")
    g.add_argument("--test_root", default="Dataset/TestDataset/", help="测试集根目录")
    g.add_argument("--test_datasets", default="CAMO,COD10K", help="测试集名，逗号分隔")
    g.add_argument("--train_imgs", default=None, help="训练图片目录（默认 <train_root>Imgs）")
    g.add_argument("--ann", default="outputs/prior_annotations.json", help="Step1 标注 json")
    g.add_argument("--regions", default="sam3_regions", help="Step2 区域 npz 目录")
    g.add_argument("--prior", default="prior_sam3", help="Step3 先验 npz 目录")
    g.add_argument("--save_root", default="model_pth/SAM3/", help="Step4 权重目录")
    g.add_argument("--res_root", default="results_sam3/", help="Step5 预测图目录")
    g.add_argument("--view_dir", default="_pipeline_tmp", help="多目录合并视图的落盘位置")

    g = p.add_argument_group("模型与算力")
    g.add_argument("--qwen_model", default="/opt/models/Qwen2.5-VL-3B-Instruct/",
                   help="Qwen 模型目录")
    g.add_argument("--sam3_ckpt", default="sam3-main/sam3-main/checkpoints/sam3.pt",
                   help="SAM3 权重")
    g.add_argument("--gpus", default="0", help="逗号分隔的 GPU id，如 0,1,2")
    g.add_argument("--num_runs", type=int, default=1, help="Step1 多轮投票次数")

    g = p.add_argument_group("流程开关")
    g.add_argument("--step", default="all", help="all | 1,2,3 | 1-3 | 4 | 5")
    g.add_argument("--splits", default="train,test", help="要生成先验的 split")
    g.add_argument("--fresh", dest="fresh", action="store_true", default=True,
                   help="跑之前先删掉上一次的产出（默认行为；要保留旧产出用 --keep）")
    g.add_argument("--keep", dest="fresh", action="store_false",
                   help="保留旧产出（分多次执行、断点续跑时用）")
    g.add_argument("--purge_only", action="store_true", help="只清理旧产出，不做别的")
    g.add_argument("--purge_extra", action="append", default=[],
                   help="额外要删的路径/通配（可重复，如 --purge-extra 'prior_sam3_0814'）")
    g.add_argument("--purge_keep", action="append", default=[],
                   help="从默认清理清单里排除某一项（可重复，如 --purge-keep outputs）")
    g.add_argument("--archive_stale_ckpt", action="store_true",
                   help="把 model_pth/SAM3/ 下非 8ch 的老权重移到 _stale_pre8ch/")
    g.add_argument("--quality_report", action="store_true",
                   help="Step3 之后生成 outputs/prior_quality/*.json 质量报告")
    g.add_argument("--max_images", type=int, default=0,
                   help=">0 时 Step1/2/3 只处理前 N 张（冒烟用）")
    g.add_argument("--min_free_gb", type=float, default=150.0,
                   help="Step2/3 前的磁盘剩余空间要求（GB）；全量约需 140 GB")

    g = p.add_argument_group("Step 1-3 参数")
    g.add_argument("--ann_mode", default="prescreen", choices=["prescreen", "legacy"])
    g.add_argument("--ann_box_fallback", default="quad", choices=["quad", "whole", "off"],
                   help="Step1 显著性没找到候选框时的兜底：quad=整图+四象限（默认，"
                        "Qwen 只需给框内物体命名）, whole=只用整图框, "
                        "off=旧行为（回落开放式提问，3B 会大量 SKIP 且写出无标签的图）")
    g.add_argument("--noncam_policy", default="skip_train",
                   choices=["skip_train", "skip_all", "annotate"],
                   help="COD10K NonCAM（GT 全黑、无目标、neg_flag=1）："
                        "skip_train=训练集跳过、Gate1 分母=4040（默认）, "
                        "skip_all=两个 split 都跳过, annotate=照旧标注（回滚用）")
    g.add_argument("--prompt_mode", default="box", choices=["box", "text", "hybrid"])
    g.add_argument("--sel_mode", default="qwen", choices=["qwen", "heuristic"],
                   help="Step3 区域选择：默认 qwen（视觉判别），heuristic 为本地启发式")
    g.add_argument("--min_target_ratio", type=float, default=0.005)
    g.add_argument("--conf_thresh", type=float, default=0.6)
    g.add_argument("--max_aux", type=int, default=3)
    g.add_argument("--max_cand", type=int, default=5)
    g.add_argument("--prior_size", type=int, default=448,
                   help="Step3 保存先验的分辨率（uint8 0/1，默认 448=352 的 1.27 倍）。"
                        "0 = 旧行为：原图分辨率 float32 未压缩，8275 张约 78 GB，"
                        "且每张样本都要在 loader 里缩小一次")
    g.add_argument("--draw_dir", default=None, help="qwen 选择时的可视化目录（默认临时目录）")
    g.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Step1-3 断点续跑（默认开）")
    g.add_argument("--no_resume", dest="resume", action="store_false")

    g = p.add_argument_group("Step 4-5 参数")
    g.add_argument("--epoch", type=int, default=80, help="训练轮数（闭区间：--epoch 80 训练 1..80 轮）")
    g.add_argument("--batch", type=int, default=6, help="batch size")
    g.add_argument("--lr", type=float, default=2e-4, help="AdamW 学习率")
    g.add_argument("--trainsize", type=int, default=352, help="输入尺寸")
    g.add_argument("--seed", type=int, default=0, help="全局随机种子")
    g.add_argument("--img_filter", default=None, help="只保留文件名含该子串的训练图")
    g.add_argument("--load", default=None, help="从权重续训（8ch checkpoint）")
    g.add_argument("--pth", default=None, help="Step5 用的权重（默认取 save_root 下最新）")
    g.add_argument("--save_name", default=None, help="Step5 结果子目录名")
    g.add_argument("--prior_mode", default="real", choices=["real", "zero", "oracle", "shuffle"],
                   help="消融开关：real=真实先验（默认）")
    g.add_argument("--minmax", action="store_true", help="历史非标准 min-max 拉伸（默认关）")
    g.add_argument("--strict_prior", action="store_true",
                   help="任何一张图缺真实先验就中止")
    g.add_argument("--workers", type=int, default=8,
                   help="Step4 DataLoader 进程数（0=主进程加载，用于排查卡死）")
    g.add_argument("--aug_mode", default="fast", choices=["fast", "legacy"],
                   help="Step4 数据增强：fast=先缩到 1.25x 再增强（默认），"
                        "legacy=原图分辨率增强（2026-10-17 之前的旧行为）")
    g.add_argument("--dp", action="store_true",
                   help="Step4 开启 DataParallel（默认关：实测 2 卡 19.06 s/step "
                        "vs 单卡 0.271 s/step，约 70x 慢）")
    g.add_argument("--size_rates", default="0.75,1",
                   help="Step4 每步的缩放序列，如 '0.75,1'（2 次前向）或 '1'（最快）")
    g.add_argument("--amp", action="store_true",
                   help="Step4 用 bf16 autocast 跑前向（损失仍 fp32，无需 GradScaler）")

    g = p.add_argument_group("闸门 / 执行控制")
    g.add_argument("--min_ann_ratio", type=float, default=0.30,
                   help="Gate1：标注条数 / 图片数（Qwen 在伪装图上召回有限，"
                        "默认只要 30%%，低于 70%% 会额外告警）")
    g.add_argument("--min_region_ratio", type=float, default=0.99,
                   help="Gate2：区域 npz / 标注条数")
    g.add_argument("--min_prior_ratio", type=float, default=0.99,
                   help="Gate3：先验 npz / 区域 npz")
    g.add_argument("--max_zero_prior_ratio", type=float, default=0.50,
                   help="Gate4：target_mask 全 0 的先验占比上限（本地实测 0.21~0.32，"
                        "超过 0.30 会警告）")
    g.add_argument("--skip_zero_scan", action="store_true", help="跳过 Gate4 全零扫描")
    g.add_argument("--force", action="store_true", help="闸门不达标也继续（不推荐）")
    g.add_argument("--yes", action="store_true", help="不交互确认（nohup 必须加）")
    g.add_argument("--dry_run", action="store_true",
                   help="只打印计划，不删不跑（也可写作 --dry-run）")
    return p.parse_args(_normalize_argv(argv))


def _abs_dir(p, root):
    """绝对路径 + 结尾斜杠（MyTrain_SAM3 用 root + "KKK_best_x.pth" 拼接）。"""
    p = resolve(root, p)
    if not p.endswith(("/", os.sep)):
        p += "/"
    return p.replace(os.sep, "/")


def main(argv=None):
    opts = parse_args(argv)
    opts.steps = parse_steps(opts.step)
    opts.split_list = [s.strip() for s in opts.splits.split(",") if s.strip()]
    for s in opts.split_list:
        if s not in ("train", "test"):
            die("--splits 只支持 train,test：%s" % s)
    opts.test_dataset_list = [s.strip() for s in opts.test_datasets.split(",") if s.strip()]
    if not opts.test_dataset_list:
        die("--test_datasets 不能为空")

    # 所有路径统一成绝对路径：闸门在同进程里读文件，不能依赖 cwd
    opts.train_root = _abs_dir(opts.train_root, opts.root)
    opts.test_root = _abs_dir(opts.test_root, opts.root)
    opts.save_root = _abs_dir(opts.save_root, opts.root)
    opts.res_root = resolve(opts.root, opts.res_root).replace(os.sep, "/")
    opts.ann = resolve(opts.root, opts.ann).replace(os.sep, "/")
    opts.regions = resolve(opts.root, opts.regions).replace(os.sep, "/")
    opts.prior = resolve(opts.root, opts.prior).replace(os.sep, "/")
    opts.view_dir = resolve(opts.root, opts.view_dir).replace(os.sep, "/")
    opts.qwen_model = _abs_dir(opts.qwen_model, opts.root) if not os.path.exists(opts.qwen_model) \
        else resolve(opts.root, opts.qwen_model)
    opts.sam3_ckpt = resolve(opts.root, opts.sam3_ckpt)
    opts.train_imgs_dir = resolve(opts.root, opts.train_imgs or (opts.train_root + "Imgs")).replace(
        os.sep, "/")
    for name in ("pth", "load", "draw_dir"):
        val = getattr(opts, name)
        if val:
            setattr(opts, name, resolve(opts.root, val))
    opts.purge_patterns = [p for p in DEFAULT_PURGE if p not in opts.purge_keep] \
        + list(opts.purge_extra)

    check_fresh_conflict(opts)
    preflight(opts)

    if opts.purge_only:
        purge(opts)
        log("[purge_only] 结束（未执行任何步骤）。")
        return 0

    t0 = time.time()
    log("=" * 74)
    log("开始执行：root=%s steps=%s splits=%s fresh=%s"
        % (opts.root, ",".join(sorted(opts.steps)), ",".join(opts.split_list), opts.fresh))
    log("=" * 74)

    if opts.fresh:
        purge(opts)
    if opts.archive_stale_ckpt:
        archive_stale_ckpts(opts)

    if opts.steps & {"1", "2", "3"}:
        for split in opts.split_list:
            sources = split_imgs(opts, split)
            if not opts.dry_run:
                for s in sources:
                    if not os.path.isdir(s):
                        die("图片目录不存在：%s" % s)
            imgs_dir = build_image_view(opts, split, sources, opts.dry_run)
            run_split(opts, split, imgs_dir)

    if "4" in opts.steps:
        step4_train(opts)
    if "5" in opts.steps:
        step5_test(opts)

    if opts.quality_report and (opts.steps & {"2", "3"}):
        quality_report(opts, "train", opts.train_imgs_dir, os.path.join(opts.train_root, "GT"),
                       "train")
        for name in opts.test_dataset_list:
            quality_report(opts, "test", os.path.join(opts.test_root, name, "Imgs"),
                           os.path.join(opts.test_root, name, "GT"), name)

    summary(opts)
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
