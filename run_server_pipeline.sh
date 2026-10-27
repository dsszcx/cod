#!/bin/bash
# =============================================================================
# run_server_pipeline.sh — 一键跑 SAM3 伪装目标检测全流程
#   （默认行为：先删掉上一次的产出，再重新生成先验并训练）
#
# 用法：
#   bash run_server_pipeline.sh --dry-run --yes              # 只看计划（删什么/跑什么）
#   bash run_server_pipeline.sh --fresh --yes                # 全流程 Step1-5
#   bash run_server_pipeline.sh --step 1,2,3 --gpus 0,1,2 --yes   # 只生成先验（3 卡并行）
#   bash run_server_pipeline.sh --purge_only --yes           # 只清理旧产出
#   bash run_server_pipeline.sh --help                       # 全部开关
#
# 参数原样透传给 run_server_pipeline.py（真正的逻辑都在那个文件里）。
# 环境变量：
#   PYTHON    各步骤使用的解释器（默认 python3；建议写 SAM3 环境的全路径）
#   CX_ROOT   工作根目录（默认 /srv/cod-workspace；也可以直接传 --root）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
ROOT="${CX_ROOT:-/srv/cod-workspace}"

# 用户自己给了 --root 就不再追加
has_root=""
for a in "$@"; do
    [ "$a" = "--root" ] && has_root=1
done

echo ">>> 解释器: $PY"
if [ -z "$has_root" ]; then
    exec "$PY" run_server_pipeline.py --root "$ROOT" "$@"
else
    exec "$PY" run_server_pipeline.py "$@"
fi
