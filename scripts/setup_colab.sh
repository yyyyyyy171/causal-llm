#!/usr/bin/env bash
# Colab / Linux 环境一键搭建（可复现、固定版本）。
#
# 为什么做成脚本而不是 Notebook 里几行命令：
#   1. Colab 每次会话都是新虚拟机，环境必须能一条命令重建；
#   2. 版本必须固定，否则会撞上"torch 半新半旧""cu130 与驱动不匹配"这类问题；
#   3. 出问题时只需重跑这一个脚本，不用回头翻 Notebook 的单元格。
#
# 用法（在 Colab 里）：
#   !bash scripts/setup_colab.sh
#   !bash scripts/setup_colab.sh --vllm 0.31.0 --force-torch-cu121
#
# 环境变量：VENV 指定虚拟环境路径，默认 /content/venv

set -euo pipefail

# Kaggle 的基础镜像会把 PYTHONPATH 指向系统 site-packages，这个变量会渗进
# 我们新建的隔离环境里，有两个后果：
#   1) 启动时 sitecustomize 报 "No module named 'wrapt'"（只是噪音，非致命）；
#   2) 更要紧的是 PYTHONPATH 在 sys.path 里排在 venv 的 site-packages 之前，
#      于是 venv 里实际可能用到系统的 numpy / torch，而不是刚装的那些 ——
#      我们固定版本的整个意义就没了（踩过一次 torch 版本对不上的坑）。
# 所以这里直接清掉，让虚拟环境真的隔离。
unset PYTHONPATH

VENV="${VENV:-/content/venv}"
VLLM_VERSION="0.31.0"
FORCE_CU121=0

while [ $# -gt 0 ]; do
    case "$1" in
        --vllm) VLLM_VERSION="$2"; shift 2 ;;
        --force-torch-cu121) FORCE_CU121=1; shift ;;
        *) echo "未知参数: $1"; exit 2 ;;
    esac
done

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "===== 1) 显卡与驱动 ====="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)"
echo "驱动主版本: $DRIVER_MAJOR"

echo
echo "===== 2) 建立隔离虚拟环境 ====="
if [ ! -x "$PY" ]; then
    # 上一次建到一半失败会留下一个"看起来存在、其实不可用"的目录，
    # 直接重建会撞上各种奇怪错误，所以先清掉。
    # 删之前先确认这个路径是我们自己的虚拟环境目录，不是什么奇怪的值。
    case "$VENV" in
        ""|"/"|"/content"|"/content/")
            echo "VENV 的取值不安全，拒绝删除：'$VENV'"
            exit 2
            ;;
    esac
    if [ -d "$VENV" ]; then
        echo "$VENV 存在但不是可用的虚拟环境（上次可能建到一半失败），先删掉重建"
        rm -rf -- "$VENV"
    fi
    pip install -q -U virtualenv
    python -m virtualenv -q "$VENV"
    echo "已创建 $VENV"
else
    echo "$VENV 已存在，复用"
fi
"$PIP" install -q -U pip

echo
echo "===== 3) 安装 vLLM（固定 $VLLM_VERSION）====="
"$PIP" install -q -U "vllm==$VLLM_VERSION"

# CUDA 13 的 torch 需要驱动 >= 580；Colab 的 T4 实例常常低于这个数，
# 这时 torch 初始化 CUDA 会失败（表现为 vLLM "Engine core initialization failed"）。
if [ "$FORCE_CU121" = "1" ] || [ "$DRIVER_MAJOR" -lt 580 ]; then
    echo
    echo "===== 4) 驱动 $DRIVER_MAJOR < 580（或强制指定）：换成 CUDA 12.1 版 torch ====="
    "$PIP" install -q -U torch --index-url https://download.pytorch.org/whl/cu121
fi

echo
echo "===== 5) 验证 ====="
"$PY" - <<'PYCODE'
import torch
print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    # 真的分配一次显存，光看 is_available 不能暴露驱动不匹配
    print(f"tensor test: {torch.zeros(3, 3, device='cuda').sum().item()}")
import vllm
print(f"vLLM {vllm.__version__}")
PYCODE

echo
echo "环境就绪。后续用这个解释器运行脚本："
echo "  $PY scripts/run_llm_prior_vllm.py --network ASIA --mode both --task direction --model Qwen/Qwen3-8B-AWQ"
