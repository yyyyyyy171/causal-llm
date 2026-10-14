#!/usr/bin/env bash
# 在 Colab / Linux 上装 R 与 bnlearn，用来跑融合实验这类纯 CPU 的长任务。
#
# 为什么需要它：融合实验的代价随变量数增长很快 —— 每个重复都要对每个目标变量
# 各跑一遍 IAMB 与 HITON-MB，HAILFINDER 有 56 个变量，全量跑一次是几小时。
# 本机能跑，但笔记本不能休眠；放到 Colab 上跑更省心。
#
# 但要清楚一点：Colab 免费实例是 2 核，而 run_fusion_both.R 是单线程的，
# 所以**它不会比本机快多少**。它买到的是"不占自己的机器"，不是速度。
# 真正省时间的开关是 --max-targets：只对固定种子抽出的目标子集求马尔可夫毯。
#
# 用法（在 Colab 里）：
#   !bash scripts/setup_colab_r.sh
#   !cd /content/Causal-LLM && CAUSAL_LLM_ROOT=/content/Causal-LLM \
#       Rscript --vanilla scripts/run_fusion_both.R --networks=ALARM --sizes=100 \
#       --repeats=30 --mode=real --model-tag=qwen3-8b-awq --tag=8b_real --max-targets=15

set -euo pipefail

echo "===== 1) 安装 R ====="
if command -v Rscript >/dev/null 2>&1; then
    echo "已存在: $(Rscript --version 2>&1 | head -1)"
else
    export DEBIAN_FRONTEND=noninteractive
    apt-get -qq update
    apt-get -qq install -y r-base-core >/dev/null
    echo "已安装: $(Rscript --version 2>&1 | head -1)"
fi

echo
echo "===== 2) 安装 bnlearn（源码编译，约 3-6 分钟）====="
Rscript -e '
if (!requireNamespace("bnlearn", quietly = TRUE)) {
  install.packages("bnlearn", repos = "https://cloud.r-project.org")
}
cat("bnlearn", as.character(packageVersion("bnlearn")), "OK\n")
'

echo
echo "===== 3) 冒烟测试：确认真的能算，而不只是装上了 ====="
Rscript --vanilla -e '
suppressPackageStartupMessages(library(bnlearn))
data(learning.test)
cat("hc() 学到", length(arcs(hc(learning.test))), "条边\n")
cat("iamb 给 A 找到", length(learn.mb(learning.test, node = "A", method = "iamb")), "个变量\n")
'

echo
echo "环境就绪。跑融合实验时记得带上仓库根目录："
echo "  CAUSAL_LLM_ROOT=/content/Causal-LLM Rscript --vanilla scripts/run_fusion_both.R ..."
