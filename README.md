# Causal-LLM：语言模型先验引导的因果特征选择

选题二的工作仓库。目标是用本地开源语言模型提供**语义因果先验**，
引导马尔可夫毯发现，在小样本条件下提高因果特征选择的恢复质量。

## 目录结构

```
configs/          实验配置（随机种子、样本量、阈值、先验强度都在这里）
data/<NET>/      5 个 bnlearn 网络的采样数据、真值马尔可夫毯、语言模型先验
docs/            选题要求、运行手册、使用约定、实验日志
logs/            每次运行的日志（不进版本库，只保留目录）
results/         结果表与图，全部由脚本生成
R/               R 源码：HITON-PC / HITON-MB 的移植实现（bnlearn 5.2.1 已移除该算法）
scripts/         可执行脚本，以及一键复现入口 reproduce_all.ps1
src/             llm_prior（语言模型先验）、mb（马尔可夫毯发现）、fs（特征选择工具箱）、common（配置与日志）
```

## 环境

### 在 Google Colab（免费 T4）上跑语言模型

本机是 Windows + 6GB 显卡：vLLM 没有 Windows 发行版，8B 长跑又会因显存分页退化。
Colab 的免费 T4（16GB）官方支持 vLLM，是跑大网络先验的推荐路径。

**仓库通过 git 同步，不打压缩包**——每次改动只需 Colab 里 `git pull`，
结果与缓存也随提交回到仓库，导师和同组同学都能看到。

一次性准备（本机执行）：

```powershell
cd E:\Causal-LLM
git remote add origin https://github.com/<用户名>/causal-llm.git
git push -u origin yyyyyyy171/topic2
```

然后在 Colab 里：

1. 左侧密钥面板配 `HF_TOKEN`（可选，避免下载限速）与 `GIT_TOKEN`（写权限，用于回传结果）；
2. 打开 `notebooks/colab_prior.ipynb`，运行时类型选 **T4 GPU**；
3. 改第 1 格的仓库地址与分支，然后从上到下运行。

环境由 `scripts/setup_colab.sh` 一键搭建，它做三件事：建隔离虚拟环境（避开 Colab
预装包冲突）、固定 vLLM 版本、**在驱动低于 580 时自动把 torch 换成 CUDA 12.1 版**
（CUDA 13 的 torch 需要驱动 ≥580，Colab 的 T4 常常达不到，这是 vLLM 报
`Engine core initialization failed` 的常见原因）。

推理用的是 `scripts/run_llm_prior_vllm.py`：缓存格式、断点续跑、聚合方式与本机的
transformers 版完全一致，两条路径的产物可以互相续跑。

离线备选：没有 git 访问条件时，用 `scripts/make_transfer_bundle.ps1` 打一个
约 1.5MB 的 zip，再把 notebook 第 3 格换成手动解压（脚本里已兼容两种布局）。

两个 conda 环境分工明确，不要混用：

| 环境 | 用途 | 关键包 |
|---|---|---|
| `D:\Anaconda`（base） | CPU 分析：马尔可夫毯发现、融合实验、评测、画图 | numpy、scipy、pandas、matplotlib |
| `D:\Anaconda\envs\llm_causal_fs` | GPU 推理：加载本地 Qwen 模型 | torch 2.5.1+cu121、transformers、bitsandbytes |

语言模型缓存位于 `D:\hf-cache`（环境变量 `HF_HOME`），
推理脚本默认 `local_files_only=True`，完全离线运行。

### 工具与版本（重要）

| 工具 | 位置 | 用途 |
|---|---|---|
| R 4.6.1 + bnlearn 5.2.1 | `D:\R\R-4.6.1` | IAMB 系列基线与先验融合实验（`scripts/*_bnlearn.R`） |
| MATLAB R2023a | `C:\Program Files\MATLAB\R2023a` | HITON-MB 基线（Causal-Learner 工具箱，`run_HITONMB_4networks.m`） |

**为什么 HITON-MB 用 MATLAB 而不是 bnlearn**：bnlearn 5.2.1 已经移除 `hiton.mb()`，
`learn.mb()` 只支持 gs / iamb / fast.iamb / inter.iamb / iamb.fdr；
`si.hiton.pc()` 是学父子集的、且改走新的 cluster 接口，不是马尔可夫毯学习器。
本机没有 Rtools，无法从源码编译带 `hiton.mb` 的旧版 bnlearn。
因此 IAMB 走 bnlearn（参考实现），HITON-MB 继续用 Causal-Learner 的结果。

**bnlearn 的自定义条件独立检验**（先验融合的关键）：5.2.1 不再接受直接传函数，
必须写成 `test = "custom-test"` + `fun = f`，其中
`f <- function(x, y, z, data, args)` 且**必须返回两个标量** `c(统计量, p 值)`。
验证方式：把 p 值原样返回的包装函数应当与基线结果完全一致（已确认）。

## 一键复现

不需要显卡、不需要语言模型推理，一条命令重建 `results/` 下的表：

```powershell
cd E:\Causal-LLM
powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1
```

加 `-SkipSlow` 跳过 20–40 分钟的融合实验，加 `-Only fs` 只跑某一步。
语言模型先验需要 GPU，属于前置条件：`data/<网络>/` 下没有先验文件时，
依赖它的步骤会跳过并提示。

每一步与产出的对应关系：

| 步骤 | 脚本 | 产出 | 耗时 |
|---|---|---|---|
| 1 | `scripts/export_dags.py` | `data/<网络>/<网络>_dag_edges.csv`（含道德图一致性交叉验证） | 秒级 |
| 2 | `scripts/verify_dag_bnlearn.R` | 控制台输出，bnlearn 与 Python 解析的边集应全为 `identical=TRUE` | 秒级 |
| 3 | `scripts/run_mb_baseline_bnlearn.R` | `results/bnlearn_mb_summary.csv`（IAMB，5 网络×4 样本量×两种检验） | 秒级 |
| 4 | `scripts/run_hiton_baseline_bnlearn.R` | `results/hiton_baseline_hiton.csv`、`hiton_baseline_vs_matlab_hiton.csv` | 约 1 分钟 |
| 5 | `scripts/run_fs_benchmark.py` | `results/fs_method_by_dataset.csv` 等三张表 | 约 105 秒 |
| 6 | `scripts/eval_llm_prior.py`、`scripts/eval_mb_prior.R` | `results/llm_prior_eval.csv`、`mb_prior_quality.csv` | 秒级 |
| 7 | `scripts/diagnose_prior_vs_spouses.R` | `results/prior_vs_spouses_real.csv` | 秒级 |
| 8 | `scripts/run_fusion_both.R` | `results/fusion_both_summary_both_nreal.csv`、`..._paired_...csv` | 20–40 分钟 |
| 9 | `scripts/make_week_report.py` | `E:\汇报\周产出\*.docx`（7 份周产出文档） | 秒级 |

## 快速开始（单项）

```powershell
cd E:\Causal-LLM

# 1) 语言模型先验（GPU，用 llm_causal_fs 环境）
& 'D:\Anaconda\envs\llm_causal_fs\python.exe' scripts\run_llm_prior.py `
    --network CHILD --mode both

# 2) 先验质量评测（CPU）
python scripts\eval_llm_prior.py --networks ASIA,ALARM --modes real,anon --out llm_prior_eval.csv

# 3) 小样本融合实验（CPU）
python scripts\run_fusion_experiment.py --networks ASIA,CHILD --mode real

# 4) 生成周报
& 'C:\Users\yuhan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
    scripts\make_week_report.py --spec scripts\week_reports\2026.10.16.json

# 5) 共用特征选择工具箱（周 2 产出，纯 CPU，约 105 秒）
python scripts\run_fs_benchmark.py
```

一体化的批量先验推理（建议夜间跑）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all_priors.ps1 -Networks CHILD,INSURANCE
```

## 实验规范

1. **配置文件优先。** 会变的参数一律放 `configs/`，命令行只做覆盖。
   `configs/llm_prior.json` 管推理，`configs/mb_discovery.json` 管马尔可夫毯实验。
2. **随机种子固定。** 推理的种子 = `base_seed + 批次起点`；
   融合实验的抽样种子 = `seed + 重复编号`。同一配置必然复现同一结果。
3. **日志落盘。** 脚本开头调用 `src/common/run_log.py`，
   每次运行在 `logs/` 下生成一个日志，头部记录命令行、配置、
   git 提交、Python 与显卡信息，结尾记录耗时与退出码。
4. **结果表由脚本生成。** `results/` 下任何一张表都能用一条命令重跑出来，
   不要手工改 CSV。
   试跑（调参数、验证代码能不能跑）一律加 `--tag scratch-xxx`，
   这类文件被 `.gitignore` 排除；只有正式实验的结果才用正式 tag 并进版本库。
5. **语言模型输入输出全部缓存。** 同一输入不重复计算，
   `data/<NET>/llm_cache/` 是断点续跑用的缓存，
   最终聚合结果落在 `data/<NET>/<NET>_LLM_prior_<mode>.json`（含每次采样的原始输出）。
   改动 Prompt 后必须把 `src/llm_prior/prompt.py` 里的 `PROMPT_VERSION` 加一，
   旧缓存会自动失效。

### 编码约定（踩过的坑）

仓库里的文本文件一律 UTF-8。**但 `.ps1` 脚本必须带 UTF-8 BOM**：
Windows PowerShell 5.1 在没有 BOM 时按 GBK 解码，中文注释和字符串会直接导致
「表达式或语句中包含意外的标记」这类语法错误（PowerShell 7 默认 UTF-8，不会有这个问题，
所以在 PS 7 里测过不代表在 PS 5.1 里能跑）。

新增 `.ps1` 后这样补 BOM：

```powershell
$path = (Resolve-Path 'scripts\xxx.ps1').Path
$text = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($true)))
```

## 主要发现速览

（每条都给出可复核的结果文件，细节见 `docs/实验日志.md` 与各周产出文档。）

**实现与真值的可信度（先把尺子校准）**

* DAG 真值有三条独立来源互相印证：Python 解析 `rds`、bnlearn 独立读取、仓库既有的真值马尔可夫毯矩阵做道德图交叉验证，5 个网络全部一致 → `data/<网络>/<网络>_dag_edges.csv`、`scripts/verify_dag_bnlearn.R`。
* IAMB 用 bnlearn 参考实现复跑，20 个网络×样本量组合里 14 个与 MATLAB 逐位相同，其余差异集中在 n=100 → `results/bnlearn_mb_summary.csv`。
* HITON-MB 在 bnlearn 5.2.1 中已被移除，按 MATLAB 参考移植到 R，与本仓库基线平均 |ΔF1| = 0.0037 → `results/hiton_baseline_vs_matlab_hiton.csv`。

**语言模型的因果结构知识（这是本选题的核心观察）**

* 方向判断：ASIA 上 7/7 全对，CHILD 上只有 13/22 = 0.591——网络一大就明显下降 → `results/llm_prior_eval.csv`。
* 邻接判断偏松：ASIA 精确率 0.400、CHILD 0.180（召回 0.88–1.00），它把只有间接通路的变量对也判成直接相连。
* 匿名后模型对全部变量对都判"没有直接边"（ASIA 28/28、CHILD 190/190），说明判断确实依赖语义而非编号。
* **配偶错配**：马尔可夫毯 = 父子 + 配偶，而配偶不是邻居；模型对配偶的判断在两个网络上完全相反（ASIA 0/2 判成有边、CHILD 5/5 判成有边）→ `results/prior_vs_spouses_real.csv`。这是"用邻接先验做毯发现"效果不佳的直接原因。

**融合实验（周 6–7 / 8–10 的产出）**

* 两种算法、两个网络、两个样本量、两种度量、30 次重复、配对检验下，**真实先验都没有显著超过基线**，差值全在 ±0.012 内 → `results/fusion_both_paired_both_nreal.csv`。
* 更关键的反证：ASIA n=100 上**随机先验**显著优于基线（固定毯大小 +0.029，p=0.0038），而真实先验是 −0.012。任何声称的提升都必须先过随机对照这一关。
* 方法层面确实有收获：先验得分平均 +0.358 会系统性放宽阈值，零均值化后稳健性明显改善（β=2 时未零均值化掉到 0.5346、低于基线，零均值化后仍有 0.5696）→ `results/bnlearn_fusion_summary_centered.csv`。

**共用特征选择工具箱（周 2 产出）**

* 10 个特征常常够用甚至更好：credit-g 上 L1 用 10 个特征 0.7600 高于全部 61 个特征的 0.7400。
* 方法排序随数据集变化，没有哪一个方法在所有数据集上占优 → `results/fs_method_by_dataset.csv`。

## 当前进度

| 周 | 状态 |
|---|---|
| 1 | 完成。README、配置文件、日志落盘、使用约定、验收清单；vLLM 在原生 Windows 不可用（已记录） |
| 2 | 完成。共用特征选择工具箱 `src/fs/`，5 个 OpenML 数据集 × 5 种方法的结果表 |
| 4 | 完成。5 个网络的样本量–F1 曲线（`results/BN_baseline_summary.csv`） |
| 5 | 管道完成；ASIA 已跑完真实名与匿名，其余网络待跑 |
| 6–7（bnlearn 部分） | 融合方法与小样本实验完成，已在 ASIA 上验证 |
| 6–7（TableShift 部分） | 未开始 |
| 8–12 | 未开始 |

详细结论与坑见 [`docs/选题二_实验说明与运行手册.md`](docs/选题二_实验说明与运行手册.md)。

周 1 与周 2 的产出说明见 [`docs/周1验收清单.md`](docs/周1验收清单.md)
和 [`docs/周2验收清单.md`](docs/周2验收清单.md)。

## 已知限制

* 本机是 RTX 3060 Laptop（6GB）。**文档要求的 8B 模型 4bit 实测可以跑**
  （加载 33 秒、显存 6002 MiB、约 48 秒/条），4bit 不支持分层到 CPU，
  所以必须 `device_map={"": 0}` 而不是 `auto`。
  按这个速度，ASIA 约 1.9 小时、CHILD 约 12.7 小时、INSURANCE 约 23 小时、
  ALARM 约 44 小时、HAILFINDER 约 102 小时——**小网络本机跑，大网络必须用共用的 4090，
  或只对相关变量对采样**。
* vLLM 在原生 Windows 装不了（无对应发行版），需要 WSL2 或 Linux。
* 方向准确率原先受限于"以为没有 R 就拿不到 DAG"。
  实际上 R 是装了的（`D:\R\R-4.6.1`，含 bnlearn），而我最初没找到它。
  现在 `src/mb/rds_dag.py` 直接解析 `rds` 里的 bn.fit 对象恢复有向图，
  并由 bnlearn（`scripts/verify_dag_bnlearn.R`）与
  "道德图 vs 真值马尔可夫毯矩阵"（`scripts/export_dags.py`）两路独立验证，
  5 个网络全部一致。
