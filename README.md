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
scripts/         可执行脚本
src/             llm_prior（语言模型先验）、mb（马尔可夫毯发现）、fs（特征选择工具箱）、common（配置与日志）
```

## 环境

两个 conda 环境分工明确，不要混用：

| 环境 | 用途 | 关键包 |
|---|---|---|
| `D:\Anaconda`（base） | CPU 分析：马尔可夫毯发现、融合实验、评测、画图 | numpy、scipy、pandas、matplotlib |
| `D:\Anaconda\envs\llm_causal_fs` | GPU 推理：加载本地 Qwen 模型 | torch 2.5.1+cu121、transformers、bitsandbytes |

语言模型缓存位于 `D:\hf-cache`（环境变量 `HF_HOME`），
推理脚本默认 `local_files_only=True`，完全离线运行。

## 快速开始

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

* 本机是 RTX 3060 Laptop（6GB），只能用 4B 模型的 4bit 量化。
  文档假设的 8B 模型与 vLLM 需要 4090（原生 Windows 装不了 vLLM，见使用约定）。
* 方向准确率原先受限于"以为没有 R 就拿不到 DAG"。
  实际上 R 是装了的（`D:\R\R-4.6.1`，含 bnlearn），而我最初没找到它。
  现在 `src/mb/rds_dag.py` 直接解析 `rds` 里的 bn.fit 对象恢复有向图，
  并由 bnlearn（`scripts/verify_dag_bnlearn.R`）与
  "道德图 vs 真值马尔可夫毯矩阵"（`scripts/export_dags.py`）两路独立验证，
  5 个网络全部一致。
