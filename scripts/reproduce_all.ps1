<#
一键复现已有的结果表（只跑 CPU 部分）。

用途：让任何人（包括导师）拿到仓库后，能用一条命令把 results/ 下的表重新生成一遍，
而不需要显卡、不需要语言模型推理。语言模型先验需要 GPU，属于前置条件：
如果 data/<网络>/ 下没有 <网络>_LLM_prior_<模式>_summary.csv，
依赖先验的那一步会跳过并提示。

用法：
    powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1
    powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1 -SkipSlow   # 跳过 20 分钟的融合实验
    powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1 -Only fs    # 只跑某一步

环境：R 用 D:\R\R-4.6.1，Python 分析用 base 环境，Word 文档用 Codex 自带 Python。
#>

param(
    [switch]$SkipSlow,
    [string]$Only = "",
    [string]$Rscript = "D:\R\R-4.6.1\bin\Rscript.exe",
    [string]$Python = "python",
    [string]$DocxPython = "C:\Users\yuhan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$script:Failed = @()

function Invoke-Step {
    param([string]$Name, [string]$Key, [scriptblock]$Action, [string]$Output)

    if ($Only -and $Only -ne $Key) { return }

    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    if ($Output) { Write-Host "产出: $Output" }

    $ErrorActionPreference = "Continue"
    & $Action
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"

    if ($code -ne 0) {
        Write-Host "失败，退出码 $code" -ForegroundColor Red
        $script:Failed += $Name
    } else {
        Write-Host "完成" -ForegroundColor Green
    }
}

Write-Host "仓库: $projectRoot"
Write-Host "（语言模型先验需要 GPU，不在此脚本范围内；缺先验的实验会自动跳过）"

Invoke-Step -Name "1. 导出各网络的 DAG 真值（含道德图一致性交叉验证）" -Key "dag" `
    -Output "data/<网络>/<网络>_dag_edges.csv" `
    -Action { & $Python scripts\export_dags.py }

Invoke-Step -Name "2. 用 bnlearn 独立复核 DAG 真值" -Key "dag" `
    -Output "（控制台输出，应全为 identical=TRUE）" `
    -Action { & $Rscript --vanilla scripts\verify_dag_bnlearn.R }

Invoke-Step -Name "3. IAMB 基线（5 网络 x 4 样本量 x mi/x2 两种检验）" -Key "mb" `
    -Output "results/bnlearn_mb_summary.csv" `
    -Action { & $Rscript --vanilla scripts\run_mb_baseline_bnlearn.R --methods=iamb --tests=mi,x2 }

Invoke-Step -Name "4. HITON-MB 基线（R 移植版，与 MATLAB 基线对照）" -Key "mb" `
    -Output "results/hiton_baseline_hiton.csv、results/hiton_baseline_vs_matlab_hiton.csv" `
    -Action { & $Rscript --vanilla scripts\run_hiton_baseline_bnlearn.R --networks=ASIA,CHILD,INSURANCE --sizes=100,500,1000,5000 }

Invoke-Step -Name "5. 共用特征选择工具箱（5 数据集 x 5 方法，约 105 秒）" -Key "fs" `
    -Output "results/fs_method_by_dataset.csv、fs_benchmark_long.csv、fs_selected_features.csv" `
    -Action { & $Python scripts\run_fs_benchmark.py }

Invoke-Step -Name "6. 先验质量与方向准确率（需要已有先验）" -Key "prior" `
    -Output "results/llm_prior_eval.csv、results/mb_prior_quality.csv" `
    -Action {
        & $Python scripts\eval_llm_prior.py --networks ASIA,CHILD --modes real,anon --out llm_prior_eval.csv
        & $Rscript --vanilla scripts\eval_mb_prior.R --networks=ASIA --modes=real
    }

Invoke-Step -Name "7. 先验与配偶的错配诊断" -Key "prior" `
    -Output "results/prior_vs_spouses_real.csv" `
    -Action { & $Rscript --vanilla scripts\diagnose_prior_vs_spouses.R }

if (-not $SkipSlow) {
    Invoke-Step -Name "8. 融合实验：IAMB + HITON-MB，30 次重复，配对检验（约 20-40 分钟）" -Key "fusion" `
        -Output "results/fusion_both_summary_both_nreal.csv、fusion_both_paired_both_nreal.csv" `
        -Action { & $Rscript --vanilla scripts\run_fusion_both.R --networks=ASIA,CHILD --sizes=100,500 --repeats=30 --tag=both_nreal }
}

Invoke-Step -Name "9. 生成周产出文档（7 份）" -Key "report" `
    -Output "E:\汇报\周产出\*.docx" `
    -Action {
        foreach ($week in @("week01", "week02", "week03", "week04", "week05", "week06_07", "week08_10")) {
            & $DocxPython scripts\make_week_report.py --spec "scripts\week_reports\$week.json"
        }
    }

Write-Host ""
if ($script:Failed.Count -eq 0) {
    Write-Host "全部完成。" -ForegroundColor Green
} else {
    Write-Host "以下步骤失败：" -ForegroundColor Red
    $script:Failed | ForEach-Object { Write-Host "  - $_" }
    exit 1
}
