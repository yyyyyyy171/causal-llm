<#
一键跑完全部网络的"语言模型因果先验"。

建议在夜间或周末整段运行（文档里"长任务放在夜间整段跑"的约定）。
每个网络跑完立刻落盘，中断后重跑会跳过已完成的部分，不会白跑。

用法：
    powershell -ExecutionPolicy Bypass -File scripts\run_all_priors.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_all_priors.ps1 -Networks CHILD,ALARM

停止：直接关掉这个窗口，或 Ctrl+C。已经完成的部分会保存在
      data\<NET>\llm_cache\ 下，下次继续。
#>

param(
    [string]$Networks = "CHILD,INSURANCE,ALARM,HAILFINDER",
    [int]$Runs = 5,
    [int]$BatchSize = 12,
    [string]$Python = "D:\Anaconda\envs\llm_causal_fs\python.exe"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot "logs"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if (-not (Test-Path $Python)) {
    throw "找不到 python：$Python"
}

foreach ($network in $Networks.Split(",")) {

    $network = $network.Trim()
    if (-not $network) { continue }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $logDir "$network`_$stamp.log"

    Write-Host ""
    Write-Host "===== $network : 真实名 + 匿名，每对 $Runs 次 =====" -ForegroundColor Cyan
    Write-Host "日志：$log"

    & $Python (Join-Path $projectRoot "scripts\run_llm_prior.py") `
        --network $network `
        --mode both `
        --runs $Runs `
        --batch-size $BatchSize `
        *>&1 | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$network 运行失败，退出码 $LASTEXITCODE" -ForegroundColor Red
        break
    }

    Write-Host "$network 完成" -ForegroundColor Green
}

Write-Host ""
Write-Host "全部结束。接下来在 base 环境里跑评测与融合实验：" -ForegroundColor Yellow
Write-Host "  python scripts\eval_llm_prior.py --networks ASIA,ALARM --modes real,anon --out llm_prior_eval.csv"
Write-Host "  python scripts\run_fusion_experiment.py --networks ASIA,CHILD,INSURANCE --sizes 100,500 --repeats 5 --mode real"
Write-Host "  python scripts\run_fusion_experiment.py --networks ASIA,CHILD,INSURANCE --sizes 100,500 --repeats 5 --mode anon"
