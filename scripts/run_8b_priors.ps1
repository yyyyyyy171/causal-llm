<#
按文档要求用 4bit 的 8B 模型重跑先验。

背景：文档写的是"本地 8B 模型""4bit 的 8B 模型即可"，
而之前为了省显存擅自降级成了 4B。实测 Qwen3-8B 4bit 在这台 6GB 卡上
占 6.08 GB、单条生成 13 秒，跑得动，所以按文档补上。

顺序：先 ASIA 的两种提问方式（方向 / 马尔可夫毯成员），各约 30 分钟。
     ASIA 对了再谈 CHILD（约 3.4 小时）与 INSURANCE（约 6.5 小时）。

用法：
    powershell -ExecutionPolicy Bypass -File scripts\run_8b_priors.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_8b_priors.ps1 -Networks ASIA -Tasks direction,mb
#>

param(
    [string]$Networks = "ASIA",
    [string]$Tasks = "direction,mb",
    [string]$Model = "Qwen/Qwen3-8B",
    [string]$Python = "D:\Anaconda\envs\llm_causal_fs\python.exe"
)

$ErrorActionPreference = "Stop"

# tqdm / transformers 的进度条走 stderr。在 PowerShell 5.1 里，
# 原生命令的 stderr 会被包装成 ErrorRecord，配合 ErrorActionPreference=Stop
# 会在"正在加载模型"这一步直接把脚本打断（第一次跑就是这么失败的）。
# 所以调用 Python 期间放宽错误处理，改用 $LASTEXITCODE 判断成败。
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

Set-Location $projectRoot

foreach ($network in $Networks.Split(",")) {
    foreach ($task in $Tasks.Split(",")) {

        $network = $network.Trim()
        $task = $task.Trim()
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $log = Join-Path $logDir "$network`_$task`_8b_$stamp.log"

        Write-Host ""
        Write-Host "===== $network / $task / 8B 4bit =====" -ForegroundColor Cyan
        Write-Host "日志: $log"

        $ErrorActionPreference = "Continue"

        & $Python (Join-Path $projectRoot "scripts\run_llm_prior.py") `
            --network $network --mode real --task $task `
            --model $Model --quant 4bit --device-map gpu0 --batch-size 1 2>&1 |
            Tee-Object -FilePath $log

        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = "Stop"

        if ($exitCode -ne 0) {
            Write-Host "$network/$task 失败，退出码 $exitCode" -ForegroundColor Red
            exit $exitCode
        }
    }
}

Write-Host ""
Write-Host "全部完成。用 R 复核先验质量：" -ForegroundColor Yellow
Write-Host "  & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\eval_mb_prior.R --networks=ASIA --modes=real"
