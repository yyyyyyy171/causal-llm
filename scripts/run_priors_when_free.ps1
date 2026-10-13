<#
等前一个任务结束后，接着跑某个网络的语言模型先验。

用途：6GB 显卡上一次只能跑一个模型，所以匿名版必须等真实名版跑完再开始。
把这个脚本挂到后台，它就会自动接着跑，不用守着。

用法：
    powershell -ExecutionPolicy Bypass -File scripts\run_priors_when_free.ps1 `
        -WaitForPid 1408 -Network CHILD

停止：直接结束这个 powershell 进程即可；已经跑完的部分会保留，
      重跑会自动跳过。
#>

param(
    [int]$WaitForPid = 0,
    [string]$Network = "CHILD",
    [string]$Python = "D:\Anaconda\envs\llm_causal_fs\python.exe",
    [int]$MaxWaitMinutes = 240
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

if ($WaitForPid -gt 0) {
    Write-Host "等待进程 $WaitForPid 结束（最多 $MaxWaitMinutes 分钟）..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddMinutes($MaxWaitMinutes)

    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
            Write-Host "进程 $WaitForPid 已结束，开始下一个任务。" -ForegroundColor Green
            break
        }
        Start-Sleep -Seconds 30
    }
}

# --mode both 会先检查真实名部分：已经全部完成的会直接跳过，只补匿名部分。
Write-Host "开始 $Network 的先验推理（real 已完成则只跑 anon）..." -ForegroundColor Cyan

Set-Location $projectRoot

& $Python (Join-Path $projectRoot "scripts\run_llm_prior.py") `
    --network $Network --mode both

Write-Host "完成，退出码 $LASTEXITCODE" -ForegroundColor Green
