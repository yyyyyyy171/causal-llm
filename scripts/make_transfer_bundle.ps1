<#
打包成上传到 Google Drive 的 zip（供 Colab 解压使用）。

注意：真正的打包由 scripts/make_transfer_bundle.py 完成，本脚本只是 Windows 侧的
入口。原因是 PowerShell 5.1 的 Compress-Archive 会把 zip 条目名写成反斜杠
（data\ALARM\x.csv），而 zip 规范要求正斜杠；Linux 解压时会把它当成一个文件名，
目录结构全部丢失（这个坑已经实测踩到过）。Python 的 zipfile 不会有这个问题。

用法：
    powershell -ExecutionPolicy Bypass -File scripts\make_transfer_bundle.ps1
    powershell -ExecutionPolicy Bypass -File scripts\make_transfer_bundle.ps1 -Out D:\causal-llm-bundle.zip

之后：把生成的 zip 上传到 Google Drive 的 MyDrive/causal-llm/ 下，
再打开 notebooks/colab_prior.ipynb 从上到下运行。
#>

param(
    [string]$Out = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    $fallback = "C:\Users\$env:USERNAME\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $fallback) { $Python = $fallback }
    else { throw "找不到 python，请用 -Python 指定路径" }
}

$arguments = @("scripts\make_transfer_bundle.py")
if ($Out) { $arguments += @("--out", $Out) }

& $Python @arguments

if ($LASTEXITCODE -ne 0) { throw "打包失败，退出码 $LASTEXITCODE" }

Write-Host ""
Write-Host "接下来：" -ForegroundColor Yellow
Write-Host "  1. 把这个 zip 上传到 Google Drive 的 MyDrive/causal-llm/ 目录"
Write-Host "  2. 打开 notebooks/colab_prior.ipynb，运行时类型选 T4 GPU"
Write-Host "  3. 从上到下依次运行；每段结束前务必运行“同步回 Drive”那一格"
