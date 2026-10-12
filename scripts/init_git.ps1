<#
首次提交 + 建立个人分支（周 1 的 Git 规范）。

为什么需要你自己跑一次：Codex 的沙箱把 .git 目录设成只读，
提交这类写操作在沙箱里会被拒绝，必须在你的终端里执行。

用法：
    powershell -ExecutionPolicy Bypass -File scripts\init_git.ps1
    powershell -ExecutionPolicy Bypass -File scripts\init_git.ps1 -Branch "zhangsan/topic2"

做完之后如果要推到远程仓库，再执行：
    git remote add origin <你的仓库地址>
    git push -u origin yyyyyyy171/topic2
#>

param(
    [string]$Repo = "E:\Causal-LLM",
    [string]$Branch = "yyyyyyy171/topic2"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    throw "$Repo 不是 git 仓库"
}

# 1) 让 git 信任这个目录（属主与当前用户不同会导致 dubious ownership）
git config --global --add safe.directory $Repo

Set-Location $Repo

Write-Host "=== 当前分支与状态 ===" -ForegroundColor Cyan
git branch --show-current
git status --short | Measure-Object -Line | ForEach-Object {
    "待提交文件数：$($_.Lines)"
}

# 2) 首次提交
Write-Host "`n=== 暂存并提交 ===" -ForegroundColor Cyan
git add -A

$message = @(
    "选题二：基线复现、语言模型先验管道与融合方法",
    "",
    "基线：5 个 bnlearn 网络的 IAMB/HITON-MB 结果与样本量-F1 曲线；",
    "Python 版 IAMB 在 ASIA n=5000 上复算 0.79643，与 MATLAB 的 0.796428571 一致。",
    "",
    "先验：补齐 5 个网络 148 个变量的语义字典与匿名映射；重写因果先验 Prompt",
    "（加直连判断、修正置信度恒为 0 的缺陷、加 PROMPT_VERSION 版本控制）；",
    "新增批量推理脚本（磁盘缓存、断点续跑、批处理）。",
    "",
    "融合：新增 Python 版 IAMB/HITON-MB、先验阈值调节与搜索优先级两种融合方式、",
    "随机先验对照、匿名消融；ASIA 小样本实验完成。",
    "",
    "规范（周1）：configs/ 配置文件、logs/ 日志落盘（记录命令行与 git 提交）、",
    "README、4090 使用约定、实验日志、周 1 验收清单。",
    "",
    "工具箱（周2）：src/fs/ 共用特征选择评测框架（固定划分落盘、固定下游模型、",
    "统一预处理、五个方法返回完整排序），5 个 OpenML 数据集 x 5 种方法的结果表，",
    "以及数据集体检（多数类基线 + 全特征准确率告警）。"
) -join "`n"

git commit -m $message

# 3) 个人分支
Write-Host "`n=== 建立并切换到个人分支 $Branch ===" -ForegroundColor Cyan
git branch $Branch
git checkout $Branch

Write-Host "`n=== 结果 ===" -ForegroundColor Green
git log --oneline --all --decorate
git status --short

Write-Host "`n下一步（可选）：推送" -ForegroundColor Yellow
Write-Host "  git remote add origin <你的仓库地址>"
Write-Host "  git push -u origin $Branch"
