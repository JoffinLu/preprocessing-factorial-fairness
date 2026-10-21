# 一键复现：环境自检 -> 数据下载/审计 -> 管线自检 -> 主实验 -> 统计 -> 可复现性比对
#
# 用法
#   .\reproduce.ps1                  # 全流程（主实验会按已有结果断点续跑）
#   .\reproduce.ps1 -Jobs 6          # 指定并行进程数
#   .\reproduce.ps1 -SkipRun         # 只做环境自检 + 统计 + 可复现性比对
#   .\reproduce.ps1 -Quick           # 冒烟：每数据集只跑 2 个预处理单元
#
# 前置
#   Python 3.13 + pip install -r requirements.txt
#   （干净环境验证：python -m venv .venv-clean; .venv-clean\Scripts\pip install -r requirements.txt）
param(
    [int]$Jobs = 4,
    [switch]$SkipRun,
    [switch]$Quick,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $Python) {
    $candidates = @("$Root\.venv\Scripts\python.exe", "python")
    foreach ($c in $candidates) {
        if ($c -eq "python" -or (Test-Path $c)) { $Python = $c; break }
    }
}
$env:PYTHONIOENCODING = "utf-8"
function Step($name, $script, $extra) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $argv = @("$Root\code\$script") + $extra
    & $Python @argv
    if ($LASTEXITCODE -ne 0) { throw "$name 失败（退出码 $LASTEXITCODE）" }
}

Step "1/6 依赖与关键算法自检" "00_verify_deps.py" @()
Step "2/6 数据下载与审计" "01_download_and_audit.py" @()

Write-Host "`n=== 3/6 管线模块自检 ===" -ForegroundColor Cyan
foreach ($m in @("config", "data", "missing", "features", "impute", "imbalance",
                 "intervention", "model", "metrics", "run_cell")) {
    & $Python "$Root\code\pipeline\$m.py" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pipeline/$m.py 自检失败" }
}
Write-Host "10 个模块自检全部通过"

if (-not $SkipRun) {
    $extra = @("--jobs", "$Jobs", "--out", "results/full_results.csv", "--resume")
    if ($Quick) { $extra += @("--limit", "2") }
    Step "4/6 主实验" "02_run_experiment.py" $extra
    if (-not $Quick) {
        Step "4b/6 自然缺失稳健性对照（adult × NAT）" "02_run_experiment.py" @(
            "--datasets", "adult", "--mechanisms", "NAT", "--natural-missing",
            "--jobs", "$Jobs", "--out", "results/nat_results.csv", "--resume")
    }
} else {
    Write-Host "`n=== 4/6 主实验：已跳过（-SkipRun）===" -ForegroundColor Yellow
}

Step "5/6 统计分析" "03_statistical_analysis.py" @()
Step "6/6 可复现性比对" "04_verify_reproducibility.py" @("--n", "10")

Write-Host "`n全部完成。产出：" -ForegroundColor Green
Write-Host "  results/full_results.csv      主设计 1,620 行"
Write-Host "  results/nat_results.csv       稳健性 162 行"
Write-Host "  results/stats/*.csv           5 张统计表"
