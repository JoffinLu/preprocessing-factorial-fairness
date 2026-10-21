# Revision-v2 reproduction. Each result CSV has a matching provenance manifest.
# Full: .\reproduce.ps1 -Jobs 4
# Audit existing completed main results: .\reproduce.ps1 -SkipRun
# Independent small smoke grid, without full statistical analysis: .\reproduce.ps1 -Quick
# Use an isolated environment: .\reproduce.ps1 -Python C:\path\to\venv\Scripts\python.exe
param(
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [switch]$SkipRun,
    [switch]$Quick,
    [string]$Python = "",
    [string]$ConfigPath = "configs/mvp.yaml",
    [string]$QuickOutput = "results/revised/quick.csv"
)

$ErrorActionPreference = "Stop"
$RevisionRoot = $PSScriptRoot
if (-not $Python) {
    $LocalPython = Join-Path $RevisionRoot ".venv/Scripts/python.exe"
    $Python = if (Test-Path -LiteralPath $LocalPython) { $LocalPython } else { "python" }
}
$env:PYTHONIOENCODING = "utf-8"
foreach ($ThreadOption in @("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
    [Environment]::SetEnvironmentVariable($ThreadOption, "1", "Process")
}

function Invoke-RevisionPython([string]$Description, [string[]]$Arguments) {
    Write-Host $Description -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Description failed (exit $LASTEXITCODE)" }
}

Push-Location -LiteralPath $RevisionRoot
try {
    $RawFiles = @("adult/adult.data", "german/german.data", "bank/bank-additional-full.csv",
                  "diabetes/diabetic_data.csv", "compas/compas-scores-two-years.csv")
    $MissingRaw = @($RawFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path "data/raw" $_)) })
    if ($MissingRaw.Count) {
        Invoke-RevisionPython "Download and audit public source data" @("code/01_download_and_audit.py")
    }
    Invoke-RevisionPython "Run protocol and inference regression checks" @("-m", "unittest", "discover", "-s", "code/tests", "-v")

    if ($Quick) {
        $ResultPath = $QuickOutput
        if (-not $SkipRun) {
            Invoke-RevisionPython "Run independent smoke grid" @(
                "code/05_run_revision.py", "--config", $ConfigPath,
                "--datasets", "german", "--mechanisms", "M1", "M2", "--a", "A1",
                "--b", "B1", "B3", "--c", "C1", "C2", "C3", "--seeds", "0",
                "--models", "LR", "LGBM", "--jobs", [string]$Jobs,
                "--out", $ResultPath, "--resume")
        }
        Invoke-RevisionPython "Check smoke provenance and replay two smoke treatments" @(
            "code/06_verify_revision.py", "--csv", $ResultPath, "--n", "2",
            "--report", "revision/reproduction_quick.json")
        Write-Host "Smoke completed. Full paired statistics were intentionally not run." -ForegroundColor Green
    }
    else {
        $ResultPath = "results/revised/main.csv"
        if (-not $SkipRun) {
            Invoke-RevisionPython "Run or resume the complete revised factorial" @(
                "code/05_run_revision.py", "--config", $ConfigPath,
                "--jobs", [string]$Jobs, "--out", $ResultPath, "--resume")
        }
        $ManifestPath = [IO.Path]::ChangeExtension($ResultPath, ".manifest.json")
        $ArchiveManifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
        if ($ArchiveManifest.status -ne "complete") {
            throw "Paired analysis requires a complete main manifest; current status is $($ArchiveManifest.status)"
        }
        $SeedCount = @($ArchiveManifest.config.seeds).Count
        Invoke-RevisionPython "Recompute paired contrasts and descriptive decompositions" @(
            "code/03_statistical_analysis.py", "--input", $ResultPath,
            "--out-dir", "results/revised/stats", "--expected-seeds", [string]$SeedCount,
            "--bootstrap", "5000")
        Invoke-RevisionPython "Verify completed provenance and replay ten representative treatments" @(
            "code/06_verify_revision.py", "--csv", $ResultPath, "--n", "10",
            "--report", "revision/reproduction_check.json")
        Write-Host "Completed main reproduction, paired analysis and sampled replay." -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
