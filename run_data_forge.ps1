# =============================================================================
# run_data_forge.ps1 (Repository Root)
# Master launcher & orchestrator for the Krisna Data-Forge pipeline (Windows native).
#
# Usage:
#   .\run_data_forge.ps1 doctor
#   .\run_data_forge.ps1 run --dry-run
#   .\run_data_forge.ps1 manifest stats
#   .\run_data_forge.ps1 sync-training
#   .\run_data_forge.ps1 all [--dry-run]
#   .\run_data_forge.ps1 setup
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

# Navigate to repo root
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RepoRoot

# Force UTF-8 on Windows Console
if ([Console]::OutputEncoding.CodePage -ne 65001) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   Krisna Data-Forge -- Master Pipeline Launcher (PowerShell)   " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# 1. Check or initialize .env
$EnvFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $EnvFile)) {
    $EnvExample = Join-Path $RepoRoot ".env.example"
    if (Test-Path $EnvExample) {
        Write-Host "[!] .env not found. Initializing from .env.example..." -ForegroundColor Yellow
        Copy-Item $EnvExample $EnvFile
        Write-Host "[!] Please update .env with your HF_TOKEN before production runs." -ForegroundColor Yellow
    }
}

# 2. Auto-load .env into process environment
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $name = $parts[0].Trim()
            $val = $parts[1].Trim().Trim("'", '"')
            if ($name -and -not [System.Environment]::GetEnvironmentVariable($name)) {
                [System.Environment]::SetEnvironmentVariable($name, $val, "Process")
            }
        }
    }
}

# 3. Activate virtual environment if present
foreach ($candidate in @(".venv", ".venv-forge")) {
    $act = Join-Path $RepoRoot "$candidate\Scripts\Activate.ps1"
    if (Test-Path $act) {
        & $act
        break
    }
}

# 4. Configure PYTHONPATH
$DataForgeSrc = Join-Path $RepoRoot "data-forge\src"
$TrainingSrc = Join-Path $RepoRoot "training\src"
$InferenceSrc = Join-Path $RepoRoot "inference\src"
$env:PYTHONPATH = "$DataForgeSrc;$TrainingSrc;$InferenceSrc;$RepoRoot;" + $env:PYTHONPATH

# 5. Check Python command
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Host "ERROR: python not found in PATH." -ForegroundColor Red
    exit 1
}

# 6. Parse command
$Action = ""
if ($ScriptArgs -and $ScriptArgs.Count -gt 0) {
    $Action = $ScriptArgs[0].ToLower()
}

if (-not $Action -or $Action -eq "help" -or $Action -eq "--help") {
    Write-Host "Usage: .\run_data_forge.ps1 <command> [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  doctor         Run pre-flight checks (stage graph, toolchains, APIs)"
    Write-Host "  run            Run the data pipeline (supports --dry-run, --resume, --stages)"
    Write-Host "  manifest       Inspect SQLite database (stats, query, export)"
    Write-Host "  registry       Check models and datasets upstream releases"
    Write-Host "  sync-training  Bridge model_data/ into ./data/ ready for train.sh"
    Write-Host "  all            Complete flow: doctor -> run -> sync-training"
    Write-Host "  setup          Set up Windows environment"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_data_forge.ps1 run --dry-run"
    Write-Host "  .\run_data_forge.ps1 sync-training"
    Write-Host "  .\run_data_forge.ps1 all"
    exit 0
}

# Slice remaining arguments
$Remaining = @()
if ($ScriptArgs.Count -gt 1) {
    $Remaining = $ScriptArgs[1..($ScriptArgs.Count - 1)]
}

if ($Action -eq "setup") {
    & powershell -File ".\scripts\data-forge\setup_env.ps1"
    exit $LASTEXITCODE
}

if ($Action -eq "sync-training") {
    Write-Host "[*] Synchronizing exported data into training-ready datasets..." -ForegroundColor Cyan
    & python ".\scripts\data-forge\sync_to_training.py" @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "all") {
    $DryRunArg = @()
    if ($Remaining -contains "--dry-run") {
        $DryRunArg = @("--dry-run")
    }

    Write-Host "[Step 1/3] Running doctor check..." -ForegroundColor Cyan
    & python -m data_forge.cli doctor
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Doctor identified warnings; continuing..." -ForegroundColor Yellow
    }

    Write-Host "[Step 2/3] Executing pipeline..." -ForegroundColor Cyan
    & python -m data_forge.cli run @DryRunArg

    Write-Host "[Step 3/3] Synchronizing datasets to training layer..." -ForegroundColor Cyan
    & python ".\scripts\data-forge\sync_to_training.py"

    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "   Complete Data-Forge Pipeline & Training Bridge Finished!     " -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "Ready for model training:" -ForegroundColor Yellow
    Write-Host "   ./train.sh polish-default" -ForegroundColor Yellow
    Write-Host "   ./train.sh polish-dpo" -ForegroundColor Yellow
    Write-Host "   ./train.sh sketch-stage1" -ForegroundColor Yellow
    exit 0
}

# Standard data-forge CLI pass-through
& python -m data_forge.cli @ScriptArgs
exit $LASTEXITCODE
