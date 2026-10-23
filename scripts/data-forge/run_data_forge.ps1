# =============================================================================
# scripts/data-forge/run_data_forge.ps1
# Master launcher & orchestrator for the Krivex Data-Forge pipeline (Windows native).
#
# Usage:
#   .\scripts\data-forge\run_data_forge.ps1 doctor
#   .\scripts\data-forge\run_data_forge.ps1 run --dry-run
#   .\scripts\data-forge\run_data_forge.ps1 manifest stats
#   .\scripts\data-forge\run_data_forge.ps1 sync-training
#   .\scripts\data-forge\run_data_forge.ps1 all [--dry-run]
#   .\scripts\data-forge\run_data_forge.ps1 setup
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

# Navigate to monorepo root (2 levels up from scripts/data-forge)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
Set-Location $RepoRoot

# Force UTF-8 on Windows Console
if ([Console]::OutputEncoding.CodePage -ne 65001) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   Krivex Data-Forge -- Master Pipeline Launcher (PowerShell)   " -ForegroundColor Cyan
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

# 5. Resolve Python executable
$Python = ""
if ($env:PYTHON_EXECUTABLE -and (Test-Path $env:PYTHON_EXECUTABLE)) {
    $Python = $env:PYTHON_EXECUTABLE
} elseif (Test-Path "C:\Users\anubh\miniconda3\envs\torch_env\python.exe") {
    $Python = "C:\Users\anubh\miniconda3\envs\torch_env\python.exe"
} elseif (Test-Path (Join-Path $RepoRoot ".venv\Scripts\python.exe")) {
    $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
} elseif (Test-Path (Join-Path $RepoRoot ".venv-forge\Scripts\python.exe")) {
    $Python = Join-Path $RepoRoot ".venv-forge\Scripts\python.exe"
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $Python = $cmd.Source
    }
}

if (-not $Python) {
    Write-Host "ERROR: python not found in environment or PATH." -ForegroundColor Red
    Write-Host "Run '.\scripts\data-forge\run_data_forge.ps1 setup' to configure the environment." -ForegroundColor Yellow
    exit 1
}

# 6. Parse command
$Action = ""
if ($ScriptArgs -and $ScriptArgs.Count -gt 0) {
    $Action = $ScriptArgs[0].ToLower()
}

if (-not $Action -or $Action -eq "help" -or $Action -eq "--help" -or $Action -eq "-h") {
    Write-Host "Usage: .\scripts\data-forge\run_data_forge.ps1 <command> [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  doctor         Run pre-flight checks (stage graph, toolchains, APIs)"
    Write-Host "  audit          Run forensic registry audit & contract validation"
    Write-Host "  registry       Check models and datasets upstream releases / audit"
    Write-Host "  pin-revisions  Resolve & pin model/dataset revisions to immutable git SHAs"
    Write-Host "  verify-schemas Validate JSON schemas against Pydantic models"
    Write-Host "  run            Run the data pipeline (supports --dry-run, --resume, --stages)"
    Write-Host "  manifest       Inspect SQLite database (stats, query, export)"
    Write-Host "  sync-training  Bridge model_data/ into training directory"
    Write-Host "  all            Complete flow: doctor -> audit -> run -> sync-training"
    Write-Host "  setup          Set up Windows environment"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\data-forge\run_data_forge.ps1 audit"
    Write-Host "  .\scripts\data-forge\run_data_forge.ps1 pin-revisions --apply"
    Write-Host "  .\scripts\data-forge\run_data_forge.ps1 run --dry-run"
    Write-Host "  .\scripts\data-forge\run_data_forge.ps1 sync-training"
    Write-Host "  .\scripts\data-forge\run_data_forge.ps1 all"
    exit 0
}

# Slice remaining arguments
$Remaining = @()
if ($ScriptArgs.Count -gt 1) {
    $Remaining = $ScriptArgs[1..($ScriptArgs.Count - 1)]
}

if ($Action -eq "setup") {
    & powershell -File (Join-Path $RepoRoot "scripts\data-forge\setup_env.ps1") @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "audit" -or $Action -eq "registry") {
    Write-Host "[*] Auditing dataset & model registry preflight contracts..." -ForegroundColor Cyan
    & $Python (Join-Path $RepoRoot "scripts\data-forge\audit_registry.py") @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "pin-revisions") {
    Write-Host "[*] Resolving unpinned revisions to immutable commit SHAs..." -ForegroundColor Cyan
    & $Python (Join-Path $RepoRoot "scripts\data-forge\pin_revisions.py") @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "verify-schemas") {
    Write-Host "[*] Verifying JSON schemas against Pydantic definitions..." -ForegroundColor Cyan
    & $Python (Join-Path $RepoRoot "scripts\data-forge\verify_schemas.py") @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "doctor") {
    Write-Host "[*] Running Data-Forge Pre-Flight Diagnostics..." -ForegroundColor Cyan
    & $Python -m data_forge.cli doctor @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "run") {
    Write-Host "[*] Executing Data-Forge Pipeline Engine..." -ForegroundColor Cyan
    & $Python -m data_forge.cli run @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "manifest") {
    Write-Host "[*] Querying Data-Forge Manifest SQLite Database..." -ForegroundColor Cyan
    & $Python -m data_forge.cli manifest @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "sync-training") {
    Write-Host "[*] Synchronizing exported data into training-ready datasets..." -ForegroundColor Cyan
    & $Python (Join-Path $RepoRoot "scripts\data-forge\sync_to_training.py") @Remaining
    exit $LASTEXITCODE
}

if ($Action -eq "all") {
    $DryRunArg = @()
    if ($Remaining -contains "--dry-run") {
        $DryRunArg = @("--dry-run")
    }

    Write-Host "[Step 1/4] Running doctor diagnostics..." -ForegroundColor Cyan
    & $Python -m data_forge.cli doctor
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Doctor identified warnings; continuing..." -ForegroundColor Yellow
    }

    Write-Host "[Step 2/4] Running forensic registry audit..." -ForegroundColor Cyan
    & $Python (Join-Path $RepoRoot "scripts\data-forge\audit_registry.py")

    Write-Host "[Step 3/4] Executing pipeline engine..." -ForegroundColor Cyan
    & $Python -m data_forge.cli run @DryRunArg

    Write-Host "[Step 4/4] Synchronizing datasets to training layer..." -ForegroundColor Cyan
    & $Python (Join-Path $RepoRoot "scripts\data-forge\sync_to_training.py")

    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "   Complete Data-Forge Pipeline & Training Bridge Finished!     " -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "Ready for model training:" -ForegroundColor Yellow
    Write-Host "   .\scripts\training\train_all.ps1" -ForegroundColor Yellow
    Write-Host "   ./scripts/training/train.sh polish-default" -ForegroundColor Yellow
    Write-Host "   ./scripts/training/train.sh polish-dpo" -ForegroundColor Yellow
    exit 0
}

# Standard data-forge CLI pass-through
& $Python -m data_forge.cli @ScriptArgs
exit $LASTEXITCODE
