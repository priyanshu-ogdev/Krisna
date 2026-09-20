# =============================================================================
# scripts/inference/setup_env_inference.ps1
# Sets up virtual environment and installs dependencies for Krisna Inference.
#
# Usage:
#   .\scripts\inference\setup_env_inference.ps1
#   .\scripts\inference\setup_env_inference.ps1 -InstallBackends
# =============================================================================

[CmdletBinding()]
param(
    [switch]$InstallBackends,
    [switch]$RequireGPU
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

# Force UTF-8 on Windows Console
if ([Console]::OutputEncoding.CodePage -ne 65001) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "       Krisna Inference -- Environment Setup (PowerShell)       " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# 1. Resolve or create virtualenv
$VenvDir = Join-Path $RepoRoot ".venv"
$PythonBin = "python"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[+] Creating virtual environment in .venv..." -ForegroundColor Green
    python -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
}

Write-Host "[i] Using Python: $PythonBin" -ForegroundColor DarkGray

# 2. Upgrade pip and build tools
Write-Host "[+] Ensuring pip, setuptools, and wheel are up to date..." -ForegroundColor Green
& $PythonBin -m pip install --upgrade pip setuptools wheel

# 3. Install packages in editable mode
Write-Host "[+] Installing krisna-inference and krisna-training in editable mode..." -ForegroundColor Green
& $PythonBin -m pip install -e "$RepoRoot\inference"
& $PythonBin -m pip install -e "$RepoRoot\training"

# 4. Hardware compatibility check
Write-Host "[+] Running hardware preflight verification..." -ForegroundColor Green
$checkArgs = @("$RepoRoot\scripts\inference\check_hardware.py")
if ($RequireGPU) {
    $checkArgs += "--require-gpu"
}
& $PythonBin @checkArgs
if ($LASTEXITCODE -ne 0 -and $RequireGPU) {
    Write-Host "[ERROR] GPU check failed and -RequireGPU was specified. Halting." -ForegroundColor Red
    exit 1
}

# 5. Install backend dependencies if requested
if ($InstallBackends) {
    Write-Host "[+] Installing real model backend dependencies (torch, diffusers, transformers, bitsandbytes)..." -ForegroundColor Yellow
    & $PythonBin -m pip install -e "$RepoRoot\inference[backends]"
} else {
    Write-Host "[i] Core dev dependencies installed. Pass -InstallBackends to install full GPU/Torch stack." -ForegroundColor DarkGray
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "   Inference environment setup complete!                        " -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
