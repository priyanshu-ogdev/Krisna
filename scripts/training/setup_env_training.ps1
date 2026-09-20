<#
.SYNOPSIS
    Set up Python virtual environment and dependencies for Krisna Model Training on Windows.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

Write-Host "Setting up Krisna training environment on Windows..." -ForegroundColor Cyan

if (-not (Test-Path "$RepoRoot\.venv")) {
    Write-Host "Creating Python virtual environment in .venv..." -ForegroundColor DarkGray
    python -m venv "$RepoRoot\.venv"
}

$Python = "$RepoRoot\.venv\Scripts\python.exe"
$Pip = "$RepoRoot\.venv\Scripts\pip.exe"

Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor DarkGray
& $Python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch with CUDA 12 support..." -ForegroundColor DarkGray
& $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing training package in editable mode..." -ForegroundColor DarkGray
& $Pip install -e "$RepoRoot\training"
& $Pip install -e "$RepoRoot\inference"

Write-Host "`n[SUCCESS] Training environment ready." -ForegroundColor Green
Write-Host "To activate in PowerShell:" -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
