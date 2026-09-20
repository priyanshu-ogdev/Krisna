# =============================================================================
# scripts/inference/run_service.ps1
# Runs the Krisna Inference FastAPI service on Windows native PowerShell.
#
# Usage:
#   .\scripts\inference\run_service.ps1
#   .\scripts\inference\run_service.ps1 -BindHost "0.0.0.0" -Port 8080
# =============================================================================

[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8420,
    [switch]$NoReload,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

# Force UTF-8 on Windows Console
if ([Console]::OutputEncoding.CodePage -ne 65001) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}

Write-Host "Starting Krisna Inference service on http://$($BindHost):$Port..." -ForegroundColor Cyan

if ($env:KRISNA_USE_REAL_BACKENDS -eq "1") {
    Write-Host "KRISNA_USE_REAL_BACKENDS=1 -- real model backends, GPU required." -ForegroundColor Yellow
    Write-Host "Critic tier launches subprocess interpreter (override with KRISNA_CRITIC_VENV_PYTHON)." -ForegroundColor Yellow
    if ($env:KRISNA_LOW_VRAM_MODE -eq "1") {
        Write-Host "KRISNA_LOW_VRAM_MODE=1 -- targeting low-VRAM envelope with system RAM offload." -ForegroundColor Magenta
    }
} else {
    Write-Host "MockBackend mode (default) -- no GPU/weights needed." -ForegroundColor DarkCyan
    Write-Host "Set `$env:KRISNA_USE_REAL_BACKENDS='1' to use real model backends." -ForegroundColor DarkGray
}

# Resolve Python interpreter
$PythonBin = "python"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
}

# Ensure PYTHONPATH
$InfSrc = Join-Path $RepoRoot "inference\src"
$TrainSrc = Join-Path $RepoRoot "training\src"
$env:PYTHONPATH = "$InfSrc;$TrainSrc;$env:PYTHONPATH"

$UvicornArgs = @(
    "-m", "uvicorn",
    "krisna_inference.orchestrator.service:app",
    "--host", $BindHost,
    "--port", [string]$Port
)

if (-not $NoReload) {
    $UvicornArgs += "--reload"
}

if ($RemainingArgs) {
    $UvicornArgs += $RemainingArgs
}

& $PythonBin $UvicornArgs
