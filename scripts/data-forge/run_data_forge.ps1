# =============================================================================
# scripts/data-forge/run_data_forge.ps1
# Native PowerShell launcher for the Krisna Data-Forge automated pipeline.
#
# Usage:
#   .\scripts\data-forge\run_data_forge.ps1 doctor
#   .\scripts\data-forge\run_data_forge.ps1 run --dry-run
#   .\scripts\data-forge\run_data_forge.ps1 run --resume
#   .\scripts\data-forge\run_data_forge.ps1 run --stages 0,1,2
#   .\scripts\data-forge\run_data_forge.ps1 manifest stats
#   .\scripts\data-forge\run_data_forge.ps1 manifest query --status routed
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

# Navigate to monorepo root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
Set-Location $RepoRoot

# Auto-load .env from repo root if present
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Write-Host "Loading environment variables from .env..." -ForegroundColor Cyan
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

# Activate virtual environment if present
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

# Ensure data-forge package path is in PYTHONPATH
$DataForgePath = Join-Path $RepoRoot "data-forge"
$DataForgeSrc = Join-Path $RepoRoot "data-forge\src"
$ExistingPythonPath = $env:PYTHONPATH
if ($ExistingPythonPath) {
    $env:PYTHONPATH = "$DataForgePath;$DataForgeSrc;$ExistingPythonPath"
} else {
    $env:PYTHONPATH = "$DataForgePath;$DataForgeSrc"
}

# Force UTF-8 on Windows Console
if ([Console]::OutputEncoding.CodePage -ne 65001) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}

if (-not $ScriptArgs -or $ScriptArgs.Count -eq 0) {
    Write-Host "No command given -- showing data-forge help:" -ForegroundColor Yellow
    & python -m data_forge.cli --help
    exit $LASTEXITCODE
}

& python -m data_forge.cli @ScriptArgs
exit $LASTEXITCODE
