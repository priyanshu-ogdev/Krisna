# =============================================================================
# run_inference.ps1 (Repository Root)
# Master launcher & orchestrator for the Krisna Inference subsystem (Windows native).
#
# Usage:
#   .\run_inference.ps1                     # MockBackend, no GPU needed (port 8420)
#   .\run_inference.ps1 -Real              # Real backends, GPU required
#   .\run_inference.ps1 -Real -LowVram     # Real backends, CPU offload for <16GB VRAM
#   .\run_inference.ps1 -WithFrontend      # Runs backend and launches Node.js frontend
#   .\run_inference.ps1 -CliSession        # Runs interactive CLI agentic session
#   .\run_inference.ps1 -Setup             # Installs dependencies for inference
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Help,
    [switch]$Real,
    [switch]$LowVram,
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8420,
    [switch]$WithFrontend,
    [switch]$CliSession,
    [switch]$Setup,
    [double]$RamEnvelopeGb = 128.0,
    [double]$VramEnvelopeGb = 0.0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RepoRoot

# Force UTF-8 on Windows Console
if ([Console]::OutputEncoding.CodePage -ne 65001) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "     Krisna Inference -- Master Service Launcher (PowerShell)   " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

if ($Help) {
    Write-Host @"
Usage:
  .\run_inference.ps1                     # MockBackend, no GPU needed (port 8420)
  .\run_inference.ps1 -Real              # Real backends, GPU required
  .\run_inference.ps1 -Real -LowVram     # Real backends, CPU offload for <16GB VRAM
  .\run_inference.ps1 -WithFrontend      # Runs backend and launches Node.js frontend
  .\run_inference.ps1 -CliSession        # Runs interactive CLI agentic session
  .\run_inference.ps1 -Setup             # Installs dependencies for inference
  .\run_inference.ps1 -Help              # Displays this help screen

Parameters:
  -Real               Use real models (GPU + models/ weights required)
  -LowVram            Offload models to system RAM for 12-16GB GPUs
  -BindHost <ip>      IP to bind service (default: 127.0.0.1)
  -Port <int>         Port to listen on (default: 8420)
  -WithFrontend       Concurrently starts the Node.js frontend on port 3100
  -CliSession         Run interactive agentic session directly in terminal
  -RamEnvelopeGb <n>  System RAM allocation for offloading (default: 128.0)
  -VramEnvelopeGb <n> Max GPU VRAM budget (default: 48.0 or 12.0 in LowVram)
"@ -ForegroundColor Yellow
    exit 0
}


# Setup environment if requested
if ($Setup) {
    Write-Host "[+] Running inference environment setup..." -ForegroundColor Green
    & "$RepoRoot\scripts\inference\setup_env_inference.ps1"
    exit $LASTEXITCODE
}

# 1. Load .env.inference or .env if present
$EnvInference = Join-Path $RepoRoot ".env.inference"
$EnvRoot = Join-Path $RepoRoot ".env"
$ActiveEnvFile = if (Test-Path $EnvInference) { $EnvInference } elseif (Test-Path $EnvRoot) { $EnvRoot } else { $null }

if ($ActiveEnvFile) {
    Write-Host "[i] Loading environment from $ActiveEnvFile" -ForegroundColor DarkGray
    Get-Content $ActiveEnvFile | ForEach-Object {
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

# 2. Configure Execution Mode
if ($Real) {
    $env:KRISNA_USE_REAL_BACKENDS = "1"
    Write-Host "[!] Mode: REAL MODEL BACKENDS (GPU + weights required)" -ForegroundColor Yellow
    if ($LowVram) {
        $env:KRISNA_LOW_VRAM_MODE = "1"
        $env:KRISNA_RAM_ENVELOPE_GB = [string]$RamEnvelopeGb
        if ($VramEnvelopeGb -gt 0) {
            $env:KRISNA_VRAM_ENVELOPE_GB = [string]$VramEnvelopeGb
        } else {
            $env:KRISNA_VRAM_ENVELOPE_GB = "12.0"
        }
        Write-Host "    Low-VRAM mode active: VRAM=$($env:KRISNA_VRAM_ENVELOPE_GB)GB, RAM Offload=$($env:KRISNA_RAM_ENVELOPE_GB)GB" -ForegroundColor Magenta
    } else {
        if ($VramEnvelopeGb -gt 0) {
            $env:KRISNA_VRAM_ENVELOPE_GB = [string]$VramEnvelopeGb
        } else {
            $env:KRISNA_VRAM_ENVELOPE_GB = "48.0"
        }
        Write-Host "    Targeting high-capacity envelope: VRAM=$($env:KRISNA_VRAM_ENVELOPE_GB)GB, RAM=$RamEnvelopeGb GB" -ForegroundColor Green
    }
} else {
    Write-Host "[i] Mode: MockBackend (safe, zero GPU/weights needed)" -ForegroundColor DarkCyan
}

# 3. Resolve Python interpreter
$PythonBin = "python"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
    Write-Host "[i] Using virtualenv Python: $PythonBin" -ForegroundColor DarkGray
}

# Ensure PYTHONPATH contains monorepo package roots
$InfSrc = Join-Path $RepoRoot "inference\src"
$TrainSrc = Join-Path $RepoRoot "training\src"
$env:PYTHONPATH = "$InfSrc;$TrainSrc;$env:PYTHONPATH"

# 4. Handle CLI Session mode
if ($CliSession) {
    Write-Host "[+] Launching Krisna Agentic Session CLI..." -ForegroundColor Cyan
    $CliArgs = @("-m", "krisna_inference.cli")
    if ($Real) { $CliArgs += "--real" }
    if ($LowVram) { $CliArgs += "--low-vram" }
    if ($RemainingArgs) { $CliArgs += $RemainingArgs }
    & $PythonBin $CliArgs
    exit $LASTEXITCODE
}

# 5. Handle Frontend concurrent launch if requested
if ($WithFrontend) {
    $FrontendDir = Join-Path $RepoRoot "inference\frontend"
    if (-not (Test-Path (Join-Path $FrontendDir "package.json"))) {
        $FrontendDir = Join-Path $RepoRoot "inference-frontend"
    }
    if (Test-Path (Join-Path $FrontendDir "package.json")) {
        Write-Host "[+] Starting Node.js Frontend in parallel on port 3100..." -ForegroundColor Cyan
        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "Set-Location '$FrontendDir'; npm start"
    } else {
        Write-Host "[!] Frontend directory not found at $FrontendDir" -ForegroundColor Yellow
    }
}


# 6. Preflight hardware check for real backend mode
if ($Real) {
    Write-Host "[+] Running hardware & deployment verification..." -ForegroundColor Green
    $checkArgs = @("$RepoRoot\scripts\inference\check_hardware.py", "--require-gpu")
    if ($LowVram) { $checkArgs += "--low-vram" }
    & $PythonBin @checkArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Hardware verification failed for real backends. Halting." -ForegroundColor Red
        exit 1
    }
}

# 7. Launch Backend Service
Write-Host "[+] Starting Krisna Inference Service on http://$($BindHost):$Port..." -ForegroundColor Green
& "$RepoRoot\scripts\inference\run_service.ps1" -BindHost $BindHost -Port $Port -RemainingArgs $RemainingArgs
