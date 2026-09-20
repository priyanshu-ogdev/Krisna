# =============================================================================
# scripts/docker/run_docker.ps1
# Host preflight check and launcher for Krisna Docker services (PowerShell)
#
# Usage:
#   .\scripts\docker\run_docker.ps1            # Launch inference + frontend
#   .\scripts\docker\run_docker.ps1 -Build     # Build containers first
#   .\scripts\docker\run_docker.ps1 -LowVram   # Launch with 12GB offloading
#   .\scripts\docker\run_docker.ps1 -Down      # Stop services
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Detach,
    [switch]$LowVram,
    [switch]$Down
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
Write-Host "       Krisna Docker Deployment & Preflight (PowerShell)        " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# 1. Verify Docker installation
$DockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $DockerCmd) {
    Write-Host "[ERROR] Docker is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Install Docker Desktop for Windows: https://docs.docker.com/desktop/setup/install/windows-install/" -ForegroundColor Red
    exit 1
}

# 2. Check if Docker daemon is running
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[ERROR] Failed to query Docker daemon. Please ensure Docker Desktop is running." -ForegroundColor Red
    exit 1
}

# 3. Handle stop action
if ($Down) {
    Write-Host "[+] Stopping Krisna Docker services..." -ForegroundColor Yellow
    docker compose down
    exit 0
}

# 4. Check for GPU support in Docker Desktop
Write-Host "[+] Probing Docker WSL2 GPU acceleration..." -ForegroundColor Green
$gpuFound = $false
if ($dockerInfo -match "nvidia") {
    $gpuFound = $true
    Write-Host "[i] NVIDIA runtime detected in Docker info." -ForegroundColor DarkGray
} else {
    Write-Host "[!] Note: Ensure Docker Desktop Settings -> Resources -> WSL2 engine is enabled for GPU acceleration." -ForegroundColor Yellow
}

# 5. Low-VRAM setting
if ($LowVram) {
    $env:KRISNA_LOW_VRAM_MODE = "1"
    Write-Host "[i] Low-VRAM mode enabled (target: 12GB envelope)." -ForegroundColor Green
}

# 6. Compose command assembly
$composeArgs = @("compose", "up")
if ($Build) {
    $composeArgs += "--build"
}
if ($Detach) {
    $composeArgs += "-d"
}

Write-Host "[+] Starting services: docker $($composeArgs -join ' ')..." -ForegroundColor Green
& docker @composeArgs

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Krisna services started." -ForegroundColor Cyan
Write-Host "Inference API : http://localhost:8420 (health: http://localhost:8420/healthz)" -ForegroundColor Cyan
Write-Host "Web Studio UI : http://localhost:3000" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
