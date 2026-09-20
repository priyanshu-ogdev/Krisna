<#
.SYNOPSIS
    Download boris/vqgan_f16_16384 pretrained checkpoint and config for Windows.
.DESCRIPTION
    Downloads model.yaml and last.ckpt into checkpoints/vqgan/ for MaskGIT sketch tokenization.
.PARAMETER Destination
    Target directory (default: ./checkpoints/vqgan).
#>

[CmdletBinding()]
param(
    [string]$Destination = "",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

if (-not $Destination) {
    $Destination = "$RepoRoot\checkpoints\vqgan"
}

if ($CheckOnly) {
    $yamlPath = "$Destination\model.yaml"
    $ckptPath = "$Destination\last.ckpt"
    $hasYaml = Test-Path $yamlPath
    $hasCkpt = Test-Path $ckptPath
    if ($hasYaml -and $hasCkpt) {
        Write-Host "[OK] VQGAN checkpoint and config exist in $Destination" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "[MISSING] VQGAN files not fully present in $Destination (model.yaml=$hasYaml, last.ckpt=$hasCkpt)" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}

$yamlUrl = "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/configs/model.yaml&dl=1"
$ckptUrl = "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/ckpts/last.ckpt&dl=1"

$yamlPath = "$Destination\model.yaml"
$ckptPath = "$Destination\last.ckpt"

Write-Host "Downloading boris/vqgan_f16_16384 weights..." -ForegroundColor Cyan

if (-not (Test-Path $yamlPath)) {
    Write-Host "  Fetching model.yaml -> $yamlPath" -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $yamlUrl -OutFile $yamlPath
} else {
    Write-Host "  model.yaml already exists." -ForegroundColor Green
}

if (-not (Test-Path $ckptPath)) {
    Write-Host "  Fetching last.ckpt -> $ckptPath (~3.8GB, please wait)..." -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $ckptUrl -OutFile $ckptPath
} else {
    Write-Host "  last.ckpt already exists." -ForegroundColor Green
}

Write-Host "`n[SUCCESS] VQGAN weights ready in $Destination" -ForegroundColor Green
