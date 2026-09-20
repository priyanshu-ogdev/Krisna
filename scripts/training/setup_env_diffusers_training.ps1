<#
.SYNOPSIS
    Clone and configure Hugging Face diffusers and DreamBooth dependencies for Z-Image LoRA training on Windows.
.PARAMETER DiffusersDir
    Destination path for diffusers checkout (default: ./third_party/diffusers).
#>

[CmdletBinding()]
param(
    [string]$DiffusersDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

if (-not $DiffusersDir) {
    $DiffusersDir = "$RepoRoot\third_party\diffusers"
}

Write-Host "Cloning diffusers repository (official Z-Image training scripts)..." -ForegroundColor Cyan

if (-not (Test-Path $DiffusersDir)) {
    $parent = Split-Path -Parent $DiffusersDir
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    git clone https://github.com/huggingface/diffusers.git "$DiffusersDir"
} else {
    Write-Host "Diffusers already cloned at $DiffusersDir. Pulling latest..." -ForegroundColor DarkGray
    git -C "$DiffusersDir" pull
}

$Pip = "$RepoRoot\.venv\Scripts\pip.exe"
if (-not (Test-Path $Pip)) {
    $Pip = "pip"
}

Write-Host "Installing diffusers in editable mode..." -ForegroundColor DarkGray
& $Pip install -e "$DiffusersDir"

$dreamboothReq = "$DiffusersDir\examples\dreambooth\requirements.txt"
if (Test-Path $dreamboothReq) {
    Write-Host "Installing DreamBooth requirements..." -ForegroundColor DarkGray
    & $Pip install -r "$dreamboothReq"
}

$zimageReq = "$DiffusersDir\examples\dreambooth\requirements_z_image.txt"
if (Test-Path $zimageReq) {
    Write-Host "Installing Z-Image specific requirements..." -ForegroundColor DarkGray
    & $Pip install -r "$zimageReq"
}

Write-Host "Installing accelerate and peft..." -ForegroundColor DarkGray
& $Pip install accelerate peft

Write-Host "`n[SUCCESS] Diffusers training environment configured." -ForegroundColor Green
Write-Host "Official Z-Image LoRA script located at:" -ForegroundColor DarkGray
Write-Host "    $DiffusersDir\examples\dreambooth\train_dreambooth_lora_z_image.py" -ForegroundColor DarkGray
Write-Host "Run 'accelerate config' if you have not configured accelerate yet." -ForegroundColor Yellow
