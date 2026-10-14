<#
.SYNOPSIS
    Clone and configure the pinned Hugging Face diffusers checkout and Qwen-Image-2.1 DreamBooth dependencies.
.PARAMETER DiffusersDir
    Destination path for diffusers checkout (default: ./third_party/diffusers).
#>

[CmdletBinding()]
param(
    [string]$DiffusersDir = "",
    [string]$DiffusersRevision = "efe46a5b177af67c64b3666da8ddddac7ee12163"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

if (-not $DiffusersDir) {
    $DiffusersDir = "$RepoRoot\third_party\diffusers"
}

Write-Host "Preparing pinned diffusers repository (pinned Qwen-Image-2.1 training script)..." -ForegroundColor Cyan

if (-not (Test-Path $DiffusersDir)) {
    $parent = Split-Path -Parent $DiffusersDir
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    git clone https://github.com/huggingface/diffusers.git "$DiffusersDir"
} else {
    Write-Host "Diffusers already cloned at $DiffusersDir." -ForegroundColor DarkGray
}

$head = (git -C "$DiffusersDir" rev-parse HEAD).Trim()
$dirty = (git -C "$DiffusersDir" status --porcelain --untracked-files=all)
if ($head -ne $DiffusersRevision) {
    if ($dirty) {
        throw "Refusing to change dirty Diffusers checkout at $DiffusersDir (HEAD $head; expected $DiffusersRevision)."
    }
    git -C "$DiffusersDir" fetch origin "$DiffusersRevision"
    git -C "$DiffusersDir" checkout --detach "$DiffusersRevision"
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

Write-Host "Installing pinned training dependencies..." -ForegroundColor DarkGray
& $Pip install "accelerate==1.10.1" "peft==0.17.1" "datasets==4.2.1" "pyarrow==21.0.0"

Write-Host "`n[SUCCESS] Diffusers training environment configured." -ForegroundColor Green
Write-Host "Official Qwen-Image-2.1 LoRA script located at:" -ForegroundColor DarkGray
Write-Host "    $DiffusersDir\examples\dreambooth\train_dreambooth_lora_qwenimage21_img2img.py" -ForegroundColor DarkGray
Write-Host "Run 'accelerate config' if you have not configured accelerate yet." -ForegroundColor Yellow
