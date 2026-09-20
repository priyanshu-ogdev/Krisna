<#
.SYNOPSIS
    Tokenize an image directory using VQGAN for MaskGIT sketch training.
.DESCRIPTION
    Encodes images into discrete VQ tokens and writes manifest.jsonl.
.PARAMETER ImageDir
    Directory containing input images.
.PARAMETER OutputDir
    Destination directory for tokens/ and manifest.jsonl.
.PARAMETER ImageSize
    Target resolution (256 or 512, default: 256).
.PARAMETER Captions
    Path to captions file (.json or .jsonl).
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ImageDir,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir,

    [int]$ImageSize = 256,

    [string]$Captions = "",

    [string]$VqganDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

if (-not $VqganDir) {
    $VqganDir = "$RepoRoot\checkpoints\vqgan"
}

$Python = "python"
if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") {
    $Python = "$RepoRoot\.venv\Scripts\python.exe"
}

Write-Host "Tokenizing $ImageDir -> $OutputDir at ${ImageSize}px..." -ForegroundColor Cyan

$pyCode = @'
import sys
from krisna_training.sketch.vq_tokenizer import VQTokenizer
from krisna_training.sketch.prepare_dataset import prepare

image_dir, output_dir, image_size, vqgan_dir, captions = sys.argv[1:6]
tokenizer = VQTokenizer(
    checkpoint_path=f"{vqgan_dir}/last.ckpt",
    config_path=f"{vqgan_dir}/model.yaml",
)
manifest_path = prepare(
    image_dir=image_dir,
    output_dir=output_dir,
    tokenizer=tokenizer,
    image_size=int(image_size),
    captions_path=captions or None,
)
print(f"Manifest written to {manifest_path}")
'@

& $Python -c $pyCode $ImageDir $OutputDir $ImageSize $VqganDir $Captions
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Tokenization failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[SUCCESS] Done. Manifest at $OutputDir\manifest.jsonl" -ForegroundColor Green
