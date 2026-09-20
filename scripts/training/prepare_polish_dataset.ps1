<#
.SYNOPSIS
    Prepare Polish tier image dataset with metadata.jsonl for DreamBooth LoRA.
.PARAMETER ImageDir
    Directory containing input images.
.PARAMETER OutputDir
    Destination directory.
.PARAMETER Captions
    Path to captions file (.json or .jsonl).
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ImageDir,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir,

    [string]$Captions = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

$Python = "python"
if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") {
    $Python = "$RepoRoot\.venv\Scripts\python.exe"
}

Write-Host "Preparing polish-tier dataset: $ImageDir -> $OutputDir..." -ForegroundColor Cyan

$pyCode = @'
import sys
from krisna_training.polish.prepare_dataset import prepare, count_prepared

image_dir, output_dir, captions_path = sys.argv[1:4]
out = prepare(
    image_dir=image_dir,
    output_dir=output_dir,
    captions=captions_path or None,
    use_shared_instance_prompt="a UI design"
)
print(f"Prepared {count_prepared(out)} images at {out}")
'@

& $Python -c $pyCode $ImageDir $OutputDir $Captions
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Polish dataset prep failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[SUCCESS] Done. Dataset ready at $OutputDir" -ForegroundColor Green
