<#
.SYNOPSIS
    Krisna Model Training Pipeline Launcher for Windows (PowerShell)
    Target Hardware: Intel Core i9, NVIDIA RTX A6000 (48GB VRAM), 128GB RAM.

.DESCRIPTION
    Native PowerShell dispatcher for the current Krisna training contract.
    Only the Qwen-Image-2.1 paired image-edit LoRA path is an active intended
    training path. Planner is frozen, Layout IR is deterministic, and the
    historical Sketch/DPO launchers are not source-backed active trainers.

.PARAMETER All
    Execute the active Qwen-Image-2.1 supervised Polish tier after pre-flight.

.PARAMETER Tier
    Specific tier to run: 'sketch-stage1', 'sketch-stage2', 'polish-default', 'polish-dpo', 'planner', 'critic'.

.PARAMETER Config
    Optional custom YAML configuration path.

.PARAMETER Sync
    Run Data-Forge -> Training bridge sync (sync_to_training.py) before training.

.PARAMETER DataRoot
    Root directory where Data-Forge exported data lives (default: $env:DATA_ROOT or D:\data_krisna).

.PARAMETER DryRun
    Perform pre-flight verification without executing training commands.

.PARAMETER Log
    Path to tee output logs (default: logs/training/<tier>_<timestamp>.log).

.PARAMETER List
    List all model tiers, active status, and architectural design rationale.

.EXAMPLE
    .\scripts\training\train_all.ps1 -List
    .\scripts\training\train_all.ps1 -All -DryRun
    .\scripts\training\train_all.ps1 -Sync -DataRoot "D:\data_krisna" -All
    .\scripts\training\train_all.ps1 -Tier sketch-stage1
#>

[CmdletBinding()]
param(
    [switch]$All,
    [string]$Tier = "",
    [string]$Config = "",
    [switch]$Sync,
    [string]$DataRoot = "",
    [switch]$DryRun,
    [string]$Log = "",
    [switch]$List,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RepoRoot

function Show-Help {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
}

function Show-Tiers {
    Write-Host @'
Tier            Status       Design intent (see docs/training/ARCHITECTURE.md)
--------------  -----------  ------------------------------------------------
sketch-stage1   RESEARCH     Deterministic Layout IR is the current baseline; no complete
                              source-backed learned Sketch trainer is present.

sketch-stage2   RESEARCH     Same: keep as a future LayoutFlow/MaskGIT ablation only.

polish-default  PENDING      Qwen/Qwen-Image-2.1 paired image-conditioned LoRA. The
                              local code prepares pairs and delegates to an external
                              trainer that must be pinned and one-batch validated.

polish-dpo      DEFERRED     Historical path. It is not part of the active Qwen training
                              pipeline and must not be launched by -All.

planner         FROZEN       Qwen3.5-9B GGUF + UICrit RAG. No Planner LoRA training.

critic          FROZEN       No Critic training in the active no-RLHF pipeline.
'@ -ForegroundColor Cyan
}

if ($Help) {
    Show-Help
    exit 0
}

if ($List -or (-not $All -and -not $Tier -and -not $Sync)) {
    Show-Tiers
    exit 0
}

# Resolve Python executable
$Python = "python"
if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") {
    $Python = "$RepoRoot\.venv\Scripts\python.exe"
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"

# GPU & System Check
function Check-Hardware {
    Write-Host "Checking hardware environment..." -ForegroundColor DarkGray
    try {
        $smi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        if ($smi) {
            $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            Write-Host "  [GPU] $gpuInfo" -ForegroundColor Green
        } else {
            Write-Host "  [WARNING] nvidia-smi not found. Training requires an NVIDIA GPU (e.g. RTX A6000 48GB)." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  [WARNING] Could not probe GPU info: $_" -ForegroundColor Yellow
    }
}

# Pre-flight checks
function Test-Preflight {
    param([string]$StageName, [string]$ConfigPath)

    $failed = $false
    switch ($StageName) {
        "sketch-stage1" {
            $m = "$RepoRoot\data\sketch_train_256\manifest.jsonl"
            if (-not (Test-Path $m)) {
                Write-Host "  [PRE-FLIGHT FAILED] Manifest not found: $m" -ForegroundColor Red
                Write-Host "    Run data sync first: .\scripts\training\train_all.ps1 -Sync -DataRoot <DATA_ROOT>" -ForegroundColor Yellow
                $failed = $true
            } else {
                Write-Host "  [PRE-FLIGHT OK] Manifest found: $m" -ForegroundColor Green
            }
        }
        "sketch-stage2" {
            $m = "$RepoRoot\data\sketch_train_512\manifest.jsonl"
            $initCkpt = "$RepoRoot\checkpoints\sketch_stage1_256\checkpoint_final.pt"
            if (-not (Test-Path $m)) {
                Write-Host "  [PRE-FLIGHT FAILED] Manifest not found: $m" -ForegroundColor Red
                Write-Host "    Run data sync first: .\scripts\training\train_all.ps1 -Sync -DataRoot <DATA_ROOT>" -ForegroundColor Yellow
                $failed = $true
            } else {
                Write-Host "  [PRE-FLIGHT OK] Manifest found: $m" -ForegroundColor Green
            }
            if (-not (Test-Path $initCkpt)) {
                if ($All) {
                    Write-Host "  [PRE-FLIGHT NOTE] Stage 1 checkpoint ($initCkpt) will be generated during Stage 1." -ForegroundColor DarkYellow
                } else {
                    Write-Host "  [PRE-FLIGHT WARNING] Stage 1 checkpoint not found: $initCkpt" -ForegroundColor Yellow
                    if (-not $DryRun) { $failed = $true }
                }
            } else {
                Write-Host "  [PRE-FLIGHT OK] Stage 1 checkpoint found: $initCkpt" -ForegroundColor Green
            }
        }
        "polish-default" {
            $d = "$RepoRoot\data\polish_qwen_image21_pairs"
            if (-not (Test-Path $d)) {
                Write-Host "  [PRE-FLIGHT FAILED] Polish training directory not found: $d" -ForegroundColor Red
                Write-Host "    Build the condition/target pair manifest first; see docs/training/TRAINING_RUNBOOK.md" -ForegroundColor Yellow
                $failed = $true
            } else {
                $manifest = Join-Path $d "manifest.jsonl"
                $receipt = Join-Path $d "dataset_receipt.json"
                if (-not (Test-Path $manifest) -or -not (Test-Path $receipt)) {
                    Write-Host "  [PRE-FLIGHT FAILED] Qwen pair manifest/receipt missing under $d" -ForegroundColor Red
                    Write-Host "    A directory alone is not a training corpus; run the audited pair preparation/export first." -ForegroundColor Yellow
                    $failed = $true
                } else {
                    $rows = @(Get-Content -LiteralPath $manifest | Where-Object { $_.Trim().Length -gt 0 })
                    $receiptJson = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
                    $pairCount = [int]($receiptJson.pair_count)
                    if ($rows.Count -eq 0 -or $pairCount -le 0 -or $rows.Count -ne $pairCount) {
                        Write-Host "  [PRE-FLIGHT FAILED] Qwen pair receipt is empty or inconsistent (rows=$($rows.Count), receipt=$pairCount)" -ForegroundColor Red
                        $failed = $true
                    } else {
                        Write-Host "  [PRE-FLIGHT OK] Qwen pair corpus verified: $pairCount pairs" -ForegroundColor Green
                    }
                }
            }
        }
        "polish-dpo" {
            $db = "$RepoRoot\krisna_preference_pairs.db"
            if (-not (Test-Path $db)) {
                Write-Host "  [PRE-FLIGHT FAILED] Preference pairs DB not found: $db" -ForegroundColor Red
                Write-Host "    Run data sync first: .\scripts\training\train_all.ps1 -Sync -DataRoot <DATA_ROOT>" -ForegroundColor Yellow
                $failed = $true
            } else {
                Write-Host "  [PRE-FLIGHT OK] Preference pairs DB found: $db" -ForegroundColor Green
            }
        }
    }
    return (-not $failed)
}

# Bridge Sync
if ($Sync) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "  RUNNING DATA-FORGE -> TRAINING DATA SYNC" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $syncArgs = @("$RepoRoot\scripts\data-forge\sync_to_training.py")
    if ($DataRoot) {
        $syncArgs += @("--data-root", $DataRoot)
    }

    if ($DryRun) {
        $syncStr = $syncArgs -join " "
        Write-Host "  [DRY RUN] Would execute: $Python $syncStr" -ForegroundColor Magenta
    } else {
        & $Python @syncArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERROR] Data sync failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }

    if (-not $All -and -not $Tier) {
        Write-Host "Sync completed successfully." -ForegroundColor Green
        exit 0
    }
}

Check-Hardware

# Setup Logging
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ($Log -eq "auto" -or ($Log -eq "" -and -not $DryRun)) {
    $logDir = "$RepoRoot\logs\training"
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    if ($All) {
        $Log = "$logDir\all_stages_$Timestamp.log"
    } else {
        $Log = "$logDir\${Tier}_$Timestamp.log"
    }
}

function Execute-Tier {
    param([string]$StageName, [string]$CustomConfig)

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "  LAUNCHING TRAINING TIER: $StageName" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    if ($StageName -eq "planner" -or $StageName -eq "critic") {
        Write-Host "  [WARNING] $StageName is DEPRECATED under PRD section 6 (no-RLHF revision)." -ForegroundColor Red
        Write-Host "  The live inference stack keeps $StageName frozen." -ForegroundColor Red
        $confirm = Read-Host "Proceed anyway? (y/N)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Host "Aborted." -ForegroundColor Yellow
            return
        }
    }

    $ok = Test-Preflight $StageName $CustomConfig
    if (-not $ok) {
        if ($DryRun) {
            Write-Host "  [DRY RUN NOTE] Pre-flight would fail in a live run." -ForegroundColor DarkYellow
        } else {
            Write-Host "  Pre-flight check failed for $StageName. Aborting." -ForegroundColor Red
            exit 1
        }
    }

    switch ($StageName) {
        "sketch-stage1" {
            Write-Host "  [RESEARCH ONLY] No source-backed Sketch trainer exists in this checkout." -ForegroundColor Yellow
            if (-not $DryRun) { exit 2 }
            return
        }
        "sketch-stage2" {
            Write-Host "  [RESEARCH ONLY] No source-backed Sketch trainer exists in this checkout." -ForegroundColor Yellow
            if (-not $DryRun) { exit 2 }
            return
        }
        "polish-default" {
            $defaultCfg = "training/configs/polish_qwen_image21_lora.yaml"
            $cfg = if ($CustomConfig) { $CustomConfig } else { $defaultCfg }
            $pyScript = @'
import subprocess, sys, yaml
from pathlib import Path
config_path = sys.argv[1]
cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
diffusers_dir = Path(cfg.pop('diffusers_dir'))
script = diffusers_dir / 'examples' / 'dreambooth' / 'train_dreambooth_lora_qwenimage21_img2img.py'
if not script.exists():
    raise SystemExit(f'Qwen image-edit trainer not found: {script}. Pin/provide the trainer before launching.')
args = [sys.executable, '-m', 'accelerate.commands.launch', str(script)]
for k, v in cfg.items():
    if v is None: continue
    if isinstance(v, bool):
        if v: args.append(f'--{k}')
        continue
    args.append(f'--{k}={v}')
print('Running:', ' '.join(args))
subprocess.run(args, check=True)
'@
            if ($DryRun) {
                Write-Host "  [DRY RUN] Would launch diffusers dreambooth LoRA via accelerate using $cfg" -ForegroundColor Magenta
                return
            }
            & $Python -c $pyScript $cfg 2>&1 | Tee-Object -FilePath $Log -Append
        }
        "polish-dpo" {
            Write-Host "  [DEFERRED] polish-dpo is not part of the active Qwen-Image-2.1 pipeline." -ForegroundColor Yellow
            Write-Host "  Build and validate supervised Polish first; see docs/training/ARCHITECTURE.md." -ForegroundColor Yellow
            if (-not $DryRun) { exit 2 }
            return
        }
        default {
            Write-Host "Unknown tier: $StageName" -ForegroundColor Red
            Show-Tiers
            exit 1
        }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n  [ERROR] Stage $StageName failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "  [COMPLETE] Stage $StageName finished successfully." -ForegroundColor Green
}

# Main Execution Loop
if ($All) {
    $stages = @("polish-default")
    $total = $stages.Count
    $idx = 1

    Write-Host "`n============================================================" -ForegroundColor Magenta
    Write-Host (" KRISNA FULL SEQUENTIAL TRAINING PIPELINE ({0} stages)" -f $total) -ForegroundColor Magenta
    if ($DryRun) { Write-Host " [DRY RUN MODE - NO COMMANDS WILL ACTUALLY BE EXECUTED]" -ForegroundColor Yellow }
    if ($Log) { Write-Host " Pipeline log destination: $Log" -ForegroundColor Cyan }
    Write-Host "============================================================" -ForegroundColor Magenta

    foreach ($stg in $stages) {
        Write-Host ("`n>>> Stage {0}/{1}: {2} <<<" -f $idx, $total, $stg) -ForegroundColor Yellow
        Execute-Tier $stg ""
        $idx++
    }

    Write-Host "`n============================================================" -ForegroundColor Green
    Write-Host " ALL ACTIVE TRAINING TIERS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    exit 0
}

if ($Tier) {
    Execute-Tier $Tier $Config
    exit 0
}
