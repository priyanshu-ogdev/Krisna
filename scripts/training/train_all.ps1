<#
.SYNOPSIS
    Krisna Model Training Pipeline Launcher for Windows (PowerShell)
    Target Hardware: Intel Core i9, NVIDIA RTX A6000 (48GB VRAM), 128GB RAM.

.DESCRIPTION
    Native PowerShell training dispatcher for all active Krisna model tiers:
      1. sketch-stage1   (MaskGIT transformer, 256px resolution)
      2. sketch-stage2   (MaskGIT transformer, 512px resolution, continued from stage 1)
      3. polish-default  (Z-Image-Turbo DreamBooth LoRA base fine-tuning)
      4. polish-dpo      (Diffusion-DPO flow-matching preference alignment)

.PARAMETER All
    Execute all 4 active training tiers sequentially in dependency order.

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
Tier            Status       Design intent (per PRD & docs/review/)
--------------  -----------  ------------------------------------------------
sketch-stage1   ACTIVE       MaskGIT transformer, from scratch, 256px/16x16 grid.
                             batch=32, lr=3e-4, 1000-step warmup, adam_beta1=0.9,
                             adam_beta2=0.96. caption_mix_ratio=0.95, cfg_dropout=0.1.
                             num_workers=8. No gradient checkpointing needed at 256px.

sketch-stage2   ACTIVE       MaskGIT transformer, progressive init from stage1, 512px/32x32.
                             batch=16, lr=1.5e-4, adam_beta1=0.9, adam_beta2=0.96, num_workers=6.
                             use_gradient_checkpointing=true -- REQUIRED safeguard at 1024 tokens.

polish-default  ACTIVE       Z-Image-Turbo LoRA (rank=32, alpha=rank) via official diffusers
                             dreambooth-LoRA script. Base fine-tune step at resolution=1024.

polish-dpo      ACTIVE       Flow-matching Diffusion-DPO (beta=2000, candidates [500, 1000, 2000]).
                             Reference model CPU-offloaded for 48GB VRAM envelope.
                             Resolution=512 deliberate for compute efficiency.

planner         DEPRECATED   Frozen at inference under final no-RLHF PRD revision (RAG over
                             UICrit corpus). Training produces an unconsumed adapter.

critic          DEPRECATED   Frozen, zero-shot at inference under final no-RLHF PRD revision.
                             Training produces an unconsumed adapter.
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
            $d = "$RepoRoot\data\polish_default_train"
            if (-not (Test-Path $d)) {
                Write-Host "  [PRE-FLIGHT FAILED] Polish training directory not found: $d" -ForegroundColor Red
                Write-Host "    Run data sync first: .\scripts\training\train_all.ps1 -Sync -DataRoot <DATA_ROOT>" -ForegroundColor Yellow
                $failed = $true
            } else {
                Write-Host "  [PRE-FLIGHT OK] Polish training dir found: $d" -ForegroundColor Green
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
            $defaultCfg = if (Test-Path "$RepoRoot\training\configs\sketch_stage1_256.yaml") { "training/configs/sketch_stage1_256.yaml" } else { "training/configs/sketch_train_stage1_256.yaml" }
            $cfg = if ($CustomConfig) { $CustomConfig } else { $defaultCfg }
            $cmd = @("-m", "krisna_training.sketch.train", "--config", $cfg)
            if ($DryRun) {
                $cmdStr = $cmd -join " "
                Write-Host "  [DRY RUN] Would execute: $Python $cmdStr" -ForegroundColor Magenta
                return
            }
            $cmdStr = $cmd -join " "
            Write-Host "  Executing: $Python $cmdStr" -ForegroundColor DarkGray
            & $Python @cmd 2>&1 | Tee-Object -FilePath $Log -Append
        }
        "sketch-stage2" {
            $defaultCfg = if (Test-Path "$RepoRoot\training\configs\sketch_stage2_512.yaml") { "training/configs/sketch_stage2_512.yaml" } else { "training/configs/sketch_train_stage2_512.yaml" }
            $cfg = if ($CustomConfig) { $CustomConfig } else { $defaultCfg }
            $cmd = @("-m", "krisna_training.sketch.train", "--config", $cfg)
            if ($DryRun) {
                $cmdStr = $cmd -join " "
                Write-Host "  [DRY RUN] Would execute: $Python $cmdStr" -ForegroundColor Magenta
                return
            }
            $cmdStr = $cmd -join " "
            Write-Host "  Executing: $Python $cmdStr" -ForegroundColor DarkGray
            & $Python @cmd 2>&1 | Tee-Object -FilePath $Log -Append
        }
        "polish-default" {
            $defaultCfg = if (Test-Path "$RepoRoot\training\configs\polish_stage1_default_lora.yaml") { "training/configs/polish_stage1_default_lora.yaml" } else { "training/configs/polish_default_lora_z_image.yaml" }
            $cfg = if ($CustomConfig) { $CustomConfig } else { $defaultCfg }
            $pyScript = @'
import subprocess, sys, yaml
from pathlib import Path
config_path = sys.argv[1]
cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
diffusers_dir = Path(cfg.pop('diffusers_dir'))
script = diffusers_dir / 'examples' / 'dreambooth' / 'train_dreambooth_lora_z_image.py'
if not script.exists():
    raise SystemExit(f'Script not found: {script}')
args = ['accelerate', 'launch', str(script)]
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
            $defaultCfg = if (Test-Path "$RepoRoot\training\configs\polish_stage2_dpo_general.yaml") { "training/configs/polish_stage2_dpo_general.yaml" } else { "training/configs/dpo_z_image_stage1_general.yaml" }
            $cfg = if ($CustomConfig) { $CustomConfig } else { $defaultCfg }
            $pyScript = @'
import subprocess, sys, yaml
from pathlib import Path
config_path = sys.argv[1]
cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
args = ['accelerate', 'launch', '-m', 'krisna_training.polish.train_dpo']
for k, v in cfg.items():
    if v is None: continue
    if isinstance(v, bool):
        if v: args.append(f'--{k}')
        continue
    if isinstance(v, list):
        for item in v: args.append(f'--{k}={item}')
        continue
    args.append(f'--{k}={v}')
print('Running:', ' '.join(args))
subprocess.run(args, check=True)
'@
            if ($DryRun) {
                Write-Host "  [DRY RUN] Would launch Diffusion-DPO via accelerate using $cfg" -ForegroundColor Magenta
                return
            }
            & $Python -c $pyScript $cfg 2>&1 | Tee-Object -FilePath $Log -Append
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
    $stages = @("sketch-stage1", "sketch-stage2", "polish-default", "polish-dpo")
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
