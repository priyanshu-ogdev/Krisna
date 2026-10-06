Write-Host "Setting up Krisna Data-Forge Environment (Windows)..." -ForegroundColor Cyan
Write-Host "IMPORTANT: vLLM has no official native-Windows build (the vLLM team's" -ForegroundColor Yellow
Write-Host "own guidance is WSL2). Tier-1, Tier-2, OCR, and the audit pass all go" -ForegroundColor Yellow
Write-Host "through vLLM, so most of this pipeline's GPU-model stages will NOT run" -ForegroundColor Yellow
Write-Host "under plain native-Windows PowerShell. Run this from Windows 11 + WSL2" -ForegroundColor Yellow
Write-Host "(Ubuntu) for anything past the deterministic stages, or pin/adopt an" -ForegroundColor Yellow
Write-Host "unofficial native-Windows vLLM fork explicitly if you need one." -ForegroundColor Yellow

# 1. Create and activate Conda environment
conda create -n krisna-forge python=3.10 -y
conda activate krisna-forge

# 2. Install FAISS via Conda
# faiss-gpu is published for Linux x86-64 only (CUDA 11.4/12.1 builds) across
# every channel (pytorch, nvidia, conda-forge) — conda has the SAME
# Linux-only restriction as pip, it is not a workaround for it. Native
# Windows only ever gets faiss-cpu, so Stage 2 dedup runs CPU-only here.
# Budget the extra wall-clock time at "millions of images" scale, or move
# Stage 2 (and other GPU-bound stages) to WSL2/Linux.
conda install -c pytorch faiss-cpu -y

# 3. Install PyTorch with Windows CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install Core Dependencies
# vLLM is intentionally NOT installed here — see the warning above. Installing
# it via plain `pip install vllm` on native Windows will not give you a
# working server; do that step inside WSL2 instead.
pip install openai pydantic click pyyaml jsonschema structlog rich httpx Pillow
pip install -e .[dev]

Write-Host "Environment setup complete. Run: conda activate krisna-forge" -ForegroundColor Green
Write-Host "Remember: vLLM-backed stages need WSL2 — see the warning above." -ForegroundColor Yellow
