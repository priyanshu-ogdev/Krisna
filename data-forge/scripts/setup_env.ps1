Write-Host "Setting up Krisna Data-Forge Environment (Windows)..." -ForegroundColor Cyan

# 1. Create and activate Conda environment (Recommended for Windows FAISS/CUDA)
conda create -n krisna-forge python=3.10 -y
conda activate krisna-forge

# 2. Install FAISS via Conda (Bypasses pip Linux-only wheel issue)
conda install -c pytorch faiss-gpu -y

# 3. Install PyTorch with Windows CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install vLLM and Core Dependencies
pip install vllm openai pydantic click pyyaml jsonschema structlog rich httpx Pillow
pip install -e .[dev]

Write-Host "Environment setup complete. Run: conda activate krisna-forge" -ForegroundColor Green
