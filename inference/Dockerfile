# =============================================================================
# Krivex Inference -- Unified GPU Container
# Base: NVIDIA CUDA 12.4.1 Runtime on Ubuntu 22.04 LTS
# Architecture: Single Resident Model / Qwen Foundation Stack
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="Krivex AI Team"
LABEL description="Unified GPU inference container for Krivex (Qwen3.5-9B + Qwen-Image-2.1)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv-inference/bin:/usr/local/cuda/bin:$PATH \
    KRIVEX_USE_REAL_BACKENDS=1 \
    KRIVEX_LOW_VRAM_MODE=1 \
    KRIVEX_VRAM_ENVELOPE_GB=12.0 \
    KRIVEX_RAM_ENVELOPE_GB=16.0 \
    KRIVEX_BIND_HOST=0.0.0.0 \
    KRIVEX_PORT=8420 \
    KRIVEX_BLOB_ROOT=/app/data/blobs \
    KRIVEX_SESSIONS_DB=/app/data/krivex_sessions.db \
    KRIVEX_PREFERENCE_DB=/app/data/krivex_preference_pairs.db

# 1. Install OS packages, software-properties-common, Python 3.11, and graphics libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Point default python3 and python to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# 2. Build single unified virtual environment: /opt/venv-inference
RUN python3.11 -m venv /opt/venv-inference \
    && /opt/venv-inference/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv-inference/bin/pip install --no-cache-dir \
        torch>=2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Copy unified dependency specification and install packages
COPY inference/requirements.txt /tmp/requirements.txt
RUN /opt/venv-inference/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Copy repo source code into container
COPY . /app

# 5. Install krivex_inference and krivex_training in editable mode inside /opt/venv-inference
RUN /opt/venv-inference/bin/pip install --no-cache-dir -e /app/inference --no-deps \
    && /opt/venv-inference/bin/pip install --no-cache-dir -e /app/training --no-deps

# Make entrypoint script executable
RUN chmod +x /app/inference/entrypoint.sh

# Persistent volume targets for models, checkpoints, and data
VOLUME ["/app/models", "/app/checkpoints", "/app/data"]

EXPOSE 8420

ENTRYPOINT ["/app/inference/entrypoint.sh"]
CMD ["python", "-m", "krivex_inference.orchestrator.service"]
