# =============================================================================
# Krivex Inference -- Unified GPU Container with Isolated Critic Tier
# (+ optional isolated vLLM Planner tier, see INSTALL_VLLM_PLANNER below)
# Base: NVIDIA CUDA 12.4.1 Runtime on Ubuntu 22.04 LTS
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="Krivex AI Team"
LABEL description="Unified GPU inference container for Krivex with multi-tier venv isolation"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv-inference/bin:/usr/local/cuda/bin:$PATH \
    KRIVEX_CRITIC_VENV_PYTHON=/opt/venv-critic/bin/python \
    KRIVEX_PLANNER_VLLM_VENV_PYTHON=/opt/venv-planner/bin/python \
    KRIVEX_USE_REAL_BACKENDS=1 \
    KRIVEX_BIND_HOST=0.0.0.0 \
    KRIVEX_PORT=8420 \
    KRIVEX_BLOB_ROOT=/app/data/blobs \
    KRIVEX_SESSIONS_DB=/app/data/krivex_sessions.db \
    KRIVEX_PREFERENCE_DB=/app/data/krivex_preference_pairs.db

# OPT-IN third venv for the vLLM Planner backend
ARG INSTALL_VLLM_PLANNER=true

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

# 2. Build isolated virtual environment 1: /opt/venv-inference (Main Inference Tier)
RUN python3.11 -m venv /opt/venv-inference \
    && /opt/venv-inference/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv-inference/bin/pip install --no-cache-dir \
        torch>=2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Build isolated virtual environment 2: /opt/venv-critic (Gemma-4 31B Dense + Unsloth)
RUN python3.11 -m venv /opt/venv-critic \
    && /opt/venv-critic/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv-critic/bin/pip install --no-cache-dir \
        torch>=2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 3b. Build isolated virtual environment 3: /opt/venv-planner (OPT-IN vLLM Planner backend)
RUN if [ "$INSTALL_VLLM_PLANNER" = "true" ]; then \
        python3.11 -m venv /opt/venv-planner \
        && /opt/venv-planner/bin/pip install --no-cache-dir --upgrade pip setuptools wheel; \
    fi

# 4. Copy dependency specifications and install packages into respective venvs
COPY inference/requirements-inference.txt /tmp/requirements-inference.txt
RUN /opt/venv-inference/bin/pip install --no-cache-dir -r /tmp/requirements-inference.txt

COPY inference/requirements-critic.txt /tmp/requirements-critic.txt
RUN /opt/venv-critic/bin/pip install --no-cache-dir -r /tmp/requirements-critic.txt

COPY inference/requirements-planner.txt /tmp/requirements-planner.txt
RUN if [ "$INSTALL_VLLM_PLANNER" = "true" ]; then \
        /opt/venv-planner/bin/pip install --no-cache-dir -r /tmp/requirements-planner.txt; \
    fi

# 5. Copy repo source code into container
COPY . /app

# 6. Install krivex_inference and krivex_training in editable mode inside /opt/venv-inference
RUN /opt/venv-inference/bin/pip install --no-cache-dir -e /app/inference --no-deps \
    && /opt/venv-inference/bin/pip install --no-cache-dir -e /app/training --no-deps

# Make entrypoint script executable
RUN chmod +x /app/inference/entrypoint.sh

# Persistent volume targets for models, checkpoints, and data
VOLUME ["/app/models", "/app/checkpoints", "/app/data"]

EXPOSE 8420

ENTRYPOINT ["/app/inference/entrypoint.sh"]
CMD ["python", "-m", "krivex_inference.orchestrator.service"]
