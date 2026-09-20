#!/usr/bin/env bash
# Root-level environment setup — orchestrates the existing per-component
# scripts/*/setup_env*.sh files rather than reimplementing them, so a
# single change to how a component installs never needs updating in two
# places. Run from the monorepo root: ./setup.sh [phase ...]
#
# Phases (run all by default, or pass specific ones as args, e.g.
# ./setup.sh base sketch):
#   base     - main .venv: krisna-inference[dev] + krisna-training[dev] +
#              krisna-data-forge (editable installs, CPU-only, no GPU
#              backends yet). Always run this first.
#   backends - adds the real inference backends (Planner/Sketch/Polish
#              Default+Quality) to the SAME .venv — torch/diffusers/
#              transformers(git-main)/bitsandbytes. GPU required to use,
#              not to install.
#   sketch   - adds Sketch-tier training deps to the SAME .venv.
#   polish   - adds Polish/DPO training deps (diffusers+peft) to the SAME
#              .venv, plus clones the diffusers examples this project's
#              dreambooth-LoRA script depends on.
#   critic   - creates a SEPARATE ./venv-critic. Deliberately isolated:
#              Critic needs transformers==5.5.0 exactly (Unsloth's
#              FastVisionModel pin), which directly conflicts with the
#              "backends" extra's transformers@git+main install — the
#              two cannot coexist in one environment. See
#              inference/src/krisna_inference/backends/critic_worker.py's
#              module docstring and inference/pyproject.toml's comment
#              on the "backends" extra for the full reasoning. Planner
#              and Critic training are DEPRECATED under the final
#              no-RLHF-loop design (see their own __init__.py
#              DeprecationWarnings) — this phase still sets up the venv
#              since it's needed for the live Critic INFERENCE backend,
#              not for training it.
#   verifiers - CLIP/OCR/aesthetic verifier-stack deps, same .venv as base.
#   frontend - npm install for inference/frontend/, the Node.js control
#              panel (Setup/Install tab drives scripts/inference/
#              download_weights.py; Studio tab is a session UI over the
#              FastAPI service). Node/npm only — no Python venv touched.
#              NOT in the default phase list below: the backend runs
#              fully without it (curl/any HTTP client talks to the
#              service directly), so a Node toolchain shouldn't be a
#              required part of a Python-only setup. Run explicitly:
#              ./setup.sh frontend
#
# This script only ever calls the existing scripts/*/setup_env*.sh files
# — it does not duplicate their pip install logic, so it can't drift out
# of sync with them the way a hand-rolled parallel install list could.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(base backends sketch polish verifiers critic)
fi

echo -e "\033[1;35m=========================================================\033[0m"
echo -e "\033[1;35mKrisna monorepo setup — phases: ${PHASES[*]}\033[0m"
echo -e "\033[1;35m=========================================================\033[0m"

for phase in "${PHASES[@]}"; do
    case "$phase" in
        base)
            echo -e "\033[1;36m[base] main .venv: inference[dev] + training[dev] + data-forge\033[0m"
            ./scripts/inference/setup_env.sh
            echo -e "\033[1;36m[base] adding krisna-data-forge (editable, --no-deps: shared deps already satisfied above)\033[0m"
            source .venv/bin/activate
            pip install -e "./data-forge" --no-deps
            # data-forge's OWN heavy deps (vllm, faiss, opencv, etc. — see
            # data-forge/pyproject.toml) are intentionally NOT installed
            # here by default: they're large, mostly GPU/platform-specific,
            # and only needed to actually RUN the data pipeline, not to
            # develop against krisna_training/krisna_inference. Install
            # them explicitly when you need to run data-forge for real:
            echo -e "\033[1;33m[base] To run the data pipeline for real: pip install -e \"./data-forge[dev]\" (or without [dev] for a leaner install) — see data-forge/pyproject.toml for the full dependency list; some entries are Linux-only (faiss-gpu).\033[0m"
            ;;
        backends)
            echo -e "\033[1;36m[backends] real inference backends (GPU) into the main .venv\033[0m"
            ./scripts/inference/setup_env_inference.sh ${KRISNA_REQUIRE_GPU:+--require-gpu}
            ;;
        sketch)
            echo -e "\033[1;36m[sketch] Sketch-tier training deps into the main .venv\033[0m"
            source .venv/bin/activate
            pip install -e "./training[sketch]"
            # training[sketch] (torch+pillow) alone is NOT the full sketch
            # training setup — the VQGAN tokenizer path
            # (sync_sketch_tier.py / dataset prep) additionally needs
            # taming-transformers/omegaconf, tracked separately in
            # training/requirements-training.txt rather than folded into
            # the pyproject extra (predates it, kept as-is rather than
            # merged to avoid touching a working, separately-versioned
            # dependency pin — taming-transformers requires omegaconf<2.1
            # specifically).
            ./scripts/training/setup_env_training.sh
            ;;
        polish)
            echo -e "\033[1;36m[polish] Polish/DPO training deps + diffusers examples\033[0m"
            source .venv/bin/activate
            pip install -e "./training[polish]"
            ./scripts/training/setup_env_diffusers_training.sh
            ;;
        critic)
            echo -e "\033[1;36m[critic] separate ./venv-critic (see this script's header for why)\033[0m"
            ./scripts/training/setup_env_critic.sh
            ;;
        verifiers)
            echo -e "\033[1;36m[verifiers] CLIP/OCR/aesthetic verifier-stack deps into the main .venv\033[0m"
            ./scripts/inference/setup_env_verifiers.sh
            ;;
        forge)
            echo -e "\033[1;36m[forge] full data-forge pipeline dependencies into main .venv\033[0m"
            source .venv/bin/activate
            pip install -e "./data-forge[dev]"
            ;;
        frontend)
            echo -e "\033[1;36m[frontend] inference/frontend/ (Node.js control panel) — npm install\033[0m"
            if ! command -v npm >/dev/null 2>&1; then
                echo -e "\033[1;31m[frontend] npm not found — install Node.js 18+ first (https://nodejs.org), then re-run: ./setup.sh frontend\033[0m"
                exit 1
            fi
            (cd inference/frontend && npm install)
            ;;
        *)
            echo -e "\033[1;31mUnknown phase: $phase (expected one of: base forge backends sketch polish critic verifiers frontend)\033[0m"
            exit 1
            ;;
    esac
done

echo -e "\033[1;32m=========================================================\033[0m"
echo -e "\033[1;32mSetup complete.\033[0m"
echo -e "\033[1;32m=========================================================\033[0m"
echo -e "\033[1;33mNext steps:\033[0m"
echo -e "\033[1;33m  Data pipeline:  ./run_data_forge.sh --dry-run\033[0m"
echo -e "\033[1;33m  Training:       ./train.sh --list\033[0m"
echo -e "\033[1;33m  Inference:      ./run_inference.sh\033[0m"
echo -e "\033[1;33m  Control panel:  ./setup.sh frontend && cd inference/frontend && npm start   (optional — a UI over the same API)\033[0m"
echo -e "\033[1;33m  Tests:          ./scripts/inference/run_tests.sh\033[0m"
