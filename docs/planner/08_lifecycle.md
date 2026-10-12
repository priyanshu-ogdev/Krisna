# Stage 08 — lifecycle and memory

## Current behavior

The orchestrator unloads Planner and Sketch before Polish or Critic operations,
then reloads the baseline. This means the current system does not keep Planner
resident through finalize/critique.

## Modes to measure

1. cold unload/reload;
2. persistent process with unloaded weights;
3. vLLM level-1 sleep/wake on Linux/WSL2;
4. native llama.cpp unload/reload on Windows.

Record:

```text
load_start, load_ready, first_token, request_complete,
sleep_start, sleep_ready, wake_start, wake_ready,
unload_start, unload_complete, declared/actual VRAM/RAM
```

## Gate

Choose the lifecycle mode by measured post-finalize latency and memory margin,
not by generic cold-start claims. vLLM sleep mode is only enabled on a tested
CUDA/HIP deployment with enough system RAM.
