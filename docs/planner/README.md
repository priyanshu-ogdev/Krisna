# Planner upgrade program

This folder is the durable design record for the frozen Qwen3.5 Planner and
its Planner Review mode. It is organized by delivery stage rather than by
implementation file so future runtime, model, and hardware changes can be
made without losing the decision trail.

## Target architecture

```text
PlannerService
  ├─ versioned response contract
  ├─ prompt/context builder
  ├─ offline hybrid UICrit index
  ├─ runtime adapter
  │    ├─ llama.cpp / GGUF (native PC default)
  │    ├─ vLLM / compressed-tensors (Linux/high-memory)
  │    └─ Transformers (reference/fallback)
  ├─ schema validator + safe state application
  ├─ planning mode (text, low latency)
  ├─ review mode (optional vision + verifier evidence)
  ├─ lifecycle controller
  └─ telemetry/evaluation receipts
```

The Planner remains frozen. Quantization calibration is an inference artifact,
not fine-tuning, and must never create a Planner training checkpoint.

## Stage status

| Stage | Focus | Status |
|---|---|---|
| 00 | Frozen-model policy and repository boundaries | implemented in part |
| 01 | Versioned contract and safe delta application | implemented in part |
| 02 | Runtime abstraction and native model placement | implemented in part |
| 03 | Model acquisition, manifests, and reproducibility | tooling implemented |
| 04 | Evaluation corpus and quality/memory gates | runner implemented; corpus pending |
| 05 | Quantization bake-off | design only; requires target hardware |
| 06 | Structured generation | vLLM path started; llama/Transformers hardening pending |
| 07 | RAG index and retrieval quality | design only; current retrievers retained |
| 08 | Residency, swapping, and sleep/wake | design only; telemetry pending |
| 09 | Planner Review and Critic retirement | routing implemented; quality validation pending |

Read each stage document before changing its code. A stage is not complete
because a class or flag exists; it is complete only when its gate is backed by
an executed receipt on supported hardware.

## Source record

Research citations and claims are maintained in [`../citations.md`](../citations.md).
Repository-specific evidence is cited by path in each stage document.
