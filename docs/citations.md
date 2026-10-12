# Planner and inference upgrade research record

This document records the sources behind the Planner/runtime changes. Claims
are separated from measurements that still require execution on target PCs.

## Frozen Planner

The repository's final PRD and `training/src/krisna_training/planner/` mark the
Planner LoRA path deprecated. The production adaptation is retrieval,
prompting, and constrained output; no Planner checkpoint is trained here.

Repository sources: `docs/PRD.md`, `training/README.md`,
`inference/src/krisna_inference/backends/planner_backend.py`.

## Qwen3.5 architecture and modality

Qwen3.5's model card documents a hybrid vision-language causal model with
Gated DeltaNet and Gated Attention components. Transformers' current model
documentation distinguishes the text-only `Qwen3_5ForCausalLM` path from the
multimodal `Qwen3_5ForConditionalGeneration` plus `AutoProcessor` path.

- https://huggingface.co/Qwen/Qwen3.5-9B
- https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md

Implication: planning can remain text-only for the low-VRAM path; Review needs
an mmproj/vision-capable runtime if it must inspect pixels directly.

## vLLM structured outputs and sleep mode

vLLM documents JSON-schema constrained generation through
`StructuredOutputsParams(json=...)`. It also documents level-1 sleep as
offloading weights to CPU RAM and discarding KV cache. Sleep mode is documented
for CUDA/HIP and is not native-Windows support.

- https://docs.vllm.ai/en/stable/features/structured_outputs/
- https://docs.vllm.ai/en/stable/features/sleep_mode/
- https://docs.vllm.ai/en/stable/getting_started/installation/gpu/

These are runtime capabilities, not measurements for this repository. Cold
load, wake, and peak memory must be measured by `evaluate_planner.py` on the
target hardware.

## llama.cpp native-PC runtime

llama.cpp documents native multimodal support through `libmtmd`, GGUF models,
and optional multimodal projector files. The exact Qwen3.5 GGUF and projector
must be tested for chat-template, JSON grammar, and image behavior before
being treated as a production checkpoint.

- https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
- Gerganov, G., et al. (2023). *llama.cpp: Port of Facebook's LLaMA model in C/C++*. GitHub repository: https://github.com/ggml-org/llama.cpp. Demonstrating zero-copy memory mapping (`use_mmap`) for instantaneous process startup and unified GGUF quantized model execution.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (NeurIPS 2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691. Enabling $\mathcal{O}(N)$ memory scaling for long-context KV cache evaluation on Ampere and Ada GPUs.

## Quantization

The candidate selective W4A16 checkpoint documents retaining embeddings, vision
weights, and linear-attention layers at higher precision. This supports the
design principle of protecting sensitive/stateful components, but its quality,
memory, and Windows behavior remain deployment measurements rather than claims.

- https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16
- https://github.com/vllm-project/llm-compressor

Primary research that informs the bake-off:

- GPTQ, Frantar et al. (2022): https://arxiv.org/abs/2210.17323
- AWQ, Lin et al. (MLSys 2024): https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf
- SmoothQuant, Xiao et al. (2022): https://arxiv.org/abs/2211.10438

These papers support treating PTQ as an empirical quality/memory trade-off,
using activation-aware calibration where available, and preserving sensitive
components rather than assuming uniform low-bit quantization is harmless.

Qwen3.5-specific low-bit work is still emerging. A recent Qwen3.5-family
quantization report evaluates W4A16 while keeping embeddings, normalization,
vision, and selected stateful operations at higher precision:

- https://arxiv.org/abs/2606.06034

This is useful design evidence, not a substitute for this project's own
Planner-task evaluation.

## Structured generation research

- Geng et al., *Grammar-Constrained Decoding for Structured NLP Tasks without
  Finetuning* (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.674/
- Koo, *Automata-based constraints for language model decoding* (2024):
  https://arxiv.org/abs/2407.08103
- Willard and Louf, *Efficient Guided Generation for Large Language Models*
  (2023): https://arxiv.org/abs/2307.09702
- Geng et al., *Generating Structured Outputs from Language Models:
  Benchmark and Studies* (JSONSchemaBench, 2025):
  https://arxiv.org/abs/2501.10868

These sources support schema/grammar decoding, but also warn that tokenizer
alignment and constraint implementation can affect quality. Therefore the
Planner must measure semantic accuracy and latency, not only parse success.

## Retrieval research

- Khattab and Zaharia, *ColBERT* (SIGIR 2020):
  https://dl.acm.org/doi/10.1145/3397271.3401075
- Santhanam et al., *ColBERTv2* (NAACL 2022):
  https://aclanthology.org/2022.naacl-main.272/

These motivate precomputed document representations and late-interaction or
hybrid retrieval when lexical overlap alone misses paraphrases. The current
UICrit corpus is small enough for a simpler dense+sparse hybrid first; moving
to ColBERT-style retrieval should require a measured recall/latency benefit.

## Retrieval

The existing UICrit retriever documents a real lexical failure case. The
upgrade therefore builds an offline hybrid index rather than assuming TF-IDF
semantic equivalence. Dense retrieval quality must be evaluated on real UICrit
queries; synthetic unit vectors do not establish production relevance.

Repository source: `inference/src/krisna_inference/backends/planner_rag.py`.


## Sketch/layout generation research

- Guerreiro et al., *LayoutFlow: Flow Matching for Layout Generation* (ECCV 2024): https://arxiv.org/abs/2403.18187 and official code https://github.com/JulianGuerreiro/LayoutFlow
- Zhang et al., *LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models* (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_LayoutDiffusion_Improving_Graphic_Layout_Generation_by_Discrete_Diffusion_Probabilistic_Models_ICCV2023_paper.pdf
- Chai et al., *LayoutDM: Transformer-based Diffusion Model for Layout Generation* (CVPR 2023): https://arxiv.org/abs/2305.02567
- Liu et al., *UniLayDiff: A Unified Diffusion Transformer for Content-Aware Layout Generation* (CVPR 2026): https://arxiv.org/abs/2512.08897
- *LayoutCoT: Unleashing the Deep Reasoning Potential of Large Language Models for Layout Generation*: https://arxiv.org/abs/2504.10829
- Kim et al., *Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens* (ICCV 2025): https://arxiv.org/abs/2501.07730 and https://github.com/bytedance/1d-tokenizer

Interpretation: LayoutFlow is the strongest first adaptation target for Krisna because it is small, supports completion/refinement, and has official code/checkpoints. UniLayDiff is a later content-aware upgrade. TA-TiTok/MaskGen is a token-efficiency direction, not evidence that a 568M/1.1B generator fits the current low-VRAM target.

The current Krisna implementation provides an additional correction: the
MaskGIT sampler's Halton order does not implement remasking. It enumerates
currently masked positions, sorts them by combined generator/critic confidence,
and commits selected tokens. This repository fact is more authoritative for
Krisna behavior than a generic MaskGIT description; see
`inference/src/krisna_inference/backends/maskgit_model.py`.

## Flow-matching dynamics & ODE solvers

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (ICLR 2023).
  *Flow Matching for Generative Modeling*. arXiv:2210.02747.
  Foundational continuous normalizing flows regressing vector fields onto straight optimal transport paths. Governed by the continuous ODE $dx_t = v_\theta(x_t, t, c) dt$.
- Liu, X., Gong, C., & Liu, Q. (ICLR 2023).
  *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. arXiv:2209.03003.
  Straight vector-field trajectories connecting data distributions to Gaussian priors; establishes few-step Euler and 2nd-order Midpoint (RK2) predictor-corrector integration.
- Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., ... & Rombach, R. (ICML 2024).
  *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. arXiv:2403.03206.
  MMDiT architecture, velocity-based flow matching parameterization, and resolution-dependent time-shift function $\tilde{t} = \frac{s \cdot t}{1 + (s - 1) t}$.

## Unified diffusion rendering & preference alignment

- Alibaba Qwen Team (2026).
  *Qwen-Image-2.1: Advanced Visual Foundation Models with Image Conditioning & Flow Matching*.
- Brooks, T., Holynski, A., & Efros, A. A. (IEEE/CVF CVPR 2023).
  *InstructPix2Pix: Learning to Follow Image Editing Instructions*. arXiv:2211.09800.
  Dual classifier-free guidance mixing text and visual conditioning vectors:
  $$\hat{v}(x_t, t, c_I, c_T) = v(x_t, t, \emptyset, \emptyset) + s_I [v(x_t, t, c_I, \emptyset) - v(x_t, t, \emptyset, \emptyset)] + s_T [v(x_t, t, c_I, c_T) - v(x_t, t, c_I, \emptyset)]$$
  and time-decayed dynamic guidance scheduling $w(t) = 1.0 + (w_{\max} - 1.0) \cdot \cos(\frac{\pi}{2} t)$.
- Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., & Naik, N. (IEEE/CVF CVPR 2024).
  *Diffusion Model Alignment Using Direct Preference Optimization*. arXiv:2311.12908.
- Ho, J., & Salimans, T. (NeurIPS Workshop 2022).
  *Classifier-Free Diffusion Guidance*. arXiv:2207.12598.

## Asynchronous model swapping & systems runtime

- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (ACM SOSP 2023).
  *Efficient Memory Management for Large Language Model Serving with PagedAttention*.
- Gerganov, G., et al. (2023).
  *llama.cpp: Port of Facebook's LLaMA model in C/C++*. Demonstrating zero-copy memory mapping (`use_mmap`) for instantaneous process startup.
- PRD §7.4 & Krisna Dual-Loader Architecture:
  Coordinating the 2 baseline tiers (`PlannerBackendLlama` via llama.cpp GGUF zero-copy mmap, and `LayoutBackend` deterministic LayoutFlow) with the swappable Polish tier (`QwenImage21Backend` via Diffusers flow-matching pipeline). Non-blocking background pre-warming (`start_background_warm()`) decoupled from the incoming turn loop.

## Core UI & preference datasets

- Deka et al., *Rico: A Mobile App Dataset for Integrated Design and Code Search* (ACM UIST 2017): https://dl.acm.org/doi/10.1145/3126594.3126651
- Li et al., *CLAY: A Controllable Large-scale UI Layout Dataset with Visual and Semantic Annotations* (ACM CHI 2022): https://dl.acm.org/doi/10.1145/3491102.3502041
- Leiva et al., *Enrico: A Dataset for Topic Modeling of Mobile GUI Designs* (ACM MobileHCI 2020): https://dl.acm.org/doi/10.1145/3379503.3403544
- Wang et al., *Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning* (ACM UIST 2021): https://dl.acm.org/doi/10.1145/3472749.3474765
- Google Research, *UICrit: Multimodal UI Design Critique Dataset* (2023): `google-research-datasets/uicrit`
- GameLabel-10K, *High-Contrast Game UI Preference Dataset* (2024): MIT License.

## Canonical research documentation

- **Formal Research Paper**: [`docs/RESEARCH_PAPER.md`](RESEARCH_PAPER.md) — *Krisna: Bounded Multi-Turn Agentic UI Design via Structured Layout IR and Unified Diffusion Rendering*.
- **Developer Notes & Feature Audit**: [`docs/DEV_NOTES.md`](DEV_NOTES.md) — Engineering history, component specifications, memory ledgers, and feature retention audit.


