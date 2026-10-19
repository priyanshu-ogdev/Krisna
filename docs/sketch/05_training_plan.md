# Sketch Tier Training Plan & Curriculum

## 1. Overview & Phased Roadmap

The Krisna Sketch Tier leverages a phased curriculum designed to maximize layout stability, spatial precision, and zero-hallucination compliance.

---

## 2. Curriculum Phases

### Phase A — Layout IR & Deterministic Renderer (Completed)
- Canonical Pydantic schema: `LayoutIR` (`krisna.layout.v1`).
- Deterministic Flexbox Collision Partitioner with 8pt grid snapping (`_resolve_flex_layout`).
- TrueType Typography Renderer with measured text boundary wrapping (`_wrap_text_measured`) and minimum 12px font clamping.
- Native RGBA transparent canvas support (`(0, 0, 0, 0)`).
- Complete round-trip serialization and unit test suite ($>98\%$ coverage).

### Phase B — Planner-to-Layout Baseline (Production Operational)
- Integrates Planner VLA actions (`AddComponentAction`, `RemoveComponentAction`, `ResizeAction`, `SetTextAction`, `LockRegionAction`).
- Deterministic spatial layout generation achieving $<45\text{ms}$ turnaround time and $0\text{MB}$ VRAM consumption.
- Serves as the robust foundation and ground-truth constraint legalizer for downstream models.

### Phase C — LayoutFlow Continuous Vector-Field Model
- Training a 15M continuous flow-matching Transformer over bounding box trajectories $\mathbf{b}_t \in [0, 1]^4$:
  $$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, p_t(\mathbf{b}), p(\mathbf{b}_1)} \left\| v_\theta(\mathbf{b}_t, t, c) - (\mathbf{b}_1 - \mathbf{b}_0) \right\|_2^2$$
- Multitask conditioning:
  1. *Unconditional generation*: Sampling from Gaussian noise $\mathbf{b}_0 \sim \mathcal{N}(0, I)$.
  2. *Type/Size-conditioned generation*: Category embeddings and aspect ratio prefixes.
  3. *Partial layout completion*: Masking subset of nodes while freezing existing elements.
  4. *Locked-region refinement*: Flow integration with zero-velocity clamping on locked coordinates.
- Auxiliary regularizers:
  $$\mathcal{L}_\text{total} = \mathcal{L}_\text{FM} + \lambda_\text{overlap} \mathcal{L}_\text{overlap} + \lambda_\text{alignment} \mathcal{L}_\text{alignment} + \lambda_\text{containment} \mathcal{L}_\text{containment}$$

### Phase D — Multimodal Vision-Language Layout Refinement
- Alignment with ShowUI and InstructLayout graph representations.
- Screen-space coordinate grounding for interactive multi-turn canvas manipulation.
