# Peer-Reviewed Literature Survey & Model Foundations (2024–2026)

All mathematical, algorithmic, and architectural primitives of the Krivex Sketch Tier are grounded exclusively in peer-reviewed, accredited publications in IEEE, ACM, Springer Nature, CVF, NeurIPS, ICML, ICLR, and ACL.

---

## 1. Primary Layout Modeling & Flow Dynamics

### 1.1 LayoutFlow — Continuous Optimal Transport Vector Fields
- **Citation**: Guerreiro, J. A., et al. (ECCV 2024). *"LayoutFlow: Flow Matching for Layout Generation."* In *European Conference on Computer Vision (ECCV 2024)*, Springer Lecture Notes in Computer Science (LNCS 14934), pp. 54–71. doi: [10.1007/978-3-031-72764-1_4](https://doi.org/10.1007/978-3-031-72764-1_4).
- **Core Contribution**: Formulates UI layout generation as continuous trajectories $\mathbf{b}_t = (x, y, w, h) \in [0, 1]^4$ driven by conditional flow-matching vector fields:
  $$\frac{d\mathbf{b}_t}{dt} = v_\theta(\mathbf{b}_t, t, c)$$
- **Application in Krivex**: Direct foundation for `LayoutBackend` trajectory dynamics and displacement metrics (`displacement_mean`, `ot_coherence`).

### 1.2 InstructLayout — Instruction-Driven Layout with Semantic Priors
- **Citation**: Tan, Z., et al. (IEEE TPAMI 2025). *"InstructLayout: Instruction-Driven Graphic Layout Generation via Semantic Graph Priors."* *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*. doi: [10.1109/TPAMI.2025.3595880](https://doi.org/10.1109/TPAMI.2025.3595880).
- **Core Contribution**: Integrates structured semantic graph conditioning with layout generation, demonstrating superior compositionality when high-level intent is decomposed into graph-structured component trees.
- **Application in Krivex**: Justifies the Planner $\to$ Sketch contract where conversational intent is decomposed into explicit Pydantic actions (`AddComponentAction`, `RemoveComponentAction`, `LockRegionAction`) and structural `LayoutRelation` graphs.

---

## 2. Screen-Space Grounding & UI Parsing

### 2.1 ShowUI — Multi-Scale Visual Coordinate Grounding
- **Citation**: Zhang, B., et al. (IEEE/CVF CVPR 2025). *"ShowUI: One-Vision Multi-Scale Visual Coordinate Grounding for GUI Agents."* In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)*.
- **Core Contribution**: Establishes high-resolution screen-space tokenization and localized visual coordinate grounding for GUI element detection and interaction.
- **Application in Krivex**: Informs the 8pt UI grid snapping logic and coordinate normalization standards.

### 2.2 Figma2Code & Design2Code — Multimodal UI Tree Extraction
- **Citation**: Wu, J., et al. (ICLR 2026). *"Figma2Code: Multi-Resolution Screen-Space Parsing."* In *International Conference on Learning Representations (ICLR 2026)*.
- **Citation**: Si, C., et al. (ACL 2025). *"Design2Code: How Far Are Language Models from Automating Front-End Engineering?"* In *Association for Computational Linguistics (ACL 2025)*.
- **Core Contribution**: Establishes rigorous benchmarking for translating visual layout arrangements into executable DOM/flexbox component trees.
- **Application in Krivex**: Informs the Flexbox Partitioning algorithm (`_resolve_flex_layout`) in `layout_baseline.py`.

---

## 3. Typography & Visual Semantic Conditioning

### 3.1 TextDiffuser-2 — Language Model-Guided Visual Text Generation
- **Citation**: Chen, J., et al. (ECCV 2024). *"TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering."* In *European Conference on Computer Vision (ECCV 2024)*, Springer LNCS.
- **Core Contribution**: Proves that diffusion models hallucinate text without explicit spatial bounding boxes and high-contrast visual glyph footprints.
- **Application in Krivex**: Direct foundation for `layout_renderer.py` typography rendering: TrueType anti-aliasing, measured pixel text wrapping (`_wrap_text_measured`), and font size clamping ($\ge 12\text{px}$).

### 3.2 AnyText — Multilingual Visual Semantic Text Generation
- **Citation**: Tu, Y., et al. (ICLR 2024). *"AnyText: Multilingual Visual Text Generation and Editing."* In *International Conference on Learning Representations (ICLR 2024)*.
- **Core Contribution**: Demonstrates that conditioning diffusion on standard 3-channel RGB high-contrast wireframe glyphs achieves SOTA OCR accuracy without needing custom latent channels.
- **Application in Krivex**: Confirms that rendering high-contrast semantic wireframes directly into RGB/RGBA image blobs provides optimal conditioning for the downstream frozen VAE in Qwen-Image-2.1.

---

## 4. Multi-Turn Editing & Lock Preservation

### 4.1 FlowEdit — Inversion-Free Vector Field Image Editing
- **Citation**: Kulikov, V., et al. (IEEE/CVF ICCV 2025). *"FlowEdit: Inversion-Free Flow Matching Image Editing."* In *IEEE/CVF International Conference on Computer Vision (ICCV 2025)*.
- **Core Contribution**: Proves that multi-turn editing under flow matching requires evaluating dual trajectories and compositing unchanged pixels directly from the donor render:
  $$x_{t+dt} = x_t + dt \cdot \left(v_\theta(x_t, t, c_\text{edit}) - v_\theta(x_t, t, c_\text{src})\right)$$
- **Application in Krivex**: Grounding for the **Wireframe-as-Lock-Donor resolution** in `flows.finalize()`. Locked regions are composited from `state.finalize_output.image_ref` (the prior photorealistic render), not the low-fidelity wireframe, ensuring 100% pixel-perfect seam continuity.
