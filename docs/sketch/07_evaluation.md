# Sketch Tier Evaluation & Benchmark Suite

## 1. Multi-Tier Evaluation Framework

Evaluation of the Sketch Tier is conducted across two complementary axes:
1. **Isolated Structured Layout Metrics**: Evaluating the mathematical legality, alignment, and constraint fidelity of the generated `LayoutIR` tree.
2. **End-to-End Closed-Loop Synthesis Metrics**: Evaluating the visual, typographic, and spatial fidelity of the polished render produced by the downstream Polish tier.

---

## 2. Structured Layout Metrics

| Metric | Target | Formula / Definition | Operational Role |
| :--- | :--- | :--- | :--- |
| **Legality Violation Rate** | **$0.0\%$** | $\frac{1}{N}\sum \mathbb{I}(\text{violations} > 0)$ | Hard gate: bounding boxes inside $[0, 1]^4$, $x_1 < x_2$, $y_1 < y_2$, acyclic tree. |
| **Locked Node Drift** | **$0.000$** | $\max_{i \in \text{locked}} \|\mathbf{b}_i - \mathbf{b}_i^\text{orig}\|_2$ | Invariant: locked elements must experience zero spatial displacement. |
| **Box Overlap Rate** | **$\le 5.0\%$** | $\frac{1}{\binom{N}{2}}\sum_{i < j} \mathbb{I}(\text{IoU}(\mathbf{b}_i, \mathbf{b}_j) > 0.05)$ | Validates collision partitioner effectiveness on sibling elements. |
| **Grid Alignment Score** | **$\ge 85.0\%$** | $\frac{1}{2N}\sum (\mathbb{I}(x \in 8\text{pt}) + \mathbb{I}(y \in 8\text{pt}))$ | Measures adherence to the 8pt UI grid rhythm. |
| **Containment Precision** | **$100.0\%$** | $\frac{1}{|V_\text{child}|}\sum \mathbb{I}(\mathbf{b}_u \subseteq \mathbf{b}_{\text{parent}(u)})$ | Asserts children do not breach container parent bounds. |

---

## 3. End-to-End Polish Integration Metrics

Evaluated via `VerifierStack.evaluate_finalize_output()` in `flows.finalize()`:

| Metric | Target Threshold | Description | Verifier Component |
| :--- | :--- | :--- | :--- |
| **Layout IoU** | **$\ge 0.70$** | Mean IoU between rendered components and LayoutIR bounding boxes. | `LayoutIoUVerifier` |
| **OCR Readability** | **$\ge 0.80$** | Word error rate and string match on expected typographic labels. | `OCRReadabilityVerifier` |
| **Seam Continuity** | **$\ge 0.85$** | Boundary gradient consistency between locked and edited pixels. | `LockPreservationVerifier` |
| **Safety Gate** | **$\ge 0.90$** | Multi-class safety, NSFW, and content compliance filter. | `SafetyVerifier` |
| **Handoff Consistency**| **$\ge 0.80$** | Structural embedding similarity between wireframe and final render. | `HandoffConsistencyVerifier`|

---

## 4. Closed-Loop Auto-Refine Gating

If any metric falls below its target threshold:
1. An automated retry is triggered via `build_grounded_spatial_retry_prompt()`.
2. The retry prompt explicitly instructs the Polish tier on the drifted region using exact component labels.
3. If the retry improves IoU, the new render is committed; otherwise, a diagnostic is logged in the conversation history for Planner self-correction.
