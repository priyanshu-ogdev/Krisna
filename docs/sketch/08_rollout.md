# Rollout, Operational Configuration & Monitoring

## 1. Production Configuration Flags

The Sketch Tier is configured via standard environment variables and runtime settings:

```bash
# Core operational mode: 'layout' (production default) or 'ab' (diagnostic logging)
KRIVEX_SKETCH_MODE=layout

# Wireframe rendering style: 'gray' (default), 'dark', 'light', or 'transparent'
KRIVEX_LAYOUT_WIREFRAME_STYLE=gray

# Output directory for layout JSON and wireframe raster artifacts
KRIVEX_LAYOUT_OUTPUT_DIR=models/sketch/layout_runs

# Verifier Stack strict safety gate threshold
KRIVEX_MIN_SAFETY_SCORE=0.90
```

---

## 2. Health Checks & Invariant Telemetry

During serving, each layout execution automatically computes and logs:
- `displacement_mean`: Mean coordinate adjustment applied by `project_layout` (threshold: $\le 0.04$).
- `displacement_max`: Maximum coordinate adjustment applied to any node (threshold: $\le 0.08$).
- `element_count`: Total active nodes in the layout tree ($1 \le N \le 64$).
- `ot_coherence`: Optimal transport vector-field trajectory coherence (target: $1.0$).
- `execution_ms`: Turnaround time for layout synthesis and wireframe rasterization (target: $< 50\text{ms}$).

---

## 3. Rollback & Fail-Safe Triggers

The system automatically halts or triggers self-correction if any of the following occur:
1. **Legality Breach**: `validate_layout` detects missing parents, parent cycles, or invalid bounding boxes.
2. **Lock Violation**: A node marked `locked: true` suffers coordinate movement $> 0.001$.
3. **Canvas Invariant Breach**: Canvas dimensions are not multiples of 32 ($W, H \not\equiv 0 \pmod{32}$).
4. **Safety Failure**: Generated design triggers safety score $< 0.90$ on the verifier stack.
