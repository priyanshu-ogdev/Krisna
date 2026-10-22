# Layout Intermediate Representation (`krivex.layout.v1`)

## 1. Specification Overview

The canonical intermediate representation for graphic layouts in Krivex is **`LayoutIR`** (schema version `krivex.layout.v1`). 

It decouples planning logic from generative image synthesis. Generative layout models (such as LayoutFlow), conversational planners (such as Llama-based Planner), and deterministic layout solvers emit and consume this typed tree before any rasterization or Polish-tier handoff occurs.

```json
{
  "version": "krivex.layout.v1",
  "canvas_width": 1024,
  "canvas_height": 1024,
  "nodes": [
    {
      "id": "root",
      "type": "container",
      "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
      "parent": null,
      "z_index": 0,
      "style_tokens": {
        "style": "clean dark",
        "palette": ["#0F172A", "#38BDF8", "#818CF8"],
        "asset_mode": "opaque"
      },
      "text": null,
      "content_ref": null,
      "locked": false
    },
    {
      "id": "header_title",
      "type": "text",
      "bbox": {"x1": 0.08, "y1": 0.08, "x2": 0.92, "y2": 0.16},
      "parent": "root",
      "z_index": 1,
      "style_tokens": {"role": "title"},
      "text": "Executive Analytics Overview",
      "content_ref": null,
      "locked": true
    },
    {
      "id": "primary_cta",
      "type": "button",
      "bbox": {"x1": 0.65, "y1": 0.84, "x2": 0.92, "y2": 0.92},
      "parent": "root",
      "z_index": 2,
      "style_tokens": {"variant": "filled"},
      "text": "Export Report",
      "content_ref": null,
      "locked": false
    }
  ],
  "relations": [
    {
      "source": "header_title",
      "relation": "above",
      "target": "primary_cta"
    }
  ],
  "diagnostics": []
}
```

---

## 2. Pydantic Domain Models

Defined in `inference/src/krivex_inference/layout_ir.py`:

### 2.1 `BBox`
Normalized bounding box in unit canvas space $[0.0, 1.0]^4$:
- `x1: float` ($0.0 \le x_1 \le 1.0$)
- `y1: float` ($0.0 \le y_1 \le 1.0$)
- `x2: float` ($0.0 \le x_2 \le 1.0$, validated: $x_2 > x_1$)
- `y2: float` ($0.0 \le y_2 \le 1.0$, validated: $y_2 > y_1$)
- Methods:
  - `from_xywh(x, y, w, h) -> BBox`
  - `to_xywh() -> tuple[float, float, float, float]`
  - `iou(other: BBox) -> float`

### 2.2 `LayoutNode`
Semantic UI element representing an atomic component or layout container:
- `id: str` (Unique string identifier, $1 \le \text{length} \le 128$)
- `type: str` (Member of `{"container", "text", "image", "button", "input", "list", "card", "icon", "divider", "other"}`)
- `bbox: BBox` (Geometric footprint)
- `parent: str | None` (ID of parent container; `None` for root)
- `z_index: int` (Layer stacking order for rendering, default `0`)
- `style_tokens: dict[str, Any]` (Visual tokens: palette, variant, typography role)
- `text: str | None` (Verbatim typographic string to be rendered)
- `content_ref: str | None` (Optional asset reference or icon URI)
- `locked: bool` (Immutable flag: strictly exempt from displacement)

### 2.3 `LayoutRelation`
Explicit spatial and topological constraints between pairs of nodes:
- `source: str` (Source node ID)
- `relation: str` (`"above"`, `"below"`, `"align_x"`, `"align_y"`, `"inside"`, `"near"`)
- `target: str` (Target node ID)

### 2.4 `LayoutIR`
Root document container:
- `version: str` (`"krivex.layout.v1"`)
- `canvas_width: int` ($1 \le W \le 8192$, enforced $W \equiv 0 \pmod{32}$)
- `canvas_height: int` ($1 \le H \le 8192$, enforced $H \equiv 0 \pmod{32}$)
- `nodes: list[LayoutNode]` (Max 64 nodes per layout)
- `relations: list[LayoutRelation]` (Max 256 relations)
- `diagnostics: list[str]` (Warnings, conflict diagnostics, or resolution notes)
- Methods:
  - `node_map() -> dict[str, LayoutNode]`

---

## 3. Mathematical Invariants & Validation Rules

`validate_layout(layout: LayoutIR)` enforces the following structural theorems:

1. **Unique Node Identifiers**:
   $$\forall i \ne j, \quad \text{node}_i.\text{id} \ne \text{node}_j.\text{id}$$

2. **Containment Tree Acyclicity**:
   The parent-child graph $G = (V, E)$ where $(u, v) \in E \iff u.\text{parent} = v.\text{id}$ must be a directed forest with no self-parenting ($u.\text{parent} \ne u.\text{id}$) and no directed cycles.

3. **Referential Integrity**:
   - $\forall u \in V$ with $u.\text{parent} \ne \text{None}$, $u.\text{parent} \in V$.
   - $\forall r \in \text{relations}$, $r.\text{source} \in V \land r.\text{target} \in V$.

4. **32px Canvas Invariant**:
   $$W \equiv 0 \pmod{32}, \quad H \equiv 0 \pmod{32}$$
   Guarantees exact alignment with the Polish Tier's $16\times$ VAE downsampling factor and MMDiT $2 \times 2$ patch stride.

5. **8pt UI Grid Snapping**:
   $$\Delta x = k_x \cdot \frac{8}{W}, \quad \Delta y = k_y \cdot \frac{8}{H}, \quad k_x, k_y \in \mathbb{Z}$$

---

## 4. Legalization and Projection (`project_layout`)

When layouts are sampled from probabilistic continuous flow models (or manual Planner input), small numerical boundary violations can occur.

`project_layout(layout: LayoutIR, max_displacement: float = 0.08) -> ProjectionReport` clamps coordinates into $[0.0, 1.0]$, guarantees $x_1 < x_2$ and $y_1 < y_2$, reports mean and maximum displacement, and flags any altered nodes. If $\text{displacement\_max} > 0.08$, downstream callers can flag a resampling turn.
