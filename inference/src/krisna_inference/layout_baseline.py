"""Planner-to-layout baseline.

This is intentionally not a learned model. It converts a validated Planner
delta into a deterministic LayoutIR, providing the required baseline before a
LayoutFlow-style model is trained.
"""

from __future__ import annotations

import logging

from krisna_inference.layout_ir import BBox, LayoutIR, LayoutNode, project_layout

log = logging.getLogger("krisna_inference.layout_baseline")


def _resolve_flex_layout(
    node_map: dict[str, LayoutNode],
    constraints: dict,
    canvas_w: int = 1024,
    canvas_h: int = 1024,
    conflicts: list[str] | None = None,
) -> dict[str, LayoutNode]:
    """Deterministic Box-Model Collision Partitioner.
    
    Applies deterministic 1D axis partitioning and logical 8pt UI grid snapping to resolve colliding sibling nodes.
    Locked nodes (node.locked is True) are strictly IMMUTABLE and exempt from being moved.
    """
    direction = str(constraints.get("layout_direction") or "row").lower()
    grid_x = 8.0 / canvas_w
    grid_y = 8.0 / canvas_h

    def _snap_x(val: float) -> float:
        return round(round(val / grid_x) * grid_x, 4)

    def _snap_y(val: float) -> float:
        return round(round(val / grid_y) * grid_y, 4)

    # 1. Identify mutable vs locked nodes (locked nodes and parent containers are exempt from clustering)
    structural_container_ids = {"root", "content"}
    locked_ids = {nid for nid, node in node_map.items() if node.locked and nid != "root"}
    mutable_ids = [
        nid for nid, node in node_map.items()
        if not node.locked and nid not in structural_container_ids
    ]

    # 2a. Check locked-vs-locked collisions: neither can move; surface conflict diagnostic to Planner
    locked_list = sorted(list(locked_ids))
    for i, lid1 in enumerate(locked_list):
        for lid2 in locked_list[i + 1:]:
            b1 = node_map[lid1].bbox
            b2 = node_map[lid2].bbox
            if b1.iou(b2) > 0.0 or (b1.x1 < b2.x2 and b1.x2 > b2.x1 and b1.y1 < b2.y2 and b1.y2 > b2.y1):
                msg = f"Locked node conflict: '{lid1}' collides with locked '{lid2}' (IoU={b1.iou(b2):.2f}). Immutable constraints cannot resolve."
                log.warning(msg)
                if conflicts is not None:
                    conflicts.append(msg)

    # 2b. Check collision with locked nodes: mutable nodes must yield space, locked nodes never move
    for mid in mutable_ids:
        mb = node_map[mid].bbox
        for lid in locked_ids:
            lb = node_map[lid].bbox
            if mb.iou(lb) > 0.0 or (mb.x1 < lb.x2 and mb.x2 > lb.x1 and mb.y1 < lb.y2 and mb.y2 > lb.y1):
                avail_below = 0.96 - (lb.y2 + 0.016)
                avail_above = (lb.y1 - 0.016) - 0.04
                if avail_below >= 0.08:
                    new_y1 = _snap_y(lb.y2 + 0.016)
                    new_y2 = _snap_y(min(0.96, new_y1 + min(mb.y2 - mb.y1, avail_below)))
                    node_map[mid] = node_map[mid].model_copy(update={"bbox": BBox(x1=mb.x1, y1=new_y1, x2=mb.x2, y2=new_y2)})
                elif avail_above >= 0.08:
                    new_y2 = _snap_y(lb.y1 - 0.016)
                    new_y1 = _snap_y(max(0.04, new_y2 - min(mb.y2 - mb.y1, avail_above)))
                    node_map[mid] = node_map[mid].model_copy(update={"bbox": BBox(x1=mb.x1, y1=new_y1, x2=mb.x2, y2=new_y2)})
                elif lb.x2 + 0.016 <= 0.90:
                    new_x1 = _snap_x(lb.x2 + 0.016)
                    new_x2 = _snap_x(min(0.96, new_x1 + min(mb.x2 - mb.x1, 0.96 - new_x1)))
                    node_map[mid] = node_map[mid].model_copy(update={"bbox": BBox(x1=new_x1, y1=mb.y1, x2=new_x2, y2=mb.y2)})
                else:
                    # Canvas Y-overflow fallback: clamp to bottom margin with minimum height
                    log.warning("Mutable node '%s' displaced beyond canvas bounds; clamping to canvas bottom.", mid)
                    clamped_y1 = _snap_y(max(0.04, 0.96 - min(0.08, mb.y2 - mb.y1)))
                    node_map[mid] = node_map[mid].model_copy(update={"bbox": BBox(x1=mb.x1, y1=clamped_y1, x2=mb.x2, y2=0.96)})

    # 3. Identify colliding clusters among mutable nodes (IoU > 0.05 or bounding box overlap)
    colliding_clusters: list[list[str]] = []
    visited: set[str] = set()

    for i, nid1 in enumerate(mutable_ids):
        if nid1 in visited:
            continue
        cluster = [nid1]
        b1 = node_map[nid1].bbox
        for nid2 in mutable_ids[i + 1:]:
            if nid2 in visited:
                continue
            b2 = node_map[nid2].bbox
            if b1.iou(b2) > 0.05 or (b1.x1 < b2.x2 and b1.x2 > b2.x1 and b1.y1 < b2.y2 and b1.y2 > b2.y1):
                cluster.append(nid2)
        if len(cluster) > 1:
            for cid in cluster:
                visited.add(cid)
            colliding_clusters.append(cluster)

    # 4. Partition each mutable colliding cluster deterministically
    for cluster in colliding_clusters:
        boxes = [node_map[cid].bbox for cid in cluster]
        min_x = min(b.x1 for b in boxes)
        max_x = max(b.x2 for b in boxes)
        min_y = min(b.y1 for b in boxes)
        max_y = max(b.y2 for b in boxes)

        span_w = max_x - min_x
        span_h = max_y - min_y
        is_row = (direction == "row") or (span_w >= span_h and direction != "column")
        n = len(cluster)
        gap = 0.016
        pad = 0.012

        if is_row:
            avail_w = max(0.10, span_w - (2 * pad) - (n - 1) * gap)
            item_w = avail_w / n
            for idx, cid in enumerate(cluster):
                cur = node_map[cid]
                cx1 = _snap_x(max(0.02, min_x + pad + idx * (item_w + gap)))
                cx2 = _snap_x(min(0.98, cx1 + item_w))
                if cx2 <= cx1:
                    cx2 = min(0.98, cx1 + 0.04)
                    cx1 = max(0.02, cx2 - 0.04)
                cy1 = _snap_y(max(0.02, min_y + pad))
                cy2 = _snap_y(min(0.98, max_y - pad))
                if cy2 <= cy1:
                    cy2 = min(0.98, cy1 + 0.04)
                    cy1 = max(0.02, cy2 - 0.04)
                new_box = BBox(x1=cx1, y1=cy1, x2=cx2, y2=cy2)
                node_map[cid] = cur.model_copy(update={"bbox": new_box})
        else:
            avail_h = max(0.08, span_h - (2 * pad) - (n - 1) * gap)
            item_h = avail_h / n
            for idx, cid in enumerate(cluster):
                cur = node_map[cid]
                cy1 = _snap_y(max(0.02, min_y + pad + idx * (item_h + gap)))
                cy2 = _snap_y(min(0.98, cy1 + item_h))
                if cy2 <= cy1:
                    cy2 = min(0.98, cy1 + 0.04)
                    cy1 = max(0.02, cy2 - 0.04)
                cx1 = _snap_x(max(0.02, min_x + pad))
                cx2 = _snap_x(min(0.98, max_x - pad))
                if cx2 <= cx1:
                    cx2 = min(0.98, cx1 + 0.04)
                    cx1 = max(0.02, cx2 - 0.04)
                new_box = BBox(x1=cx1, y1=cy1, x2=cx2, y2=cy2)
                node_map[cid] = cur.model_copy(update={"bbox": new_box})

    return node_map


def layout_from_constraints(constraints: dict, *, text: str = "") -> LayoutIR:
    """Build a deterministic LayoutIR synchronized with Planner constraints and DesignActions.

    Research grounding:
    - ShowUI (Lin et al., CVPR 2025): Action-token grammar for GUI elements.
    - InstructLayout (Wen et al., IEEE TPAMI 2025): Multi-modal instruction tuning for UI layout updates.
    - Cassowary (Badros et al., ACM TOCHI): Deterministic layout constraint resolution.
    """
    style = constraints.get("style")
    palette = constraints.get("palette")

    # Determine canvas dimensions from aspect_ratio or target_device (multiples of 32 for DiT)
    aspect_ratio = str(constraints.get("aspect_ratio") or constraints.get("target_device") or "").lower()
    if "mobile" in aspect_ratio or "9:16" in aspect_ratio or "phone" in aspect_ratio:
        canvas_w, canvas_h = 768, 1344
    elif "desktop" in aspect_ratio or "16:9" in aspect_ratio or "wide" in aspect_ratio:
        canvas_w, canvas_h = 1344, 768
    elif "tablet" in aspect_ratio or "4:3" in aspect_ratio or "ipad" in aspect_ratio:
        canvas_w, canvas_h = 1152, 864
    else:
        canvas_w, canvas_h = 1024, 1024

    # Start with baseline foundation hierarchy
    starter_nodes = [
        LayoutNode(
            id="root",
            type="container",
            bbox=BBox(x1=0.04, y1=0.04, x2=0.96, y2=0.96),
            style_tokens={
                "style": style,
                "palette": palette,
                "asset_mode": constraints.get("asset_mode"),
            },
        ),
        LayoutNode(
            id="title",
            type="text",
            bbox=BBox(x1=0.10, y1=0.10, x2=0.90, y2=0.20),
            parent="root",
            text=text[:240] or "Title",
            locked=False,
        ),
        LayoutNode(
            id="content",
            type="container",
            bbox=BBox(x1=0.10, y1=0.26, x2=0.90, y2=0.82),
            parent="root",
        ),
        LayoutNode(
            id="action",
            type="button",
            bbox=BBox(x1=0.10, y1=0.86, x2=0.34, y2=0.94),
            parent="root",
            text="Continue",
        ),
    ]

    node_map: dict[str, LayoutNode] = {node.id: node for node in starter_nodes}

    # Apply VLA DesignAction elements from Planner delta
    actions = constraints.get("actions") or []
    if isinstance(actions, list):
        for act in actions:
            act_dict = act.model_dump() if hasattr(act, "model_dump") else (act if isinstance(act, dict) else {})
            act_type = str(act_dict.get("action_type") or "").upper()
            target_id = act_dict.get("target_id")
            params = act_dict.get("parameters") if isinstance(act_dict.get("parameters"), dict) else {}

            if act_type == "ADD_COMPONENT":
                comp_type = act_dict.get("component_type") or params.get("component_type") or "container"
                bbox_raw = act_dict.get("bbox") or params.get("bbox")
                props = act_dict.get("props") or params.get("props") or {}
                text_content = act_dict.get("text_content") or props.get("label") or params.get("text_content")
                comp_id = target_id or f"{comp_type}_{len(node_map)}"

                if bbox_raw and len(bbox_raw) == 4:
                    try:
                        bx, by, bw, bh = [float(v) for v in bbox_raw]
                        # Support both (x, y, w, h) and (x1, y1, x2, y2)
                        if bw <= 1.0 and bh <= 1.0 and bx + bw <= 1.05 and by + bh <= 1.05:
                            x1 = max(0.0, min(0.98, bx))
                            y1 = max(0.0, min(0.98, by))
                            x2 = max(x1 + 0.01, min(1.0, bx + bw))
                            y2 = max(y1 + 0.01, min(1.0, by + bh))
                        else:
                            x1 = max(0.0, min(0.98, bx))
                            y1 = max(0.0, min(0.98, by))
                            x2 = max(x1 + 0.01, min(1.0, bw))
                            y2 = max(y1 + 0.01, min(1.0, bh))
                        new_bbox = BBox(x1=round(x1, 4), y1=round(y1, 4), x2=round(x2, 4), y2=round(y2, 4))
                    except Exception:
                        new_bbox = BBox(x1=0.15, y1=0.30, x2=0.85, y2=0.45)
                else:
                    new_bbox = BBox(x1=0.15, y1=0.30, x2=0.85, y2=0.45)

                node_map[comp_id] = LayoutNode(
                    id=comp_id,
                    type=comp_type,
                    bbox=new_bbox,
                    parent="root",
                    text=text_content,
                )

            elif act_type == "REMOVE_COMPONENT":
                if target_id and target_id in node_map and target_id != "root":
                    del node_map[target_id]

            elif act_type in ("RESIZE", "MOVE_COMPONENT"):
                if target_id and target_id in node_map and target_id != "root":
                    bbox_raw = act_dict.get("bbox") or params.get("bbox")
                    if bbox_raw and len(bbox_raw) == 4:
                        try:
                            bx, by, bw, bh = [float(v) for v in bbox_raw]
                            if bw <= 1.0 and bh <= 1.0 and bx + bw <= 1.05 and by + bh <= 1.05:
                                x1 = max(0.0, min(0.98, bx))
                                y1 = max(0.0, min(0.98, by))
                                x2 = max(x1 + 0.01, min(1.0, bx + bw))
                                y2 = max(y1 + 0.01, min(1.0, by + bh))
                            else:
                                x1 = max(0.0, min(0.98, bx))
                                y1 = max(0.0, min(0.98, by))
                                x2 = max(x1 + 0.01, min(1.0, bw))
                                y2 = max(y1 + 0.01, min(1.0, bh))
                            new_bbox = BBox(x1=round(x1, 4), y1=round(y1, 4), x2=round(x2, 4), y2=round(y2, 4))
                            node_map[target_id] = node_map[target_id].model_copy(update={"bbox": new_bbox})
                        except Exception:
                            pass

            elif act_type == "SET_TEXT":
                if target_id and target_id in node_map:
                    new_txt = act_dict.get("text_content") or params.get("text_content")
                    if new_txt:
                        node_map[target_id] = node_map[target_id].model_copy(update={"text": new_txt})

            elif act_type == "LOCK_REGION":
                if target_id and target_id in node_map:
                    node_map[target_id] = node_map[target_id].model_copy(update={"locked": True})

    # Synchronize locked_regions directly with LayoutNode.locked BEFORE Flexbox resolution
    locked_regions = constraints.get("locked_regions") or []
    if isinstance(locked_regions, list):
        for lr in locked_regions:
            bbox_raw = lr.get("bbox") if isinstance(lr, dict) else getattr(lr, "bbox", None)
            if bbox_raw and len(bbox_raw) == 4:
                try:
                    bx, by, bw, bh = [float(v) for v in bbox_raw]
                    # Check overlap with any existing node
                    for nid, node in list(node_map.items()):
                        if nid == "root":
                            continue
                        nb = node.bbox
                        ncx, ncy = (nb.x1 + nb.x2) / 2.0, (nb.y1 + nb.y2) / 2.0
                        if bx <= ncx <= (bx + bw) and by <= ncy <= (by + bh):
                            node_map[nid] = node.model_copy(update={"locked": True})
                except Exception:
                    pass

    # Apply deterministic Flexbox partitioner (with strict locked-node exemption and 8pt grid snapping)
    layout_conflicts: list[str] = []
    node_map = _resolve_flex_layout(node_map, constraints, canvas_w=canvas_w, canvas_h=canvas_h, conflicts=layout_conflicts)

    return project_layout(
        LayoutIR(
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            nodes=list(node_map.values()),
            diagnostics=layout_conflicts,
        )
    ).layout
