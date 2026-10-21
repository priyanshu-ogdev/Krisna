"""Planner-to-layout baseline.

This is intentionally not a learned model. It converts a validated Planner
delta into a deterministic LayoutIR, providing the required baseline before a
LayoutFlow-style model is trained.
"""

from __future__ import annotations

import logging

from krisna_inference.layout_ir import BBox, LayoutIR, LayoutNode, SceneArchetype, project_layout

log = logging.getLogger("krisna_inference.layout_baseline")


def _resolve_flex_layout(
    node_map: dict[str, LayoutNode],
    constraints: dict,
    canvas_w: int = 1024,
    canvas_h: int = 1024,
    conflicts: list[str] | None = None,
) -> dict[str, LayoutNode]:
    """Deterministic Box-Model Collision Partitioner with Layer-Aware Z-Index Stacking.
    
    Applies deterministic 1D axis partitioning and logical 8pt UI grid snapping to resolve colliding sibling nodes.
    Locked nodes (node.locked is True) are strictly IMMUTABLE and exempt from being moved.
    Layer-Aware: Nodes on different z_index planes or with parent-child relationships are treated as layered
    and exempt from collision separation (allowing graphic design overlays, badges, stickers, and text over art).
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

    # Enforce minimum dimension floor for facial/focal subject nodes (>= 128px on canvas)
    # Grounding: AnyDoor (CVPR 2024), PhotoMaker (CVPR 2024).
    # Faces smaller than 128px span < 4x4 DiT tokens and suffer irreversible spatial annihilation in 16x VAE.
    min_face_w = 128.0 / canvas_w
    min_face_h = 128.0 / canvas_h
    for mid in mutable_ids:
        cur_node = node_map[mid]
        if cur_node.is_focal_subject or cur_node.type in ("focal_subject", "character", "photo_frame"):
            nb = cur_node.bbox
            nw = nb.x2 - nb.x1
            nh = nb.y2 - nb.y1
            if nw < min_face_w or nh < min_face_h:
                new_w = max(nw, min_face_w)
                new_h = max(nh, min_face_h)
                cx = (nb.x1 + nb.x2) / 2.0
                cy = (nb.y1 + nb.y2) / 2.0
                nx1 = _snap_x(max(0.02, min(0.98 - new_w, cx - new_w / 2.0)))
                ny1 = _snap_y(max(0.02, min(0.98 - new_h, cy - new_h / 2.0)))
                nx2 = _snap_x(min(0.98, nx1 + new_w))
                ny2 = _snap_y(min(0.98, ny1 + new_h))
                node_map[mid] = cur_node.model_copy(
                    update={"bbox": BBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2), "is_focal_subject": True}
                )

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
            # If mutable node is at a higher z_index intended to overlay the locked element, allow layer stacking
            if node_map[mid].z_index > node_map[lid].z_index:
                continue
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

    # 3. Identify colliding clusters among mutable nodes (IoU > 0.05 on the SAME z-index plane)
    colliding_clusters: list[list[str]] = []
    visited: set[str] = set()

    for i, nid1 in enumerate(mutable_ids):
        if nid1 in visited:
            continue
        cluster = [nid1]
        b1 = node_map[nid1].bbox
        z1 = node_map[nid1].z_index
        for nid2 in mutable_ids[i + 1:]:
            if nid2 in visited:
                continue
            # Layer-Aware Z-Index Stacking: elements on different z-planes are layered overlays, NOT collisions
            z2 = node_map[nid2].z_index
            if z1 != z2:
                continue
            # Parent-child containment: nested elements inside a container do not collide with parent
            if node_map[nid1].parent == nid2 or node_map[nid2].parent == nid1:
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


def detect_scene_archetype(constraints: dict | str | None = None, text: str = "") -> SceneArchetype:
    """Infer scene archetype from constraints or intent text."""
    if isinstance(constraints, str):
        text = f"{text} {constraints}".strip()
        constraints = {}
    elif constraints is None:
        constraints = {}

    if constraints.get("scene_archetype"):
        val = str(constraints["scene_archetype"]).lower()
        for arch in SceneArchetype:
            if arch.value == val or arch.name.lower() == val:
                return arch

    combined = f"{text} {constraints.get('style', '')} {constraints.get('layout_hints', '')}".lower()

    portrait_terms = (
        "portrait", "character", "person", "avatar", "face", "warrior", "hero",
        "model", "woman", "man", "girl", "boy", "cosplay", "figure", "headshot",
        "close-up face", "selfie", "protagonist", "knight", "wizard", "facial"
    )
    if any(term in combined for term in portrait_terms):
        return SceneArchetype.PORTRAIT_CHARACTER

    poster_terms = (
        "poster", "banner", "flyer", "announcement", "cover", "album",
        "typography poster", "movie poster", "event poster", "festival"
    )
    if any(term in combined for term in poster_terms):
        return SceneArchetype.GRAPHIC_POSTER

    ecommerce_terms = (
        "product", "shoe", "sneaker", "perfume", "bottle", "watch",
        "gadget", "ecommerce", "e-commerce", "packaging", "commercial", "showcase"
    )
    if any(term in combined for term in ecommerce_terms):
        return SceneArchetype.E_COMMERCE

    landscape_terms = (
        "landscape", "scenery", "mountain", "nature", "forest", "cityscape",
        "ocean", "skyline", "environment", "sunset", "valley", "panoramic"
    )
    if any(term in combined for term in landscape_terms):
        return SceneArchetype.LANDSCAPE_SCENE

    art_terms = (
        "sketch", "drawing", "illustration", "concept art", "digital painting",
        "oil painting", "watercolor", "anime", "manga", "doodle", "line art"
    )
    if any(term in combined for term in art_terms):
        return SceneArchetype.FREEFORM_ART

    return SceneArchetype.UI_LAYOUT


def layout_from_constraints(
    constraints: dict,
    *,
    text: str = "",
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> LayoutIR:
    """Build a deterministic LayoutIR synchronized with Planner constraints and DesignActions.

    Research grounding:
    - ShowUI (Lin et al., CVPR 2025): Action-token grammar for GUI elements.
    - InstructLayout (Wen et al., IEEE TPAMI 2025): Multi-modal instruction tuning for UI layout updates.
    - Cassowary (Badros et al., ACM TOCHI): Deterministic layout constraint resolution.
    - AnyDoor (Chen et al., CVPR 2024): Open-domain focal subject and facial mapping.
    """
    style = constraints.get("style")
    palette = constraints.get("palette")
    archetype = detect_scene_archetype(constraints, text=text)

    # Determine canvas dimensions from explicit args, aspect_ratio or target_device (multiples of 32 for DiT)
    if canvas_width is not None and canvas_height is not None:
        canvas_w, canvas_h = canvas_width, canvas_height
    else:
        aspect_ratio = str(constraints.get("aspect_ratio") or constraints.get("target_device") or "").lower()
        if "mobile" in aspect_ratio or "9:16" in aspect_ratio or "phone" in aspect_ratio:
            canvas_w, canvas_h = 768, 1344
        elif "desktop" in aspect_ratio or "16:9" in aspect_ratio or "wide" in aspect_ratio:
            canvas_w, canvas_h = 1344, 768
        elif "tablet" in aspect_ratio or "4:3" in aspect_ratio or "ipad" in aspect_ratio:
            canvas_w, canvas_h = 1152, 864
        else:
            canvas_w, canvas_h = 1024, 1024

    # Initialize node_map from prior turn's current_layout or supplied nodes if present, preserving multi-turn state
    current_layout = constraints.get("current_layout") or constraints.get("nodes")
    if current_layout and isinstance(current_layout, list):
        root_node = LayoutNode(
            id="root",
            type="container",
            bbox=BBox(x1=0.04, y1=0.04, x2=0.96, y2=0.96),
            style_tokens={
                "style": style,
                "palette": palette,
                "asset_mode": constraints.get("asset_mode"),
            },
        )
        node_map: dict[str, LayoutNode] = {"root": root_node}
        for item in current_layout:
            if not isinstance(item, dict):
                continue
            nid = str(item.get("id") or item.get("node_id") or "")
            if not nid or nid == "root":
                continue
            ntype = str(item.get("type") or item.get("component") or "container")
            bbox_raw = item.get("bbox") or item.get("bounds")
            if isinstance(bbox_raw, dict):
                bx1 = max(0.0, min(0.98, float(bbox_raw.get("x1", 0.1))))
                by1 = max(0.0, min(0.98, float(bbox_raw.get("y1", 0.1))))
                bx2 = max(bx1 + 0.02, min(1.0, float(bbox_raw.get("x2", 0.9))))
                by2 = max(by1 + 0.02, min(1.0, float(bbox_raw.get("y2", 0.9))))
                bbox = BBox(x1=bx1, y1=by1, x2=bx2, y2=by2)
            elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                v = [float(x) for x in bbox_raw]
                if v[2] > v[0] and v[3] > v[1]:
                    bx1 = max(0.0, min(0.98, v[0]))
                    by1 = max(0.0, min(0.98, v[1]))
                    bx2 = max(bx1 + 0.02, min(1.0, v[2]))
                    by2 = max(by1 + 0.02, min(1.0, v[3]))
                else:
                    bx1 = max(0.0, min(0.98, v[0]))
                    by1 = max(0.0, min(0.98, v[1]))
                    bx2 = max(bx1 + 0.02, min(1.0, v[0] + v[2]))
                    by2 = max(by1 + 0.02, min(1.0, v[1] + v[3]))
                bbox = BBox(x1=bx1, y1=by1, x2=bx2, y2=by2)
            else:
                bbox = BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.3)

            is_focal = bool(item.get("is_focal_subject") or ntype in ("focal_subject", "character", "photo_frame"))
            node_map[nid] = LayoutNode(
                id=nid,
                type=ntype,
                bbox=bbox,
                parent=item.get("parent", "root"),
                z_index=int(item.get("z_index", 0)),
                text=item.get("text"),
                locked=bool(item.get("locked", False)),
                is_focal_subject=is_focal,
                facial_landmarks=item.get("facial_landmarks"),
                identity_lock=bool(item.get("identity_lock", is_focal)),
            )
    else:
        # Archetype-Aware Foundation Initialization:
        # Avoid imposing web flexbox / "Continue" buttons onto portraits, posters, products, and landscapes!
        if archetype == SceneArchetype.PORTRAIT_CHARACTER:
            starter_nodes = [
                LayoutNode(
                    id="root",
                    type="container",
                    bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                    props={"style": style, "palette": palette, "asset_mode": constraints.get("asset_mode")},
                ),
                LayoutNode(
                    id="focal_subject",
                    type="focal_subject",
                    bbox=BBox(x1=0.15, y1=0.10, x2=0.85, y2=0.85),
                    parent="root",
                    z_index=0,
                    text=text[:240] or "Character Portrait",
                    is_focal_subject=True,
                    facial_landmarks={
                        "eyes": [0.50, 0.38],
                        "nose": [0.50, 0.48],
                        "mouth": [0.50, 0.60],
                    },
                    identity_lock=True,
                    locked=False,
                ),
            ]
            if text and len(text.strip()) > 0:
                starter_nodes.append(
                    LayoutNode(
                        id="caption",
                        type="caption",
                        bbox=BBox(x1=0.10, y1=0.88, x2=0.90, y2=0.96),
                        parent="root",
                        z_index=1,
                        text=text[:120],
                        locked=False,
                    )
                )
        elif archetype == SceneArchetype.GRAPHIC_POSTER:
            starter_nodes = [
                LayoutNode(
                    id="root",
                    type="container",
                    bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                    props={"style": style, "palette": palette, "asset_mode": constraints.get("asset_mode")},
                ),
                LayoutNode(
                    id="headline",
                    type="headline",
                    bbox=BBox(x1=0.08, y1=0.08, x2=0.92, y2=0.22),
                    parent="root",
                    z_index=1,
                    text=text[:120] or "FEATURED POSTER",
                    locked=False,
                ),
                LayoutNode(
                    id="focal_subject",
                    type="illustration",
                    bbox=BBox(x1=0.10, y1=0.24, x2=0.90, y2=0.82),
                    parent="root",
                    z_index=0,
                    is_focal_subject=True,
                    text=text[:240],
                    locked=False,
                ),
                LayoutNode(
                    id="subhead",
                    type="subhead",
                    bbox=BBox(x1=0.15, y1=0.85, x2=0.85, y2=0.93),
                    parent="root",
                    z_index=1,
                    text="Special Edition",
                    locked=False,
                ),
            ]
        elif archetype == SceneArchetype.E_COMMERCE:
            starter_nodes = [
                LayoutNode(
                    id="root",
                    type="container",
                    bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                    props={"style": style, "palette": palette, "asset_mode": constraints.get("asset_mode")},
                ),
                LayoutNode(
                    id="product",
                    type="product",
                    bbox=BBox(x1=0.15, y1=0.12, x2=0.85, y2=0.72),
                    parent="root",
                    z_index=0,
                    is_focal_subject=True,
                    text=text[:240] or "Product Showcase",
                    locked=False,
                ),
                LayoutNode(
                    id="badge",
                    type="badge",
                    bbox=BBox(x1=0.08, y1=0.08, x2=0.32, y2=0.16),
                    parent="root",
                    z_index=1,
                    text="Featured",
                    locked=False,
                ),
                LayoutNode(
                    id="title",
                    type="headline",
                    bbox=BBox(x1=0.10, y1=0.75, x2=0.60, y2=0.85),
                    parent="root",
                    z_index=1,
                    text=text[:120] or "Premium Selection",
                    locked=False,
                ),
                LayoutNode(
                    id="action",
                    type="button",
                    bbox=BBox(x1=0.65, y1=0.85, x2=0.92, y2=0.93),
                    parent="root",
                    z_index=1,
                    text="Order Now",
                    locked=False,
                ),
            ]
        elif archetype == SceneArchetype.LANDSCAPE_SCENE:
            starter_nodes = [
                LayoutNode(
                    id="root",
                    type="container",
                    bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                    props={"style": style, "palette": palette, "asset_mode": constraints.get("asset_mode")},
                ),
                LayoutNode(
                    id="focal_subject",
                    type="focal_subject",
                    bbox=BBox(x1=0.05, y1=0.08, x2=0.95, y2=0.92),
                    parent="root",
                    z_index=0,
                    is_focal_subject=True,
                    text=text[:240] or "Environmental Landscape Scene",
                    locked=False,
                ),
            ]
        elif archetype == SceneArchetype.FREEFORM_ART:
            starter_nodes = [
                LayoutNode(
                    id="root",
                    type="container",
                    bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                    props={"style": style, "palette": palette, "asset_mode": constraints.get("asset_mode")},
                ),
                LayoutNode(
                    id="focal_subject",
                    type="sketch",
                    bbox=BBox(x1=0.05, y1=0.05, x2=0.95, y2=0.95),
                    parent="root",
                    z_index=0,
                    is_focal_subject=True,
                    text=text[:240] or "Artistic Composition",
                    locked=False,
                ),
            ]
        else:
            # Default UI layout
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
        node_map = {node.id: node for node in starter_nodes}

    # Apply VLA DesignAction elements from Planner delta
    actions = constraints.get("actions") or []
    if isinstance(actions, list):
        has_custom_focal = any(
            (isinstance(a, dict) and (a.get("source_asset_id") or a.get("component_type") in ("focal_subject", "character", "background")))
            or (hasattr(a, "source_asset_id") and getattr(a, "source_asset_id"))
            or (hasattr(a, "component_type") and getattr(a, "component_type") in ("focal_subject", "character", "background"))
            for a in actions
        )
        if has_custom_focal and "focal_subject" in node_map and node_map["focal_subject"].source_asset_id is None:
            del node_map["focal_subject"]

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
                z_idx = int(act_dict.get("z_index") or params.get("z_index") or props.get("z_index", 0))

                is_focal = bool(
                    act_dict.get("is_focal_subject")
                    or params.get("is_focal_subject")
                    or comp_type in ("focal_subject", "character", "photo_frame", "product")
                )

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

                if is_focal:
                    # Grounded research (NVIDIA AnyDoor CVPR 2024 / ConsisID):
                    # Prevent latent token collapse. With VAE downsampling f=16 and DiT patch stride=2,
                    # any face or focal subject < 128px collapses into < 4 latent tokens, destroying facial features.
                    # Enforce strict spatial floor of at least 128px (or 12.5% of canvas).
                    min_w = max(128.0 / canvas_w, 0.125)
                    min_h = max(128.0 / canvas_h, 0.125)
                    bw_cur = new_bbox.x2 - new_bbox.x1
                    bh_cur = new_bbox.y2 - new_bbox.y1
                    if bw_cur < min_w or bh_cur < min_h:
                        bw_adj = max(bw_cur, min_w)
                        bh_adj = max(bh_cur, min_h)
                        adj_x2 = min(1.0, new_bbox.x1 + bw_adj)
                        adj_y2 = min(1.0, new_bbox.y1 + bh_adj)
                        adj_x1 = max(0.0, adj_x2 - bw_adj)
                        adj_y1 = max(0.0, adj_y2 - bh_adj)
                        new_bbox = BBox(x1=round(adj_x1, 4), y1=round(adj_y1, 4), x2=round(adj_x2, 4), y2=round(adj_y2, 4))

                is_locked = bool(
                    act_dict.get("locked")
                    or params.get("locked")
                    or act_dict.get("identity_lock")
                    or params.get("identity_lock")
                    or is_focal
                )
                src_asset_id = act_dict.get("source_asset_id") or params.get("source_asset_id")
                raw_src_crop = act_dict.get("source_crop_bbox") or params.get("source_crop_bbox")
                src_crop = tuple(float(v) for v in raw_src_crop) if raw_src_crop and len(raw_src_crop) == 4 else None

                node_map[comp_id] = LayoutNode(
                    id=comp_id,
                    type=comp_type,
                    bbox=new_bbox,
                    parent="root",
                    z_index=z_idx,
                    text=text_content,
                    locked=is_locked,
                    is_focal_subject=is_focal,
                    facial_landmarks=act_dict.get("facial_landmarks") or params.get("facial_landmarks"),
                    identity_lock=bool(act_dict.get("identity_lock") or params.get("identity_lock", is_focal)),
                    source_asset_id=src_asset_id,
                    source_crop_bbox=src_crop,
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
                src_asset_id = act_dict.get("source_asset_id") or params.get("source_asset_id")
                raw_src_crop = act_dict.get("source_crop_bbox") or params.get("source_crop_bbox")
                src_crop = tuple(float(v) for v in raw_src_crop) if raw_src_crop and len(raw_src_crop) == 4 else None
                updates: dict[str, Any] = {"locked": True, "identity_lock": True}
                if src_asset_id:
                    updates["source_asset_id"] = src_asset_id
                if src_crop:
                    updates["source_crop_bbox"] = src_crop

                if target_id and target_id in node_map:
                    node_map[target_id] = node_map[target_id].model_copy(update=updates)
                bbox_raw = act_dict.get("bbox") or params.get("bbox")
                if bbox_raw and len(bbox_raw) == 4:
                    try:
                        bx, by, bw, bh = [float(v) for v in bbox_raw]
                        for nid, node in list(node_map.items()):
                            if nid == "root":
                                continue
                            nb = node.bbox
                            ncx, ncy = (nb.x1 + nb.x2) / 2.0, (nb.y1 + nb.y2) / 2.0
                            if bx <= ncx <= (bx + bw) and by <= ncy <= (by + bh):
                                node_map[nid] = node.model_copy(update=updates)
                    except Exception:
                        pass

            elif act_type == "SET_LAYOUT_DIRECTION":
                direction = str(act_dict.get("layout_direction") or params.get("layout_direction") or "row").lower()
                constraints["layout_direction"] = direction
                if target_id and target_id in node_map:
                    node_map[target_id].props["layout_direction"] = direction
                elif "root" in node_map:
                    node_map["root"].props["layout_direction"] = direction

            elif act_type == "REORDER_HIERARCHY":
                if target_id and target_id in node_map:
                    pos = str(act_dict.get("hierarchy_position") or params.get("hierarchy_position") or "front").lower()
                    current_z = [n.z_index for n in node_map.values()]
                    max_z = max(current_z) if current_z else 0
                    min_z = min(current_z) if current_z else 0
                    new_z = max_z + 1 if pos in ("front", "top", "up") else min_z - 1
                    node_map[target_id] = node_map[target_id].model_copy(update={"z_index": new_z})

            elif act_type == "SET_STYLE":
                new_style = act_dict.get("style") or params.get("style")
                if new_style and "root" in node_map:
                    tokens = dict(node_map["root"].style_tokens)
                    tokens["style"] = str(new_style)
                    node_map["root"] = node_map["root"].model_copy(update={"style_tokens": tokens})

            elif act_type == "SET_PALETTE":
                new_pal = act_dict.get("palette") or params.get("palette")
                if isinstance(new_pal, list) and "root" in node_map:
                    tokens = dict(node_map["root"].style_tokens)
                    tokens["palette"] = [str(c) for c in new_pal]
                    node_map["root"] = node_map["root"].model_copy(update={"style_tokens": tokens})

            elif act_type == "SET_ASSET_MODE":
                new_mode = act_dict.get("asset_mode") or params.get("asset_mode")
                if new_mode in ("opaque", "transparent") and "root" in node_map:
                    tokens = dict(node_map["root"].style_tokens)
                    tokens["asset_mode"] = str(new_mode)
                    node_map["root"] = node_map["root"].model_copy(update={"style_tokens": tokens})

    # Synchronize locked_regions directly with LayoutNode.locked BEFORE Flexbox resolution
    locked_regions = constraints.get("locked_regions") or []
    if isinstance(locked_regions, list):
        for lr in locked_regions:
            lr_id = lr.get("id") if isinstance(lr, dict) else getattr(lr, "id", None)
            bbox_raw = lr.get("bbox") if isinstance(lr, dict) else getattr(lr, "bbox", None)
            if lr_id and lr_id in node_map and node_map[lr_id].type not in ("background", "container"):
                node_map[lr_id] = node_map[lr_id].model_copy(update={"locked": True, "identity_lock": True})
            elif bbox_raw and len(bbox_raw) == 4:
                try:
                    bx, by, bw, bh = [float(v) for v in bbox_raw]
                    for nid, node in list(node_map.items()):
                        if nid == "root" or node.type in ("background", "container"):
                            continue
                        nb = node.bbox
                        cur_w, cur_h = nb.x2 - nb.x1, nb.y2 - nb.y1
                        if abs(nb.x1 - bx) < 0.10 and abs(nb.y1 - by) < 0.10 and abs(cur_w - bw) < 0.20 and abs(cur_h - bh) < 0.20:
                            node_map[nid] = node.model_copy(update={"locked": True, "identity_lock": True})
                except Exception:
                    pass

    # Two-way sync: Ensure all locked / identity-locked nodes are registered in constraints["locked_regions"]
    final_locked_regions = list(constraints.get("locked_regions") or [])
    for nid, node in node_map.items():
        if nid != "root" and (node.locked or getattr(node, "identity_lock", False)):
            box_coords = [round(node.bbox.x1, 4), round(node.bbox.y1, 4), round(node.bbox.x2 - node.bbox.x1, 4), round(node.bbox.y2 - node.bbox.y1, 4)]
            match = None
            for r in final_locked_regions:
                r_box = r.get("bbox") if isinstance(r, dict) else getattr(r, "bbox", None)
                if r_box == box_coords or (isinstance(r, dict) and r.get("id") == nid) or getattr(r, "id", None) == nid:
                    match = r
                    break
            if match is None:
                final_locked_regions.append({
                    "id": nid,
                    "bbox": box_coords,
                    "reason": f"Lock for {nid} ({node.type})",
                    "source_asset_id": getattr(node, "source_asset_id", None),
                    "source_crop_bbox": getattr(node, "source_crop_bbox", None),
                })
            elif isinstance(match, dict):
                if not match.get("source_asset_id") and getattr(node, "source_asset_id", None):
                    match["source_asset_id"] = getattr(node, "source_asset_id", None)
                if not match.get("source_crop_bbox") and getattr(node, "source_crop_bbox", None):
                    match["source_crop_bbox"] = getattr(node, "source_crop_bbox", None)
    constraints["locked_regions"] = final_locked_regions

    # Apply deterministic Flexbox partitioner (with strict locked-node exemption and 8pt grid snapping)
    layout_conflicts: list[str] = []
    node_map = _resolve_flex_layout(node_map, constraints, canvas_w=canvas_w, canvas_h=canvas_h, conflicts=layout_conflicts)

    return project_layout(
        LayoutIR(
            scene_archetype=archetype.value,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            nodes=list(node_map.values()),
            diagnostics=layout_conflicts,
        )
    ).layout
