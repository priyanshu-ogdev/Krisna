"""Pure, dependency-neutral merge logic for a Planner turn's
`constraint_updates` delta onto a plain constraints dict.

Deliberately has ZERO dependency on design_state.py or pydantic — this
module exists specifically so swap_orchestrator.py can use it without
violating its own stated architectural boundary ("the orchestrator
itself... knows nothing about DesignState — it only knows about
ModelBackend/Tier/VRAM", per flows.py's module docstring). Both
swap_orchestrator.py (to build fresh kwargs for the SAME turn's Sketch
call) and flows.py (to update persisted DesignState) call this one
function, so the two call sites cannot drift out of sync with each
other's merge logic — which is exactly the bug this module was created
to fix (see swap_orchestrator.py's run_conversational_turn docstring for
the concrete symptom this closes).
"""

from __future__ import annotations


def _normalize_and_clamp_bbox(bbox) -> list[float] | None:
    """Clamps bounding boxes to valid screen bounds to prevent out-of-bounds rendering crashes."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        raw = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None

    # Normalized [0.0, 1.0] coordinates
    if all(0.0 <= v <= 1.05 for v in raw):
        x = max(0.0, min(1.0, raw[0]))
        y = max(0.0, min(1.0, raw[1]))
        w = max(0.0, min(1.0 - x, raw[2]))
        h = max(0.0, min(1.0 - y, raw[3]))
        return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

    # Pixel coordinates (> 1.0)
    x = max(0.0, raw[0])
    y = max(0.0, raw[1])
    w = max(0.0, raw[2])
    h = max(0.0, raw[3])
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def _apply_single_action(merged: dict, action) -> None:
    """Apply a single VLA DesignAction onto the merged constraints dict."""
    if hasattr(action, "model_dump"):
        action_dict = action.model_dump()
    elif isinstance(action, dict):
        action_dict = action
    else:
        return

    act_type = str(action_dict.get("action_type") or "").upper()
    params = action_dict.get("parameters") if isinstance(action_dict.get("parameters"), dict) else {}

    if act_type == "SET_STYLE":
        style_val = action_dict.get("style") or params.get("style")
        if style_val is not None:
            merged["style"] = str(style_val)

    elif act_type == "SET_PALETTE":
        pal_val = action_dict.get("palette") or params.get("palette")
        if isinstance(pal_val, list):
            merged["palette"] = [str(c) for c in pal_val]

    elif act_type == "LOCK_REGION":
        bbox = action_dict.get("bbox") or params.get("bbox")
        reason = action_dict.get("reason") or params.get("reason") or "Action lock"
        clamped = _normalize_and_clamp_bbox(bbox)
        if clamped is not None:
            current_locked = list(merged.get("locked_regions") or [])
            current_locked.append({"bbox": clamped, "reason": str(reason)})
            merged["locked_regions"] = current_locked

    elif act_type == "RESIZE":
        target = action_dict.get("target_id") or params.get("target_id")
        bbox = action_dict.get("bbox") or params.get("bbox")
        clamped = _normalize_and_clamp_bbox(bbox)
        hint = f"Resize {target} to {clamped}" if target else f"Resize element to {clamped}"
        existing_hints = merged.get("layout_hints") or ""
        merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "MOVE_COMPONENT":
        target = action_dict.get("target_id") or params.get("target_id")
        bbox = action_dict.get("bbox") or params.get("bbox")
        clamped = _normalize_and_clamp_bbox(bbox)
        hint = f"Move {target} to {clamped}" if target else f"Move element to {clamped}"
        existing_hints = merged.get("layout_hints") or ""
        merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "ADD_COMPONENT":
        comp_type = action_dict.get("component_type") or params.get("component_type") or "component"
        bbox = action_dict.get("bbox") or params.get("bbox")
        clamped = _normalize_and_clamp_bbox(bbox)
        hint = f"Add {comp_type} at {clamped}" if clamped else f"Add {comp_type}"
        existing_hints = merged.get("layout_hints") or ""
        merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "REMOVE_COMPONENT":
        target = action_dict.get("target_id") or params.get("target_id")
        if target:
            hint = f"Remove {target}"
            existing_hints = merged.get("layout_hints") or ""
            merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "SET_TEXT":
        target = action_dict.get("target_id") or params.get("target_id")
        text = action_dict.get("text_content") or params.get("text_content")
        if target and text:
            hint = f"Set text of {target} to '{text}'"
            existing_hints = merged.get("layout_hints") or ""
            merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "SET_LAYOUT_DIRECTION":
        direction = action_dict.get("layout_direction") or params.get("layout_direction") or "row"
        target = action_dict.get("target_id") or params.get("target_id")
        hint = f"Set {target} direction to {direction}" if target else f"Set layout direction to {direction}"
        existing_hints = merged.get("layout_hints") or ""
        merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "REORDER_HIERARCHY":
        pos = action_dict.get("hierarchy_position") or params.get("hierarchy_position") or "front"
        target = action_dict.get("target_id") or params.get("target_id")
        hint = f"Reorder {target} to {pos}" if target else f"Reorder hierarchy to {pos}"
        existing_hints = merged.get("layout_hints") or ""
        merged["layout_hints"] = f"{existing_hints}; {hint}".strip("; ")

    elif act_type == "SET_ASSET_MODE":
        mode_val = action_dict.get("asset_mode") or params.get("asset_mode")
        if mode_val in ("opaque", "transparent"):
            merged["asset_mode"] = str(mode_val)


def apply_constraint_updates(constraints: dict, constraint_updates) -> dict:
    """Returns a NEW dict — never mutates `constraints` — with a
    Planner-produced `constraint_updates` delta or action grammar list merged on top. Shape of
    `constraints` matches DesignState.constraints.model_dump(); shape of
    `constraint_updates` matches the `constraint_updates` key inside a
    Planner backend's `design_state_delta` output (see
    planner_backend.py's `_extract_json_delta`), or a list of DesignAction objects.

    Any key/type this delta doesn't recognize or validate is left alone —
    same permissive-but-not-silent posture as the original inline version
    of this logic in flows.py: an update that doesn't parse is simply not
    applied, not raised as an error (the Planner is a frozen, not-trained
    model whose JSON output this project cannot guarantee is always
    perfectly shaped — see planner_backend.py's generate-validate-retry
    loop for the same posture at the JSON-parsing layer one level up).
    """
    merged = dict(constraints)

    # Direct list of DesignAction items
    if isinstance(constraint_updates, list):
        merged["actions"] = list(constraint_updates)
        for act in constraint_updates:
            _apply_single_action(merged, act)
        return merged

    if not isinstance(constraint_updates, dict):
        return merged

    if constraint_updates.get("style") is not None:
        merged["style"] = str(constraint_updates["style"])

    if isinstance(constraint_updates.get("palette"), list):
        merged["palette"] = [str(c) for c in constraint_updates["palette"]]

    if constraint_updates.get("layout_hints") is not None:
        merged["layout_hints"] = str(constraint_updates["layout_hints"])

    if constraint_updates.get("asset_mode") in ("opaque", "transparent"):
        merged["asset_mode"] = str(constraint_updates["asset_mode"])

    if isinstance(constraint_updates.get("locked_regions"), list):
        new_locked = []
        for r in constraint_updates["locked_regions"]:
            if isinstance(r, dict) and "bbox" in r and "reason" in r:
                new_locked.append(r)
            elif hasattr(r, "model_dump"):
                new_locked.append(r.model_dump())
        if new_locked:
            merged["locked_regions"] = new_locked

    # Also apply nested actions list if provided in constraint_updates dict
    if isinstance(constraint_updates.get("actions"), list):
        merged["actions"] = list(constraint_updates["actions"])
        for act in constraint_updates["actions"]:
            _apply_single_action(merged, act)

    return merged
