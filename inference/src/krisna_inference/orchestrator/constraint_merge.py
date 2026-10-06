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


def apply_constraint_updates(constraints: dict, constraint_updates) -> dict:
    """Returns a NEW dict — never mutates `constraints` — with a
    Planner-produced `constraint_updates` delta merged on top. Shape of
    `constraints` matches DesignState.constraints.model_dump(); shape of
    `constraint_updates` matches the `constraint_updates` key inside a
    Planner backend's `design_state_delta` output (see
    planner_backend.py's `_extract_json_delta`).

    Any key/type this delta doesn't recognize or validate is left alone —
    same permissive-but-not-silent posture as the original inline version
    of this logic in flows.py: an update that doesn't parse is simply not
    applied, not raised as an error (the Planner is a frozen, not-trained
    model whose JSON output this project cannot guarantee is always
    perfectly shaped — see planner_backend.py's generate-validate-retry
    loop for the same posture at the JSON-parsing layer one level up).
    """
    merged = dict(constraints)
    if not isinstance(constraint_updates, dict):
        return merged

    if constraint_updates.get("style") is not None:
        merged["style"] = str(constraint_updates["style"])

    if isinstance(constraint_updates.get("palette"), list):
        merged["palette"] = [str(c) for c in constraint_updates["palette"]]

    if constraint_updates.get("layout_hints") is not None:
        merged["layout_hints"] = str(constraint_updates["layout_hints"])

    if isinstance(constraint_updates.get("locked_regions"), list):
        new_locked = []
        for r in constraint_updates["locked_regions"]:
            if isinstance(r, dict) and "bbox" in r and "reason" in r:
                new_locked.append(r)
            elif hasattr(r, "model_dump"):
                new_locked.append(r.model_dump())
        if new_locked:
            merged["locked_regions"] = new_locked

    return merged
