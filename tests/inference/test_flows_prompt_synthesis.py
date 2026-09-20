"""E1: Tests for flows.py::_synthesize_dpo_prompt \u2014 the canonical prompt
synthesis used by both finalize() (passed to Polish backends) and
critique_pass() (stored in DPO preference pair records). Added as part
of the Phase 3 test coverage pass after the P1 fix extracted the
duplicated inline synthesis into a shared function.
"""

from __future__ import annotations

from krisna_inference.orchestrator.design_state import (
    Constraints,
    ConversationTurn,
    DesignState,
)
from krisna_inference.orchestrator.flows import _synthesize_dpo_prompt


class TestSynthesizeDpoPrompt:
    """_synthesize_dpo_prompt must never return None or empty string
    (the DPO training loop calls encode_prompt() on the result, and an
    empty string is a degenerate conditioning input for a diffusion model).
    """

    def test_user_intent_and_full_constraints(self):
        """Full case: history + style + palette + layout_hints."""
        state = DesignState()
        state.conversation_history = [
            ConversationTurn(role="user", content="make me a dark crypto dashboard"),
            ConversationTurn(role="planner", content="Got it \u2014 dark themed, trading focus"),
            ConversationTurn(role="user", content="add a sidebar with nav icons"),
        ]
        state.constraints = Constraints(
            style="dark minimalist",
            palette=["#1A1A2E", "#16213E"],
            layout_hints="sidebar navigation",
        )
        result = _synthesize_dpo_prompt(state)
        # Should use LAST user turn as intent anchor
        assert result.startswith("add a sidebar with nav icons")
        assert "style: dark minimalist" in result
        assert "#1A1A2E" in result
        assert "layout: sidebar navigation" in result

    def test_user_intent_only_no_constraints(self):
        """No constraints populated \u2014 result is just the last user turn."""
        state = DesignState()
        state.conversation_history = [
            ConversationTurn(role="user", content="a clean login screen"),
        ]
        state.constraints = Constraints()  # all None / empty
        result = _synthesize_dpo_prompt(state)
        assert result == "a clean login screen"

    def test_constraints_only_no_history(self):
        """No conversation history \u2014 result uses constraints with a prefix."""
        state = DesignState()
        state.constraints = Constraints(
            style="material dark",
            palette=["#121212"],
            layout_hints=None,
        )
        result = _synthesize_dpo_prompt(state)
        assert result.startswith("High quality UI design")
        assert "style: material dark" in result
        assert "#121212" in result

    def test_no_history_no_constraints_is_never_empty(self):
        """Absolute fallback \u2014 nothing in state at all."""
        state = DesignState()
        result = _synthesize_dpo_prompt(state)
        assert result == "High quality UI design"
        assert result is not None
        assert len(result) > 0

    def test_only_planner_turns_no_user_turns(self):
        """Edge case: history exists but has no user turns (all planner).
        Should behave the same as no history."""
        state = DesignState()
        state.conversation_history = [
            ConversationTurn(role="planner", content="I will generate a sketch"),
        ]
        state.constraints = Constraints(style="glassmorphism")
        result = _synthesize_dpo_prompt(state)
        assert "style: glassmorphism" in result
        assert result.startswith("High quality UI design")

    def test_palette_empty_list_excluded(self):
        """Empty palette list should not produce 'palette: ' in result."""
        state = DesignState()
        state.conversation_history = [
            ConversationTurn(role="user", content="landing page"),
        ]
        state.constraints = Constraints(style="minimalist", palette=[])
        result = _synthesize_dpo_prompt(state)
        assert "palette:" not in result
        assert "style: minimalist" in result

    def test_result_is_deterministic(self):
        """Both callers (finalize + critique_pass) get identical output
        for the same state \u2014 the key correctness property of unifying them."""
        state = DesignState()
        state.conversation_history = [
            ConversationTurn(role="user", content="e-commerce product page"),
        ]
        state.constraints = Constraints(style="light airy", palette=["#FFFFFF", "#F5F5F5"])
        r1 = _synthesize_dpo_prompt(state)
        r2 = _synthesize_dpo_prompt(state)
        assert r1 == r2
