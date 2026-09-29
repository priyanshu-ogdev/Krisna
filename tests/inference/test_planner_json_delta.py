from __future__ import annotations

from krisna_inference.backends.planner_backend import PlannerBackend


class TestExtractJsonDelta:
    def test_extracts_trailing_json_object(self):
        text = (
            'Sure, I can help with that. Let\'s go with a minimalist style.\n'
            '{"stage": "conversing", "constraint_updates": {"style": "minimalist"}, '
            '"tool_call": null, "reasoning_note": "User wants minimalist."}'
        )
        delta, error = PlannerBackend._extract_json_delta(text)
        assert error is None
        assert delta["stage"] == "conversing"
        assert delta["constraint_updates"] == {"style": "minimalist"}

    def test_handles_nested_braces_in_constraint_updates(self):
        text = (
            '{"stage": "sketching", "constraint_updates": {"palette": {"primary": "#000"}}, '
            '"tool_call": "update_sketch", "reasoning_note": "ok"}'
        )
        delta, error = PlannerBackend._extract_json_delta(text)
        assert error is None
        assert delta["constraint_updates"]["palette"]["primary"] == "#000"

    def test_missing_stage_key_is_an_error(self):
        text = '{"constraint_updates": {}, "tool_call": null}'
        delta, error = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert "stage" in error

    def test_invalid_stage_value_is_an_error(self):
        text = '{"stage": "not_a_real_stage", "constraint_updates": {}}'
        delta, error = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert "stage" in error.lower()

    def test_no_json_object_at_all_is_an_error(self):
        text = "I think a minimalist style would work well here."
        delta, error = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert error is not None

    def test_malformed_json_is_an_error_not_a_crash(self):
        text = '{"stage": "conversing", "constraint_updates": {clearly not json}'
        delta, error = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert "parse error" in error.lower()

    def test_missing_optional_keys_get_sane_defaults(self):
        text = '{"stage": "finalized"}'
        delta, error = PlannerBackend._extract_json_delta(text)
        assert error is None
        assert delta["constraint_updates"] == {}
        assert delta["tool_call"] is None
        assert delta["reasoning_note"] == ""

    def test_top_level_array_is_rejected(self):
        text = "[1, 2, 3]"
        delta, error = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert error is not None

    def test_all_five_valid_stages_accepted(self):
        for stage in ("conversing", "sketching", "finalizing", "finalized", "critiquing"):
            delta, error = PlannerBackend._extract_json_delta(f'{{"stage": "{stage}"}}')
            assert error is None, f"stage={stage} should be valid, got error: {error}"
            assert delta["stage"] == stage
