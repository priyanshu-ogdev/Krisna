from __future__ import annotations

from krisna_inference.backends.planner_backend import PlannerBackend


class TestExtractJsonDelta:
    def test_extracts_trailing_json_object(self):
        text = (
            'Sure, I can help with that. Let\'s go with a minimalist style.\n'
            '{"stage": "conversing", "constraint_updates": {"style": "minimalist"}, '
            '"tool_call": null, "reasoning_note": "User wants minimalist."}'
        )
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert error is None
        assert delta["stage"] == "conversing"
        assert delta["constraint_updates"] == {"style": "minimalist"}

    def test_handles_nested_braces_in_constraint_updates(self):
        text = (
            '{"stage": "sketching", "constraint_updates": {"palette": {"primary": "#000"}}, '
            '"tool_call": "update_sketch", "reasoning_note": "ok"}'
        )
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert error is None
        assert delta["constraint_updates"]["palette"]["primary"] == "#000"

    def test_missing_stage_key_is_an_error(self):
        text = '{"constraint_updates": {}, "tool_call": null}'
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert "stage" in error

    def test_invalid_stage_value_is_an_error(self):
        text = '{"stage": "not_a_real_stage", "constraint_updates": {}}'
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert "stage" in error.lower()

    def test_no_json_object_at_all_is_an_error(self):
        text = "I think a minimalist style would work well here."
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert error is not None

    def test_malformed_json_is_an_error_not_a_crash(self):
        text = '{"stage": "conversing", "constraint_updates": {clearly not json}'
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert "parse error" in error.lower()

    def test_missing_optional_keys_get_sane_defaults(self):
        text = '{"stage": "finalized"}'
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert error is None
        assert delta["constraint_updates"] == {}
        assert delta["tool_call"] is None
        assert delta["reasoning_note"] == ""

    def test_top_level_array_is_rejected(self):
        text = "[1, 2, 3]"
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert delta is None
        assert error is not None

    def test_all_five_valid_stages_accepted(self):
        for stage in ("conversing", "sketching", "finalizing", "finalized", "critiquing"):
            delta, error, span = PlannerBackend._extract_json_delta(f'{{"stage": "{stage}"}}')
            assert error is None, f"stage={stage} should be valid, got error: {error}"
            assert delta["stage"] == stage


class TestExtractJsonDeltaSpan:
    """_extract_json_delta previously discarded the JSON object's
    character span once used internally to locate the object to parse —
    the caller had no way to strip that span back out of the generation,
    so the raw model output (JSON delta still attached) flowed all the
    way through to the chat UI unmodified.
    """

    def test_span_covers_exactly_the_matched_json_object(self):
        text = (
            "Sure, let's go with a minimalist style.\n"
            '{"stage": "conversing", "constraint_updates": {}, '
            '"tool_call": null, "reasoning_note": "ok"}'
        )
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert error is None
        start, end = span
        assert text[start] == "{"
        assert text[end - 1] == "}"
        import json

        assert json.loads(text[start:end])["stage"] == delta["stage"]

    def test_span_handles_nested_braces_correctly(self):
        text = '{"stage": "sketching", "constraint_updates": {"palette": {"primary": "#000"}}, "tool_call": null, "reasoning_note": "ok"}'
        delta, error, span = PlannerBackend._extract_json_delta(text)
        start, end = span
        assert start == 0
        assert end == len(text)

    def test_span_is_none_on_failure(self):
        delta, error, span = PlannerBackend._extract_json_delta("no json here at all")
        assert delta is None
        assert span is None


class TestPlannerRunTextStripping:
    """The actual bug fix: the text returned to callers (flows.py ->
    conversation_history[].content -> the chat UI) must have the JSON
    delta removed, not just the parsed delta available separately.
    """

    def test_json_delta_is_stripped_from_conversational_text(self):
        raw = (
            "Sure, I can help with that! Here's a settings screen with a dark toggle.\n"
            '{"stage": "sketching", "constraint_updates": {"style": "minimalist"}, '
            '"tool_call": null, "reasoning_note": "User wants a settings screen."}'
        )
        text = raw
        delta, error, span = PlannerBackend._extract_json_delta(text)
        assert delta is not None
        start, end = span
        conversational_text = (text[:start] + text[end:]).strip()

        assert "Here's a settings screen with a dark toggle." in conversational_text
        assert '"stage"' not in conversational_text
        assert "constraint_updates" not in conversational_text
        assert conversational_text == "Sure, I can help with that! Here's a settings screen with a dark toggle."


class TestPlannerSystemPrompt:
    def test_system_prompt_includes_original_intent_when_present(self):
        backend = object.__new__(PlannerBackend)
        prompt = backend._build_system_prompt(
            constraints={"style": "glassmorphic", "original_intent": "crypto portfolio tracker"},
            retrieved=[],
        )
        assert "Original user goal: 'crypto portfolio tracker'" in prompt
        assert "glassmorphic" in prompt

    def test_system_prompt_omits_intent_clause_when_not_present(self):
        backend = object.__new__(PlannerBackend)
        prompt = backend._build_system_prompt(
            constraints={"style": "glassmorphic"},
            retrieved=[],
        )
        assert "Original user goal:" not in prompt
        assert "glassmorphic" in prompt
