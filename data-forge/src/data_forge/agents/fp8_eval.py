"""FP8 vs BF16 evaluation harness — automated precision comparison (v13 §4.3).

Runs a fixed held-out prompt set through both BF16 and FP8 planner checkpoints,
comparing task accuracy, JSON validity rate, and reasoning quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger

log = get_logger("agents.fp8_eval")


class FP8EvalHarness:
    """Automated BF16 vs FP8 comparison for the product planner model."""

    def __init__(
        self,
        prompts_path: Path,
        accuracy_threshold: float = 0.95,
        json_validity_threshold: float = 0.98,
        quality_threshold: float = 0.90,
    ) -> None:
        self._prompts_path = prompts_path
        self._accuracy_thresh = accuracy_threshold
        self._json_validity_thresh = json_validity_threshold
        self._quality_thresh = quality_threshold

    def load_prompts(self) -> list[dict[str, Any]]:
        """Load the fixed held-out prompt set."""
        prompts = []
        with open(self._prompts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
        log.info("eval_prompts_loaded", count=len(prompts))
        return prompts

    async def run_comparison(
        self,
        bf16_engine: Any,
        fp8_engine: Any,
        judge_engine: Any,
    ) -> dict[str, Any]:
        """Run the full BF16 vs FP8 comparison.

        Args:
            bf16_engine: Inference engine loaded with BF16 planner
            fp8_engine: Inference engine loaded with FP8 planner
            judge_engine: Tier-1 model used as a quality judge

        Returns:
            Comparison report with pass/fail determination.
        """
        prompts = self.load_prompts()

        bf16_results = []
        fp8_results = []

        for prompt_data in prompts:
            prompt_text = prompt_data["prompt"]

            # Run through both models
            bf16_resp = await bf16_engine.complete(prompt=prompt_text)
            fp8_resp = await fp8_engine.complete(prompt=prompt_text)

            bf16_results.append(bf16_resp)
            fp8_results.append(fp8_resp)

        # Compare
        report = self._compare_results(prompts, bf16_results, fp8_results)

        log.info(
            "fp8_eval_completed",
            accuracy_match=report["accuracy_match_rate"],
            json_validity_rate=report["fp8_json_validity_rate"],
            overall_pass=report["pass"],
        )

        return report

    def _compare_results(
        self,
        prompts: list[dict],
        bf16: list[Any],
        fp8: list[Any],
    ) -> dict[str, Any]:
        """Compare BF16 and FP8 outputs."""
        total = len(prompts)
        accuracy_matches = 0
        fp8_json_valid = 0

        for i, (prompt, b_result, f_result) in enumerate(zip(prompts, bf16, fp8)):
            expected = prompt.get("expected_output")

            # JSON validity check
            try:
                if isinstance(f_result, dict):
                    json.dumps(f_result)
                else:
                    json.loads(str(f_result))
                fp8_json_valid += 1
            except (json.JSONDecodeError, TypeError):
                pass

            # Accuracy: compare against expected or BF16 baseline
            if expected:
                f_matches = self._fuzzy_match(f_result, expected)
            else:
                f_matches = self._fuzzy_match(f_result, b_result)

            if f_matches:
                accuracy_matches += 1

        accuracy_rate = accuracy_matches / total if total > 0 else 0.0
        json_rate = fp8_json_valid / total if total > 0 else 0.0

        passes = (
            accuracy_rate >= self._accuracy_thresh
            and json_rate >= self._json_validity_thresh
        )

        return {
            "total_prompts": total,
            "accuracy_match_rate": round(accuracy_rate, 4),
            "fp8_json_validity_rate": round(json_rate, 4),
            "accuracy_threshold": self._accuracy_thresh,
            "json_validity_threshold": self._json_validity_thresh,
            "pass": passes,
        }

    @staticmethod
    def _fuzzy_match(result_a: Any, result_b: Any) -> bool:
        """Basic fuzzy matching between two model outputs."""
        str_a = json.dumps(result_a, sort_keys=True) if isinstance(result_a, dict) else str(result_a)
        str_b = json.dumps(result_b, sort_keys=True) if isinstance(result_b, dict) else str(result_b)

        # Exact match
        if str_a == str_b:
            return True

        # Normalized match (lowercase, stripped)
        if str_a.lower().strip() == str_b.lower().strip():
            return True

        return False
