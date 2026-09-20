"""Internal eval set runner for model comparison (registry watcher §5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger

log = get_logger("registry.evaluator")


class ModelEvaluator:
    """Side-by-side model evaluation using a fixed internal eval set."""

    def __init__(self, eval_set_path: Path) -> None:
        self._eval_set_path = eval_set_path

    def load_eval_set(self) -> list[dict[str, Any]]:
        items = []
        with open(self._eval_set_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    async def compare_models(
        self, current_engine: Any, candidate_engine: Any
    ) -> dict[str, Any]:
        """Run both models on the eval set and compare metrics."""
        eval_set = self.load_eval_set()
        current_scores: list[float] = []
        candidate_scores: list[float] = []

        for item in eval_set:
            prompt = item.get("prompt", "")
            expected = item.get("expected", {})

            c_result = await current_engine.complete(prompt=prompt)
            n_result = await candidate_engine.complete(prompt=prompt)

            c_score = self._score_result(c_result, expected)
            n_score = self._score_result(n_result, expected)

            current_scores.append(c_score)
            candidate_scores.append(n_score)

        avg_current = sum(current_scores) / len(current_scores) if current_scores else 0
        avg_candidate = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0

        return {
            "eval_set_size": len(eval_set),
            "current_avg_score": round(avg_current, 4),
            "candidate_avg_score": round(avg_candidate, 4),
            "improvement": round(avg_candidate - avg_current, 4),
            "recommendation": "swap" if avg_candidate > avg_current * 1.02 else "hold",
        }

    @staticmethod
    def _score_result(result: Any, expected: dict) -> float:
        """Basic scoring: proportion of expected keys that match."""
        if not expected or not isinstance(result, dict):
            return 0.5
        matches = sum(1 for k, v in expected.items() if result.get(k) == v)
        return matches / len(expected) if expected else 0.5
