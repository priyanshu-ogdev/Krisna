"""Stage 1.6: Planner Conversation Synthesis — closes the Planner's data gap.

Before this stage, data-forge had no path producing any training data for
the Planner (Qwen3.5-9B) at all — the entire pipeline was image-
preprocessing focused, and nothing public teaches "UI design conversation
-> JSON state delta" directly (see datasets.yaml's glaive_function_calling
and xlam_function_calling entries for the format-teacher datasets, used
as style reference for well-formed tool-call output, general-domain only).

The content layer here uses real, human-authored UICrit critique text
(via s01_5_uicrit_join's `critique_source == "uicrit_human"` records) as
seed material, generating a plausible surrounding conversation via Tier-1
and validating every output against both Pydantic's SynthesizedConversation
schema (client-side, at generation time) and the standalone JSON Schema in
configs/schemas/synthesized_conversation.json (defense-in-depth — same
principle the audit report specified: "every generated JSON delta must
parse against the same schema the Design State Manager uses in
production — reject anything that doesn't").

Output: planner_data/conversations/*.jsonl, one JSON object per line,
each a full {turns, resulting_delta, seed_critique_record_id} example.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.data.schema_validator import SchemaValidator
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s01_6")


@register_stage("s01_6_planner_synthesis")
class PlannerSynthesisStage(Stage):
    name = "s01_6_planner_synthesis"
    requires: ClassVar[tuple[str, ...]] = ("s01_5_uicrit_join",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        if engine is None:
            log.warning("planner_synthesis_skipped", reason="no engine")
            return result

        stage_cfg = config.get_stage("s01_6_planner_synthesis")
        target_count = stage_cfg.get("target_conversation_count", 2000)

        # Seed pool: real human critique text, not Gemma's self-generated
        # output — same "human calibration, not self-distillation"
        # principle as the Critic Tier's own data (see
        # s10_5_critic_preference.py's docstring). At this stage (runs
        # right after s01_5_uicrit_join, before the rest of the chunked
        # pipeline), the only critique_source in the manifest IS
        # uicrit_human — gemma4_31b entries don't exist yet since s10_5
        # runs much later — but the explicit filter documents the
        # intent regardless of run order.
        seed_records = [
            r for r in manifest.get_all_records_with_critique()
            if r.critique_output and r.critique_output.get("critique_source") == "uicrit_human"
        ]
        if not seed_records:
            log.error(
                "planner_synthesis_no_seed_data",
                note="No uicrit_human critique records found — s01_5_uicrit_join "
                     "must succeed first (check its own logs for join failures). "
                     "Planner conversation synthesis cannot proceed without real "
                     "seed content.",
            )
            return result

        tier1 = Tier1Engine(engine, config)
        validator = SchemaValidator(config.schemas_dir)
        out_dir = config.resolved_paths["planner_conversations"]
        out_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        rejected = 0
        shard: list[dict[str, Any]] = []
        shard_size = stage_cfg.get("shard_size", 500)

        # Cycle through seeds if target_count exceeds available seed
        # material — UICrit is ~983 rows, and one seed can plausibly
        # support multiple distinct synthetic conversations (different
        # plausible framings of the same feedback), same way image
        # datasets get multiple augmented views.
        i = 0
        while generated < target_count and seed_records:
            rec = seed_records[i % len(seed_records)]
            i += 1
            critique_text = (rec.critique_output or {}).get("visual_hierarchy_note", "") or \
                             (rec.critique_output or {}).get("raw_fields", {}).get("critique_text", "")
            if not critique_text:
                rejected += 1
                if i > len(seed_records) * 3:  # avoid infinite loop on empty seeds
                    break
                continue

            try:
                convo = await tier1.synthesize_conversation(rec.id, critique_text)
            except Exception as e:
                log.warning("synthesis_call_failed", seed_record_id=rec.id, error=str(e))
                rejected += 1
                continue

            if convo is None:
                rejected += 1
                continue

            convo_dict = convo.model_dump()
            valid, errors = validator.validate_synthesized_conversation(convo_dict)
            if not valid:
                log.warning("synthesis_schema_rejected", seed_record_id=rec.id, errors=errors)
                rejected += 1
                continue

            convo_dict["_synthetic_id"] = str(uuid.uuid4())
            shard.append(convo_dict)
            generated += 1

            if len(shard) >= shard_size:
                self._flush_shard(out_dir, shard)
                shard = []

            if i > target_count * 5:  # generation success rate sanity guard
                log.warning("planner_synthesis_low_yield", generated=generated, attempts=i)
                break

        if shard:
            self._flush_shard(out_dir, shard)

        result.records_processed = generated
        result.records_failed = rejected
        result.metadata = {
            "generated": generated,
            "rejected": rejected,
            "seed_pool_size": len(seed_records),
            "yield_rate": round(generated / max(i, 1), 3),
        }
        log.info("planner_synthesis_complete", **result.metadata)
        return result

    @staticmethod
    def _flush_shard(out_dir, shard: list[dict[str, Any]]) -> None:
        shard_path = out_dir / f"conversations_{uuid.uuid4().hex[:12]}.jsonl"
        shard_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in shard),
            encoding="utf-8",
        )
        log.info("conversation_shard_written", path=str(shard_path), count=len(shard))
