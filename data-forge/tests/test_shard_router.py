"""Tests for the shard router."""

from data_forge.data.shard_router import ShardRouter
from data_forge.manifest import ManifestRecord


def _make_record(id: str, domain: str) -> ManifestRecord:
    return ManifestRecord(id=id, source_dataset="test", domain=domain, status="structured")


class TestShardRouter:
    def test_ratio_enforcement(self):
        records = [_make_record(f"ui_{i}", "ui_first") for i in range(70)]
        records += [_make_record(f"gen_{i}", "general_design") for i in range(30)]

        router = ShardRouter(ui_first_ratio=0.70, overflow_action="exclude")
        assignments = router.route(records)

        routed = [a for a in assignments if a["status"] == "routed"]
        overflow = [a for a in assignments if a["status"] == "overflow_excluded"]

        assert len(routed) == 100
        assert len(overflow) == 0

    def test_overflow_excluded(self):
        # 90 UI, 10 general, but ratio is 70/30
        records = [_make_record(f"ui_{i}", "ui_first") for i in range(90)]
        records += [_make_record(f"gen_{i}", "general_design") for i in range(10)]

        router = ShardRouter(ui_first_ratio=0.70, overflow_action="exclude")
        assignments = router.route(records)

        routed = [a for a in assignments if a["status"] == "routed"]
        overflow = [a for a in assignments if a["status"] == "overflow_excluded"]

        # 70 UI + 10 general should be selected (30 target gen, only 10 available)
        assert len(overflow) == 20  # 20 excess UI records

    def test_shard_assignment(self):
        records = [_make_record(f"r_{i}", "ui_first") for i in range(15)]
        router = ShardRouter(ui_first_ratio=1.0, records_per_shard=5)
        assignments = router.route(records)

        shard_ids = {a["shard_id"] for a in assignments if a["shard_id"]}
        assert len(shard_ids) == 3  # 15 / 5 = 3 shards
