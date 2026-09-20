"""Data utilities package: fetching, storage management, deduplication, domain tagging, and routing."""

from __future__ import annotations

from data_forge.data.dedup import DedupEngine
from data_forge.data.domain_tagger import tag_domain
from data_forge.data.fetcher import DatasetFetcher
from data_forge.data.schema_validator import SchemaValidator
from data_forge.data.shard_router import ShardRouter
from data_forge.data.storage import StorageManager, StorageQuotaExceeded

__all__ = [
    "DedupEngine",
    "tag_domain",
    "DatasetFetcher",
    "SchemaValidator",
    "ShardRouter",
    "StorageManager",
    "StorageQuotaExceeded",
]
