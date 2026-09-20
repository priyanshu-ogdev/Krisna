#!/usr/bin/env bash
set -euo pipefail

DATA_FORGE_DATA_ROOT="${1:?Usage: sync_from_data_forge_dpo.sh <data_forge_data_root> [preference_pairs.db]}"
PREF_DB="${2:-krisna_preference_pairs.db}"

echo -e "\033[1;36mSyncing preference pairs from data-forge ($DATA_FORGE_DATA_ROOT) -> $PREF_DB...\033[0m"
source .venv/bin/activate

python3 - "$DATA_FORGE_DATA_ROOT" "$PREF_DB" <<'PYEOF'
import sys
from krisna_training.dpo.preference_store import PreferenceStore
from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_training.data_forge_bridge.sync_dpo_pairs import sync

data_root, pref_db = sys.argv[1:3]
store = PreferenceStore(db_path=pref_db)
try:
    counts = sync(data_root, store, get_blob_store())
    print("Imported:", counts)
finally:
    store.close()
PYEOF

echo -e "\033[1;32mDone.\033[0m"
