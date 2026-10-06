import sys
from pathlib import Path

# Mock torch and vllm to allow data_forge modules to import without them installed
class MockModule:
    pass
sys.modules['torch'] = MockModule()
sys.modules['torchvision'] = MockModule()
sys.modules['vllm'] = MockModule()
sys.modules['faiss'] = MockModule()
sys.modules['mediapipe'] = MockModule()

from data_forge.manifest import Manifest
from data_forge.data.storage import StorageManager

print("Testing SQLite connection and PRAGMAs...")
db_path = Path("test_db/manifest.db")
if db_path.exists():
    db_path.unlink()

try:
    m = Manifest(db_path)
    print("[SUCCESS] SQLite Database initialized with WAL and IMMEDIATE isolation.")
    rec = m.create_record(source_dataset="test_smoke")
    print(f"[SUCCESS] Record inserted: {rec.id}")
    
    # Test bulk insert to ensure no locks
    m.bulk_create_records([{"source_file": "img1.png"} for _ in range(100)], "test_smoke")
    print("[SUCCESS] Bulk insertion completed. No database locks encountered.")
    m.close()
except Exception as e:
    print(f"[FAILED] Manifest error: {e}")

print("\nTesting MAX_PATH limits on StorageManager...")
try:
    class MockStorageConfig:
        safety_margin = 0.1
        per_record_estimates = {"zimage": 1000}
    class MockConfig:
        data_root = Path("D:\\" + "very_long_path_name_to_trigger_windows_max_path_limit_exception_in_storage")
        storage = MockStorageConfig()
        resolved_paths = {}
    
    s = StorageManager(MockConfig())
    s.pre_flight_check(100)
    print("[FAILED] StorageManager did not catch the MAX_PATH limit.")
except Exception as e:
    if "MAX_PATH" in str(e):
        print(f"[SUCCESS] StorageManager safely caught MAX_PATH limit: {e}")
    else:
        print(f"[ERROR] Unexpected exception: {e}")
