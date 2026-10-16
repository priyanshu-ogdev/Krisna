import sqlite3
import pathlib

for p in pathlib.Path('.').rglob('*preference*.db'):
    print(p, p.stat().st_size)
    try:
        conn = sqlite3.connect(str(p))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        print("Tables:", tables)
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {count} rows")
        conn.close()
    except Exception as e:
        print("Error:", e)
