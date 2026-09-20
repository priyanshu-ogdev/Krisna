# scripts/data-forge/

`pin_revisions.py` — resolves every unpinned `revision: "main"` in
`data-forge/configs/{datasets,models}.yaml` to a real commit SHA via the
HuggingFace API. Comment-preserving (`ruamel.yaml`, verified via a
byte-for-byte round-trip test — see `../../tests/data_forge/
test_pin_revisions.py`), dry-run by default.

```bash
python scripts/data-forge/pin_revisions.py           # dry run
python scripts/data-forge/pin_revisions.py --apply    # writes real SHAs
```

`verify_schemas.py` — pre-flight structural check of every hand-authored
JSON Schema in `data-forge/configs/schemas/` against the corresponding
Pydantic model's generated schema (required fields + property
types/enums, not byte-for-byte, since the static files are deliberately
simplified). Run it before shipping a `structured_output.py` change:

```bash
python scripts/data-forge/verify_schemas.py
```

Everything else about data-forge's automation is its own installed CLI,
not shell scripts:

```bash
data-forge run                              # full pipeline
data-forge run --stages 0,1,2                # specific stages
data-forge registry check                    # dataset-release watcher
data-forge manifest stats                     # manifest introspection
```

See `../../data-forge/README.md` for the full command reference.
