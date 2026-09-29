# scripts/data-forge/

Empty by design. data-forge's automation is its own installed CLI, not
shell scripts:

```bash
data-forge run                              # full pipeline
data-forge run --stages 0,1,2                # specific stages
data-forge registry check                    # dataset-release watcher
data-forge manifest stats                     # manifest introspection
```

See `../../data-forge/README.md` for the full command reference.
