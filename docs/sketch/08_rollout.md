# Rollout and rollback

Introduce a feature flag with three modes:

```text
KRISNA_SKETCH_MODE=maskgit
KRISNA_SKETCH_MODE=layout
KRISNA_SKETCH_MODE=ab
```

`ab` logs both outputs but serves one selected result. Checkpoints and artifacts must be versioned under `models/sketch/` with manifests and schema versions.

Rollback to MaskGIT if:

- IR legality fails;
- locked regions drift;
- Polish quality decreases;
- latency exceeds the current baseline;
- the layout model cannot generalize beyond RICO/mobile layouts.
