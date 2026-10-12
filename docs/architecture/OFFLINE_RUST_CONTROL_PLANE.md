# Offline Rust control-plane design

The application is entirely local: model weights, session state, blobs,
SQLite databases, and telemetry remain on the device. Rust is introduced as a
fast deterministic control plane inside the Tauri process, not as a second
copy of the ML runtime.

## Responsibility split

| Layer | Runtime | Responsibility |
|---|---|---|
| UI/control plane | Rust/Tauri | bounded turn policy, local commands, cancellation, snapshots |
| model plane | Python | llama.cpp Planner, Layout projection, Diffusers Qwen pipeline |
| persistence | local SQLite/files | durable DesignState, blobs, manifests, events |

The Rust layer must reject invalid turn transitions before an IPC/model call.
Python remains authoritative for model execution and durable DesignState until
the SQLite ownership migration is complete. The two state machines must be
reconciled by revision/cycle identifiers; neither side may silently overwrite
the other.

## Implemented slice

`src-tauri/src/pipeline.rs` provides an allocation-light in-process policy:

- four-turn maximum;
- user/self-review source tracking;
- revision-cycle reset;
- serialized snapshots through Tauri commands;
- revision-aware command envelopes and acknowledgements;
- unit tests for budget exhaustion and cycle reset.

This reduces UI latency for policy checks and keeps invalid requests out of
the Python/model path. It is not yet the durable source of truth.

## Migration rules

1. Keep Python DesignState authoritative during the transition.
2. Attach `command_id`, `cycle`, `turn_count`, and `expected_revision` to every Rust↔Python command; mutations return the new revision and snapshot.
3. Reject stale acknowledgements rather than merging guessed state.
4. Move event append and SQLite transactions together only after parity tests.
5. Never duplicate model weights or image tensors in Rust; exchange blob IDs
   and metadata, not pixel payloads.
6. Keep all paths local and validate that model/blob references stay inside
   configured application data roots.

## Performance targets

- policy command p95 < 1 ms;
- no blocking model call on the Tauri UI thread;
- image transfer by local blob reference, not base64 JSON;
- one Python model process per model runtime;
- measured cold-start, warm-start, peak RSS/VRAM, and IPC latency.

## Next Rust slices

1. Move timer/cancellation ownership into Rust while persisting events in
   Python SQLite until parity is proven.
2. Add local event batching and crash-safe append/export.
3. Introduce a narrowly scoped local IPC protocol for Planner/Layout/Qwen
   commands.
4. Only then evaluate moving durable session storage to Rust SQLite.
