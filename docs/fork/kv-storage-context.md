# KV storage, prompt caching and context checkpoints

The fork reworks how key/value state is stored, replayed, cached and checkpointed,
both for compact sliding-window architectures and for Kluster's managed serving.

## Compact sliding-window (SWA) storage

- `common/prima-kv-layout.h` plans per-layer storage: base vs sliding layers,
  unified sequence pools sized for each sequence's window plus the complete incoming
  microbatch, and SWA padding to 256. Storage planning is independent of the pools'
  mutable sequence/cell state; full-size allocation is preserved until compact
  consumers are wired.
- `common/prima-swa-map.h` maps compact tensor rows to the full cache's
  authoritative sequence cells: reservations are generation-tracked, committing only
  after the full cache successfully reserves the corresponding global IDs, so
  preparing a slot never reclaims rows or mutates either cache (see
  [docs/development/compact-swa-map.md](../development/compact-swa-map.md)).
- Transactional compact SWA row metadata, compact SWA attention wiring and sparse
  cache snapshots follow; grouped KV transforms settle across server and CLI paths.

## KV rewind and replay

- Server context is replayed from completed token history, including
  self-extend: origin-aware KV programs rewind and rebuild state instead of trusting
  optimistic `n_past` (which can include undecoded sub-batches).
- CLI histories replay with checked rank transforms; grouped KV transforms cover the
  CLI self-extend path.
- Actual per-layer KV storage is accounted in memory estimates, including
  manifest-bound compact GPT-OSS storage.

## Prompt caching (Kluster)

Bounded prompt caching (`--cache-ram`) reserves the entire configured host RAM
ceiling on every selected participant, including GPU workers and a zero-layer
coordinator:

- Requires `prompt_cache_v1` and signed settings before preparation; serialized KV
  state stays on its owning rank, with bounded all-rank transactions and one eviction
  order for save/restore/remove.
- Partial saves are removed everywhere; partially restored sequences are invalidated
  and their prompt recomputed. Snapshots clear when system prompts or adapters change.

## Per-slot context checkpoints

Bounded per-slot context checkpoints (`--checkpoint-cache-bytes`,
`--checkpoint-min-step`, `--swa-checkpoints`) keep independent histories per HTTP
slot and a separate pool/ceiling from prompt-cache RAM:

- Actual completed prefill positions are captured, including on a zero-layer head.
- Recurrent snapshots require an entire matching prefix plus a remaining token to
  rebuild logits; sampling and draft prompt state are rebuilt after restoration.
- Count, token spacing and the planned byte ceiling are frozen in the execution
  contract (`context_checkpoints_v1`); every selected rank is priced for the full
  sequence state without charging the master twice.

## Status

Cache operation evidence establishes no throughput benefit. Cache/checkpoint
contracts, model-process branches and failure/cancellation checks are deferred with
the rest of the test phase; performance gates stay with the live cluster.

## Related commits

`b45ae7e`, `4ccaa99`, `3c64365`, `a99c14d`, `c96308f`, `bbc5755`, `023c788`,
`bcd90dd`, `48c6a57`, `33921e2`, `ab96f06`, `a5ef318`
