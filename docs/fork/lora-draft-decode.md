# LoRA adapters and draft decoding

The fork implements managed LoRA execution across ranks and isolates draft-simple
speculative decoding, both under manifest control.

## Managed LoRA

LoRA loading and application are implemented rank-wide (primarily in the vintage
monolithic `src/llama.cpp`, `include/llama.h` and `examples/server/server.cpp`, with
asset verification in `common/prima-manifest.h`):

- **Rank-owned loading** (`aeba7fc`): adapter A/B tensors are loaded under the
  model's absolute layer ownership with exception-safe cleanup; metadata, device
  contexts, buffers and model registration are contained so initializer failures
  don't leak. Unknown adapter targets fail the owning rank before the readiness
  barrier. Adapter metadata validation covers finite alpha, positive rank and
  matching dense/expert shapes.
- **Ordered selection synchronization** (`cf1890a`): managed heads bind replacement
  commands to the exact manifest, a monotonic revision and physical ring hop order,
  then wait for all-rank acknowledgement before committing; worker-local managed
  changes are refused. Backend work settles before/after KV and snapshot
  invalidation; startup status reports selection.
- **Binding verification** (`9f7abdc`): LoRA targets and scales are verified against
  model tensors, and completed adapter matrix work is recorded rather than inferred.
- **Embedding and tied-output LoRA** (`f8aa373`): adapters apply to gathered token
  embeddings and, for tied output, to the actual output buffer (decode/transpose of
  B retained alongside A), with reserved graphs.
- **Per-request LoRA batches** (`9808a10`): deployment defaults stay separate from
  slot selections; equal selections are grouped with bounded round-robin scheduling,
  batch weights settle on all ranks through a manifest/revision/mode-bound
  acknowledged command, and caches/checkpoints bind to exact scales.
- **Activated LoRA prompt phases** (`19cfbf1`): bounded invocation metadata with
  neutral cache ownership, so per-call adapter activation parses against the base
  vocabulary without corrupting sequence state.

Adapter assets enter through the execution manifest with approved basenames, scales,
sizes and SHA-256 digests; changes rebuild system prefixes and clear affected
slot/draft state.

**Enablement boundary:** this is the loading/synchronization foundation. Managed
LoRA remains gated (`managed_lora_v1` development capability, not qualified);
adapter execution, distributed selection parity and performance evidence are part of
the deferred test phase.

## Draft-simple speculative decoding

Draft decoding is isolated (`common/prima-draft.h`, `common/speculative.cpp`,
`a031404`):

- Only `draft-simple` speculative decoding is supported by this runtime; the
  validated limits require token counts in 0..1024 and probability in 0..1.
- The draft context is a clean single-rank copy of the target parameters: layer
  ownership, LoRA adapters, segmentation, prompt-cache ceiling, context checkpoints
  and the execution manifest are all cleared, so a draft never inherits the target's
  distributed state.
- Draft files resolve through the existing model resolver with verified basenames;
  draft KV types are selectable (`--spec-draft-type-k/v`), explicit CPU selection is
  honored, and mlock is never forced.
- Verified token API semantics are preserved on the main path while the draft runs
  isolated (`--model-draft`, `--draft-n`, `--draft-n-min`, `--draft-p-min`,
  `--n-gpu-layers-draft`).

## Related commits

`aeba7fc`, `cf1890a`, `9f7abdc`, `f8aa373`, `9808a10`, `19cfbf1`, `a031404`
