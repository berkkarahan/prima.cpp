# Kluster execution-manifest mode

Manifest mode is the fork's integration contract with Kluster LLM: the orchestrator
owns membership, authenticated preparation, final placement and process lifecycle;
the Prima runtime receives an immutable execution manifest and must preserve the
approved ranks and placement instead of autonomously rescheduling.

## Manifest format and validation

`common/prima-manifest.h` reads and applies execution manifests supplied through
`--execution-manifest`:

- `schema_version` 1, a SHA-256 `runtime_id`, and a non-empty `ranks` array of at
  most 32 entries; malformed identity/schema is refused.
- Adapter (LoRA) assets carry an approved size and SHA-256, verified by streaming the
  staged bytes with a checked, bounded SHA-256 implementation
  (`common/prima-sha256.h`) before use; oversized or drifted assets are rejected.
- `--check-execution-manifest` verifies the manifest against the resolved runtime
  rather than only parsing it. Contract vectors cover the checks, and the portable CI
  builds and runs them on Windows, Linux and macOS.
- Optional metadata additions preserve legacy canonical digests when absent.

## Readiness and lifecycle

- `rank_readiness_v1` adds a separate `KLLM_PRIMA_RANK_READY` acknowledgement. CLI
  workers apply the manifest before selecting rank/world startup behavior and emit
  readiness only after model loading, warm-up, thread pools and control-socket setup;
  the server emits it after request callbacks are installed, immediately before its
  request loop. Rank zero additionally requires HTTP health before serving.
  `KLLM_PRIMA_RANK_CONFIGURED` alone is explicitly not readiness.
- Zero-transformer rank-zero coordinators are supported: they skip writes to unused
  RoPE/mask tensors (whose buffers are not allocated) while still embedding inputs,
  exchanging positions from batch memory and projecting worker outputs. Controlled
  same-host dense-model tests exercise normal and zero-transformer coordinators with
  CPU and Vulkan ranks, including completed assigned GPU work.
- Manifest mode suppresses autonomous profiling, rank pruning and rescheduling;
  placement, including explicit device selection for GPU ranks, must be exactly what
  the manifest approved.

## Cancellation boundaries

CPU graph cancellation was synchronized in `ggml/src/ggml.c`: an atomic
first-not-executed node defines the cancellation boundary, metadata-only graphs and
barriers are settled, and a cancellation already requested never mutates graph
outputs or wakes workers. Contract tests (`tests/prima-cpu-graph.h`) cover the
boundary across one/multiple workers and both OpenMP and pthread implementations.

## Network utilities

`src/network-utils.{cpp,h}` gained the address/port plumbing needed by manifest mode
(derived ports and peer addresses from approved edges) while keeping native rank
traffic on the approved private network; privileged firewall mutation lives in
Kluster's helper, not here.

## Status

Manifest mode is an implementation capability: build success and contract checks do
not qualify physical distributed inference. Multi-machine acceptance, recovery and
performance remain deferred to the live three-computer Kluster cluster.

## Related commits

`ea4a953`, `a4a7e4d`, `f11a4a2`, `6ecb08a`, `a9d1099`, `720da84`
