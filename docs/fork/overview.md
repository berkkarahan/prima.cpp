# Fork implementation overview

This fork of [OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp) develops
Windows/Vulkan support and integration with [Kluster LLM](https://github.com/berkkarahan/kluster-llm)
while keeping Prima's native distributed execution architecture. Prima.cpp itself is a
distributed llama.cpp for heterogeneous home devices; this fork keeps that architecture
and makes it buildable, orchestratable and validatable through Kluster.

This document maps the fork's implementation areas. Each area has its own summary
document under `docs/fork/`; deep contract-level detail lives in
[docs/kluster-windows-vulkan.md](../kluster-windows-vulkan.md) and
[docs/development/](../development/).

## Scope of the integration work

The `codex/kluster-integration` branch carries the fork's work: 137 commits
(2026-10-10 through 2026-10-13) over the fork base, touching 165 files with
+42,233 / −5,463 lines. The largest concentrations are `common/` (including the
vendored Jinja and chat-runtime engines), the vintage monolithic `src/llama.cpp`
model/graph/execution core, `examples/server/server.cpp`, and the Vulkan backend
`ggml/src/ggml-vulkan.cpp` plus its shaders.

The work is summarized in these documents:

| Document | Area |
|---|---|
| [windows-vulkan-platform.md](windows-vulkan-platform.md) | Windows build robustness, MSVC SIMD probing, explicit Vulkan device identity, driver-unavailable handling, native error containment |
| [vulkan-execution.md](vulkan-execution.md) | Vulkan kernel correctness and optimization: quantized dispatch, batch strides, MoE expert reduction, attention batching |
| [kluster-manifest-mode.md](kluster-manifest-mode.md) | Immutable execution manifests, rank readiness, zero-transformer coordinators, cancellation boundaries |
| [segmented-placement.md](segmented-placement.md) | Absolute layer ownership, ordered ring executor, planned graph visits, output-only GPU coordinators |
| [model-support.md](model-support.md) | Rank-owned/segmented execution for 40+ architectures; embeddings, classifiers and reranking |
| [gpt-oss-mxfp4.md](gpt-oss-mxfp4.md) | GPT-OSS loader/graph, Harmony tokenizer, MXFP4 storage, attention sinks |
| [kv-storage-context.md](kv-storage-context.md) | Compact sliding-window KV storage, KV rewind/replay, prompt caching, per-slot context checkpoints |
| [chat-runtime.md](chat-runtime.md) | Vendored Jinja engine, baseline chat parser, reasoning budgets, template keyword defaults |
| [serving-apis.md](serving-apis.md) | Chat/Responses/Messages protocols, tool calling, logprobs, completions, FIM infill, rerank endpoints |
| [lora-draft-decode.md](lora-draft-decode.md) | Managed LoRA loading/selection, per-request adapters, isolated draft-simple decoding |
| [profiling-placement.md](profiling-placement.md) | Typed model/kernel profiler, verified operands, native I/O probes, GPU measurement reporting |

## Vendored baseline engines

Three engines are vendored from ggml-org/llama.cpp at the immutable Kluster baseline
commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, with provenance and MIT license
files retained verbatim:

- `common/jinja/` — the Jinja template engine (upstream PR #18462 lineage), adapted
  only to use Prima's existing `json.hpp`.
- `common/chat-runtime/` — the baseline chat/template analysis, PEG tool parser,
  JSON-schema-to-grammar and model-specific parsers, adapted to Prima's vintage
  model/token API.
- `common/reasoning-budget.{h,cpp}` — the baseline reasoning budget state machine.

The baseline pin matters: serving behavior is compared against it, and its semantics
(parsers, template rendering, logprob handling) are intentionally preserved.

## Integration boundaries

Kluster LLM owns cluster membership, authenticated preparation, final placement and
process lifecycle. Prima owns native model/backend execution, kernel profiling and its
server behavior. In manifest mode the runtime preserves the approved ranks and layer
placement instead of autonomously rescheduling; profiling supplies planning inputs and
never widens placement. Native rank traffic stays on the approved private network.

## Status and deferred acceptance

This fork is under active integration. Earlier local Windows/Vulkan runs (Qwen3 dense
and MoE on an AMD Radeon RX 9070 XT) are historical and do not qualify the current
source. Production compilation and portable CI contract checks run, but unit, native
and model tests — and test-fix work — are deferred until the full required
implementation is finished. Physical distributed cluster gates and performance
acceptance are deferred to a live three-computer Kluster cluster and remain unpassed.
Linux/macOS Vulkan hardware acceptance is also open. This fork does not claim the
original project's published speedups for its new configurations.

Capability reports from this fork (for example `segmented_placement`,
`managed_lora_v1`, `request_lora_v1`, `prompt_cache_v1`, `context_checkpoints_v1`,
`template_kwargs_defaults_v1`, `apply_template_v1`) describe implemented mechanics,
not qualification.

## Consuming the fork

Kluster consumes this repository through an immutable managed source recipe pinned to
an exact commit; the recipe's repository/commit and dependency pins are authoritative.
Standalone builds follow the CMake commands in the top-level
[README](../../README.md#build-on-windows-with-vulkan) and the
[CI workflow](../../.github/workflows/kluster-build.yml).
