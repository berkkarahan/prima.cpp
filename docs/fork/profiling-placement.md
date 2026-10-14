# Profiling and placement inputs

The fork's profiler produces typed, verified measurements that Kluster's planner
consumes as placement inputs. Profiling never widens placement, never runs in
manifest mode autonomously, and missing evidence preserves capacity-first selection.

## Typed model profiler

`common/profiler.{cpp,h}` and `common/prima-profile-cost.h` compute per-layer typed
costs for the fork's model families:

- FLOPs by tensor type, weight bytes, parameter counts, active bytes and KV bytes
  per layer, plus separate input/output costs, tied-output handling and
  embedding-row sizing.
- Layer shapes encode architecture features the planner must price:
  OAI-style gated activations (`swiglu_oai`), selected-expert softmax, GELU,
  shared/fused expert widths, rotary widths, attention softcap and fused SwiGLU.

## Verified kernel profiling

- Matrix operands are encoded into their actual tensor type; kernel compute status
  and completed outputs are checked against decoded operands, and unsupported GPU
  operations are refused without CPU substitution. Malformed storage never yields
  GFLOPS numbers (e.g. typed Qwen3 kernel profiling repairs).
- Full-model kernel jobs extract complete GGUF metadata without touching live model
  tensors, counting actual attention dimensions, head norms and active MoE experts.

## Native I/O probes

`common/prima-profile-io.h` probes storage with an exclusive owned file on the model
filesystem: explicit cache/alignment settings, unbuffered I/O that refuses
unsupported storage rather than timing the page cache, checked reads, bounded work,
and handle-owned cleanup on child death. Unavailable storage is recorded separately,
never substituted with paged/RAM throughput.

## GPU measurement reporting

- Completed host/device embedding handoffs are measured with byte-checked sample
  arrays at exact model width; device-memory reads/KV copies stay distinct from
  host uploads/downloads.
- Logits downloads are measured at the model's vocabulary width with exact positive
  samples, and missing advertised logits capability is refused before publication.
- Local versus distributed transformer/output graphs are accounted separately:
  GPU output avoids the local intermediate embedding download, while a distributed
  output graph uploads its host embedding again.
- Output projection placement, Vulkan capacity and completed-graph counts are
  reported (`720da84`), including the selected profiler device.

## Opt-in telemetry

Runtime telemetry requires explicit opt-in (`common/prima-env.h`): a set variable
containing "0" disables a flag. Optional Vulkan timing (`diagnostic_vulkan_timing_v1`)
and execution-evidence receipts are gated this way, keeping ordinary timing runs and
ordinary builds free of instrumentation.

## Consumers and limits

Kluster's planner uses these inputs for placement; estimates remain labelled
estimates — application-link proxies, cold logical weight reads and synthetic-matrix
microbenchmarks do not establish generation timing, full-model acceptance or
qualification. Model-process validation and performance gates remain deferred to the
live three-computer cluster.

## Related commits

`e7ee8f6`, `72d4eda`, `cac1288`, `fbed5cb`, `720da84`, `35d9aad`, `9f548d2`,
`5e3e2e1`, `901730e`
