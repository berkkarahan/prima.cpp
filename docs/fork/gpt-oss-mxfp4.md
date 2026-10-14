# GPT-OSS and MXFP4

The fork adds complete GPT-OSS support — loader, graph operations, tokenizer, KV
storage and profiling — including the MXFP4 quantization format on CPU and Vulkan.
Deep detail lives in [docs/development/mxfp4.md](../development/mxfp4.md),
[docs/development/gptoss-profiler.md](../development/gptoss-profiler.md) and
[docs/development/attention-sinks.md](../development/attention-sinks.md).

## Loader, graph and tokenizer

- The GPT-OSS loader integrates with the fork's absolute layer ownership and
  segmented graph paths, so GPT-OSS models can execute under manifest placement.
- The Harmony tokenizer serves GPT-OSS prompt/output conventions.
- Expert graph operations (routed MoE with sinks) run on CPU and Vulkan.

## MXFP4 storage and execution

- MXFP4 block storage is implemented in `ggml` (`ggml-quants.{c,h}`,
  `ggml-common.h`) with CPU execution paths and a Vulkan dequantization shader
  (`dequant_mxfp4.comp`), so 4-bit experts can run on either backend.
- Quantized paths are integrated with the fork's Vulkan dispatch corrections
  (view offsets, row tails) rather than assuming dense layouts.

## Attention sinks

CPU and Vulkan groundwork for attention sinks supports GPT-OSS's sink-based
attention; sink state is preserved across KV operations (see
[kv-storage-context.md](kv-storage-context.md)).

## Compact KV storage

GPT-OSS KV storage is manifest-bound and compact-capable, participating in the
compact sliding-window machinery described in
[kv-storage-context.md](kv-storage-context.md).

## Typed profiling

GPT-OSS tensors are integrated into the typed model profiler, so planning inputs
(see [profiling-placement.md](profiling-placement.md)) cover its MoE/expert
structure, including the sink-augmented attention shapes.

## Status

GPT-OSS support is implementation capability. Correctness tests, quantization
conformance and physical acceptance remain deferred; MXFP4 execution evidence from
synthetic matrices or single kernels does not establish full-model correctness or
performance.

## Related commits

`af3ff6b`, `1aeaf50`, `04aa51a`, `c31c5fe`, `901730e`, `33921e2`
