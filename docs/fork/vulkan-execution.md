# Vulkan execution and kernels

The fork substantially reworks the Vulkan backend's correctness and dispatch behavior
(`ggml/src/ggml-vulkan.cpp` and its shaders) for the models and serving paths this
fork adds. None of the changes below are performance acceptance results; measured
claims remain deferred to physical cluster validation.

## Correctness foundations

- Unmasked Vulkan softmax was fixed, and explicit GPU backend tests were added
  (`tests/test-backend-ops.cpp`) covering the affected operations directly.
- Matrix precision honors F32 inputs, and attention ranges respect exact batch
  boundaries instead of padded widths.
- Batch strides and vector dispatch were corrected for tensor views: view offsets,
  stride widths and dispatch bounds are checked, and unsupported layouts are refused
  rather than silently mis-read.

## Dispatch and decode batching

- Decode batches optimize vector and strided reads, with quantized vector loops
  (`mul_mat_vec_q*_k.comp`) covering dispatch row tails so quantized rows beyond the
  vector-aligned body are still computed correctly.
- Strided attention queries are batched without input copies, preserving the
  original tensor layout across head-dim strides.

## MoE expert handling

- Device-memory experts are reduced directly with serial-order strided sums
  (`sum_rows_strided.comp`), preserving summation order; tree and serial additions
  can differ for cancellation-sensitive F32 values, which is why order is pinned.
- Small MoE batches use selected-expert Vulkan vectors instead of full expert sweeps.
- Device-memory expert reduction preserves host aggregation semantics; when paired
  full-model measurements showed regression, host-weight paths were restored and
  device-memory optimizations are selected by actual buffer ownership.
- `swiglu_oai.comp` implements the gated OAI-style activation used by newer model
  families.

## Attention sinks

CPU and Vulkan groundwork for attention sinks was added (see
[docs/development/attention-sinks.md](../development/attention-sinks.md)), which
GPT-OSS and other sink-using architectures build on.

## Optional timing diagnostics

Optional Vulkan timing instrumentation (`ggml-vulkan-timing.h`) uses GPU timestamp
queries with queue support checks, completion-time result availability, counter wrap
and unit handling, and bounded query allocation. Diagnostics require explicit opt-in
(runtime telemetry is opt-in overall) and a `diagnostic_vulkan_timing_v1` capability;
`GGML_VULKAN_PERF` builds are diagnostic artifacts and are refused as acceptance
inputs. Instrumentation preserves normal batching and never waits for unused slots.

## Related commits

`e4eb58b`, `146ccb6`, `a024a7e`, `19adfe0`, `622d5c4`, `857833e`, `f13088f`,
`b7d101e`, `fd24c9d`, `704986f`, `3e8ee69`, `af3ff6b`, `9f548d2`, `5e3e2e1`
