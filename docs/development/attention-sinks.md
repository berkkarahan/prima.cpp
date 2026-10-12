# Attention sinks

The fork provides `ggml_soft_max_add_sinks` and
`ggml_flash_attn_ext_add_sinks` as groundwork for architectures with learned
attention sinks. Each accepts a contiguous F32 vector containing one unscaled
sink logit per query head. The sink contributes to the softmax denominator but
has no value vector, so the returned attention weights need not sum to one.
Passing NULL removes the sink dependency.

For a head with masked/scaled logits x and sink logit s, the output weights are
`exp(x - m) / (sum(exp(x - m)) + exp(s - m))`, with
`m = max(max(x), s)`. The Flash Attention CPU kernel applies the same correction
to its accumulated value vector and online normalizer. Batched head indexing
restarts for each outer batch.

Regular softmax supports sinks on CPU and Vulkan, including F16 or F32 masks
and unmasked input. Vulkan uses a separate sink buffer and bounds padded row
dispatches before reading operands. CPU Flash Attention also supports sinks.
CUDA, Metal, SYCL, CANN and Kompute do not yet implement sink-aware softmax;
their admission paths refuse this operation instead of ignoring its extra
operand. CUDA and Metal also refuse sink-aware Flash Attention. Backend refusal
is not evidence of execution on the requested accelerator.

The shared KV/attention graph builders now accept an optional sink tensor.
This does not add the GPT-OSS model loader or claim full GPT-OSS support. Expert
biases and selected-logit routing, its activation, alternating sliding-window
attention, per-layer RoPE, MXFP4 and remaining accelerator kernels still need
implementation and integration with exact placement/profiling.

Per the implementation-first schedule, unit/native/model checks and test fixes
are deferred until the full required integration is implemented. Later checks
must cover per-head/outer-batch indexing, masked/unmasked inputs, large positive
and negative sinks, finite sinks with fully masked keys, padded Vulkan dispatches,
regular versus Flash Attention, and unchanged behavior without sinks. Compilation
alone does not establish numerical/model/backend acceptance. Performance and
physical distributed acceptance remain deferred to the live three-computer
cluster; qualification stays disabled.
