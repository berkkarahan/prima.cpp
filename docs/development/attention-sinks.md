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
The GPT-OSS loader and shared segmented graph now consume the sink tensor. The MoE
builder now optionally adds router and selected-expert up/gate/down biases and
applies softmax after selecting expert logits. The CPU/Vulkan `ggml_swiglu_oai`
operation clips gate above 7 and up to [-7,7], then computes
`gate*sigmoid(1.702*gate)*(up+1)`. It accepts separate equal-shaped F32 operands.
Vulkan also copies I32 expert IDs as integers when making the strided top-k view
contiguous for bias gathering. Existing MoE callers retain their prior graph
when the optional features are absent.

The loader maps absolute owned layers to rank-local storage, loads required
expert biases and output tensors, and accepts separate or fused QKV weights
with optional combined/separate attention biases. Scalar/default and per-layer
sliding-window patterns choose each absolute layer's mask and rotary frequency.
K-shift uses the same rotary settings and absolute ownership; defragmentation
also indexes only rank-local owned tensors. Noncausal decoder masking retains
cached sequence isolation while removing the causal future-token cutoff.

The GPT-4o tokenizer regex matches the pinned baseline. Harmony return/call
markers terminate generation, while its intermediate end marker remains usable
between analysis and final channels. These are implementation changes without
model/tokenizer acceptance evidence.

MXFP4, remaining accelerator kernels, full model profiling, compact sliding-window
KV storage and broader model/placement integration remain open. The current
graph retains full-context KV storage and masks older sliding-window keys.
Loader/graph source and compilation do not establish full model support.

Per the implementation-first schedule, unit/native/model checks and test fixes
are deferred until the full required integration is implemented. Later checks
must cover per-head/outer-batch indexing, masked/unmasked inputs, large positive
and negative sinks, finite sinks with fully masked keys, padded Vulkan dispatches,
regular versus Flash Attention, and unchanged behavior without sinks. Compilation
alone does not establish numerical/model/backend acceptance. Later MoE checks
must also cover expert/token bias indexing, padded top-k IDs and their integer
copy, selected-logit softmax, activation clipping/parameters/strides and unchanged
ordinary MoE behavior. Performance and
physical distributed acceptance remain deferred to the live three-computer
cluster; qualification stays disabled.
