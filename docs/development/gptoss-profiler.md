# GPT-OSS typed profiler

The typed full-model profiler accepts GPT-OSS's separate or fused QKV weights,
post-attention RMS norm, attention sinks and router/expert biases. Dimensions
come from the loaded GGUF metadata, including per-layer expert widths. It refuses
missing required roles, invalid shapes and ambiguous fused/separate weights.
Ordinary bias roles are explicit rather than treating every unknown bias name
as supported. Fused attention includes Q/K rotary work, QK softmax, V aggregation
and the same KV traffic as separate projections, without counting them twice.

All expert tensors contribute resident weights and parameter counts. Per-token
matrix and bias arithmetic and active expert bytes use the selected expert count.
GPT-OSS's clipped/shifted activation, selected-logit softmax, router sorting proxy,
expert weighting/reduction and residual additions contribute F32 work estimates.
Non-matrix costs retain the existing roofline model's logical-work conventions;
they are not measured instruction counts or operation-specific timings.

The graph currently uses full-context KV views and masks old sliding-window keys.
The profiler therefore retains full KV attention work/traffic instead of claiming
the compact cache or skipped computation that is still implementation work.
The existing matrix-rate path encodes operands through their real GGML type and
can measure MXFP4 now that its CPU/Vulkan kernels exist. Missing requested GPU
rates refuse publication. GPT-OSS GPU profiling also refuses backends without its
sink/activation kernels, even when ordinary matrix kernels would succeed.

The typed JSON schema remains version 2. Kluster's decoder requires positive
rates for every declared model tensor type and recomputes each estimate from
those rates and byte costs. Cache identity includes the published source/recipe,
model and settings/hardware, so older model cost receipts are not reused.

Only production compilation is checked in this implementation phase. No profiling
measurements, unit/native/model checks, test fixes or performance checks run until
the user's scheduled phase. After full required implementation finishes, check
required/malformed roles and dimensions, separate/fused equivalence, expert bias
selection, activation/router estimates, original MXFP4 operand encoding and
completed CPU/requested-GPU work, full model cost/rate vectors and signed cache
identity. These estimates never establish generation quality or hardware
performance acceptance. Physical cluster/performance gates remain deferred to
the live three-computer cluster and unpassed; qualification stays disabled.
