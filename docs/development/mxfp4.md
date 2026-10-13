# MXFP4 implementation

GGUF/GGML type 39 is MXFP4. Each 32-value block occupies 17 bytes: one E8M0
exponent followed by 16 packed bytes. Low nibbles represent the first 16 values
and high nibbles the next 16. The doubled FP4 table is
`0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12`; it is multiplied by half the shared
E8M0 scale. Exponents 0 and 1 use explicit F32 subnormal bit patterns. The
reserved exponent 255 is rejected by tensor row validation.

Storage, reference quantization/dequantization, row validation, Q8_0 CPU dot
products and the generic CPU quantized dispatch paths are implemented. The
reference functions and scale conversion are adapted from the MIT-licensed
[pinned llama.cpp source](https://github.com/ggml-org/llama.cpp/tree/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src).
The native file-type identity is `LLAMA_FTYPE_MOSTLY_MXFP4_MOE` (38); its policy
uses MXFP4 for routed expert matrices and Q8_0 for other quantized matrices,
preserving explicit input/output type overrides.

Vulkan implements direct compressed matrix and matrix-vector operations,
including indexed expert dispatch and F32 accumulation selection, plus row
gathering and device dequantization. Shader generation and pipeline registration
include the new format. Its packed 17-byte block layout is preserved on device;
the implementation does not substitute IQ4_NL scales or convert the model asset.
CUDA, Metal, SYCL, CANN and Kompute MXFP4 kernels remain open; their operation
admission refuses this format. That does not qualify a requested accelerator or
establish full platform/backend parity.

Kluster's GGUF layout reader counts compressed tensor spans directly from file
offsets, including alignment. It needs no fixed-per-value storage approximation
for this format. The [typed GPT-OSS model profiler](gptoss-profiler.md) now maps
its roles and typed rate path; no measurements or acceptance checks have run.
Compact sliding-window KV storage remains implementation work, alongside broader
model/placement parity.

Production compilation is the current verification boundary. Per the user's
schedule, unit/native/model checks and test fixes start after full required
integration implementation finishes. Later checks must cover every FP4 code,
signs/zero, exponent boundaries and rejection, 17-byte block/row/expert strides,
CPU dot-input quantization, GPU matrix/vector/indexed-expert/gather dispatch,
F32 accumulation, noncontiguous views and unchanged neighboring formats. Full
model checks must use the original complete MXFP4 GGUF and native prompt/token
evidence. Performance and physical cluster acceptance remain deferred to the
live three-computer cluster and unpassed. Qualification stays disabled.
