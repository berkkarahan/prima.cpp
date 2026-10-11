# Kluster Windows/Vulkan development contracts

This fork is consumed by Kluster LLM through an immutable managed source recipe.
Kluster owns membership, preparation, placement and lifecycle. Fork build success
does not qualify normal Kluster inference selection.

The Windows Vulkan build provides server, CLI, profiler and native contract tests.
Build with `GGML_VULKAN=ON`, `BUILD_SHARED_LIBS=OFF` and pinned ZeroMQ dependencies;
run CTest after building `prima-contract-test` and `prima-server-contract-test`.
The portable CI also compiles CPU builds on Windows, Linux and macOS. Its Windows
Vulkan job establishes compilation only: hosted runners are not physical GPU tests.

`llama-server --list-devices` emits `KLLM_PRIMA_CAPABILITIES`. Vulkan devices include
the physical device UUID, raw driver version and physical enumeration index. The
physical index is the value accepted by `GGML_VK_VISIBLE_DEVICES`; after filtering,
logical names are renumbered from `Vulkan0`. Select by UUID and verify the reported
identity instead of assuming logical zero identifies a particular GPU.

Execution-manifest mode preserves Kluster's source/runtime/rank/placement identity.
Setting `KLLM_PRIMA_EXECUTION_EVIDENCE=1` records at most one completed prefill and
one completed decode graph in `KLLM_PRIMA_GPU_EXECUTION`. Receipts bind the runtime,
manifest and rank to physical device/driver identities and scheduled matrix-weight
ownership, after graph synchronization. Warm-up is excluded. Validate both phases
and every requested GPU layer's attention/FFN matrices; configured ranks and device
enumeration are insufficient evidence. Disable tracing for ordinary timing runs.

The rank-aware Qwen loader/graph now supports Qwen3 and Qwen3MoE projection widths,
Q/K normalization and normalized expert routing. Vulkan dense and MoE execution
has been exercised on an RX 9070 XT in Windows development assessments. This does
not establish every architecture, arbitrary placement or distributed performance.

The server accepts Qwen3's explicit request
`chat_template_kwargs: {"enable_thinking": false}` with its supplied template.
Nonstreaming forced named function calls use the function's JSON parameter schema
and return an OpenAI tool-call envelope. Automatic tool choice, streaming tools,
and full tool-turn/template parity remain separate work; unsupported combinations
fail explicitly. Token-limit responses use `finish_reason: "length"` even when the
internal result contains `stopped_word: false`.

Use Kluster's assessment runner to compare identical model assets/settings against
its pinned llama.cpp baseline. Keep actual GPU execution, semantic parity, quality,
memory, performance and physical distributed results as distinct checks. Only the
source-controlled Kluster qualification policy enables a release scope.

## Unmasked MoE routing regression

The Vulkan softmax shader must never compute modulo zero for an absent mask.
This caused incorrect expert probabilities on RX 9070 XT while CPU execution and
masked attention passed. `test-backend-ops test -b Vulkan0 -o SOFT_MAX` now explicitly
initializes this fork's Vulkan backend instead of silently skipping it through the
newer device registry. Cases cover 128-expert decode/prefill. Also run
`-o MUL_MAT_ID`, `-o GET_ROWS`, `-o DIV` and `-o SUM_ROWS`, then the model assessment
with partial offload; operation tests alone do not qualify model correctness.

## Native protocol backports

The server implements stateless text `/v1/responses`, `/v1/responses/input_tokens`
and `/v1/messages` through its existing completion scheduler. Text streams carry
protocol lifecycle events, termination and usage; Responses events include sequence
numbers. Structured Responses and explicit named nonstream tools share the existing
native grammar path. Invalid protocol requests return HTTP 400. Stored/background
Responses, multimodal input, automatic/streaming tools and tool-turn templates remain
explicitly unsupported; these backports do not assert complete baseline API parity.

Messages truncation preserves `max_tokens` and partial argument text, without
claiming a completed tool action from invalid JSON. Responses preserves incomplete
status for output limits. The pinned llama.cpp baseline omits Responses stream sequence
numbers and reports completed status on an output limit; conformance assessments
record those baseline deviations separately instead of reproducing them.

Kluster's `tests/prima_protocol_assessment.py` checks text/structured/forced tools,
SSE lifecycle/usage/sequence, token counting, output limits and malformed requests
on supplied dense/MoE models. Worktree binaries are developmental observations;
repeat against the fresh published managed build before qualification.

## Small Vulkan decode batches

Matrix products with 1–8 activation columns use batched vector kernels.
Each dispatch maps its column separately from tensor batch dimensions and applies
the existing broadcast ratios to weight selection. Floating-point views use the
contiguous packing copy before vector dispatch; larger batches retain the matrix
path. Packing checks include all active tensor dimensions, including padding between
batches. Specialized attention-vector kernels are restricted to their supported
three-dimensional layouts; fourth-dimensional batches use the general path.
The matrix path selects F16 pipelines after
copying strided weights to F16. Short floating-point vectors keep all lanes in the
reduction barriers and guard the final element; quantized rows retain block decoding.
Strided quantized views have no packing-copy kernel and are reported unsupported
rather than being read with incorrect strides. This does not qualify such placement.

Run `test-backend-ops test -b Vulkan0 -o MUL_MAT` for CPU-reference checks of small
batches, broadcast dimensions, short/odd rows, row/batch padding and attention-style
permutations. Cases unsupported
by a backend are reported separately and do not count as physical execution proof.
Also run `-o MUL_MAT_ID` and supplied-model protocol assessments because the vector
shaders share infrastructure with expert routing. Paired model timing, with execution
tracing disabled, is required to assess performance rather than infer it from dispatch.

Physical distributed cluster gates and validations will be tested later through
Kluster LLM. They remain unqualified; this deferral does not replace implementation.

## Quantized vector iterations and dispatch limits

The generic quantized vector shader processes four values per lane per iteration,
retaining one output row per workgroup. Floating-point vectors keep the two-value
loop and its short/odd-row guards. CPU-reference matrix and expert tests, including
four large sparse final-row cases, pass on RX 9070 XT. A four-output-row workgroup
experiment was withdrawn after paired model timing regressed despite correctness.

Vector dispatch uses rounded X/Z grid dimensions within the device's limits; final
partial groups are protected by uniform row bounds. Pure CPU contracts check coverage
and refusal for bounded/invalid grids without requiring a Vulkan device. The RX
9070 XT reports X limit4294967295: its large final-row cases pass on both the prior
and new backend and do not reproduce the lower-limit grid-splitting defect physically.
Other device/driver qualification remains separate.

Development dense paired timing improves modestly and passes the two short-prompt
throughput workloads, but the complete local performance matrix remains open. Do
not infer release qualification from operation correctness or an individual workload.
