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

`rank_readiness_v1` adds a separate `KLLM_PRIMA_RANK_READY` acknowledgement. CLI
workers apply the manifest before selecting rank/world startup behavior and emit
readiness only after model loading, warm-up, thread pools and control-socket setup.
The server emits it after request callbacks are installed, immediately before its
request loop. Reports bind runtime/rank/manifest, allocated transformer layers and
GPU-layer count, with `worker_decode_loop` or `server_request_loop` stage. A
`KLLM_PRIMA_RANK_CONFIGURED` report alone is not readiness; rank zero still requires
HTTP health before serving.

Zero-transformer rank-zero coordinators skip writes to unused RoPE/mask tensors,
whose buffers are not allocated. They still embed inputs, exchange positions from
batch memory and project worker outputs. Controlled same-host dense-model tests
exercise normal and zero-transformer coordinators with CPU and Vulkan ranks,
including completed assigned GPU work. They do not qualify a physical multi-machine
cluster. Distributed cluster gates and validations will be tested later through
Kluster's separately gated production coordination/admission path.

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

## OpenAI token logprobs

Chat responses expose selected-token and top-alternative scores in
`choices[].logprobs.content`, including streaming chunks. `logprobs` accepts a boolean;
`top_logprobs` requires it to be true and accepts integers0–20. Zero alternatives
still reports the selected token. Probabilities are normalized over the full raw
pre-sampling vocabulary, so grammar-forced tokens outside the alternatives keep
their actual model score. UTF-8 byte fragments retain their raw bytes and use valid
replacement text for JSON; end-of-generation token pieces may be empty.

Probability reporting leaves the sampling candidate set unchanged. The legacy
collector now uses each request's probability count and bounds accesses by the
available candidates. Streaming delivers each token entry once, including metadata
for framing tokens that are removed from visible content, and starts with the
assistant role. Usage continues to count actual generated tokens.

Native contracts cover probability normalization, selected tokens outside the top
set, zero alternatives, byte fragments, validation and streaming envelopes. Dense/
MoE model-backed checks and isolated Kluster gateway suites pass the new cases.
The pinned baseline has separate deviations for zero alternatives, stream metadata
and invalid counts; these are recorded rather than duplicated. Whole API/template/
model compatibility and performance qualification remain open.

## Vulkan matrix precision

F32 matrix weights and explicit `GGML_PREC_F32` requests use F32 shared tiles and
accumulation instead of narrowing operands through F16. Floating-point views retain
that representation when copied to contiguous buffers. F16 activation inputs can
be widened for the requested matrix path without changing their stored values.
Quantized matrices retain their weight format and use F32 activation tiles when
requested. Tile geometry fits the device's actual shared-memory budget, including
16 KiB/32 KiB limits checked by CPU contracts. Attention KQ multiplication requests F32 across supported architectures,
matching the baseline behavior for Qwen3 and MoE as well as existing models.

Physical RX 9070 XT regressions reproduce 15 infinity mismatches before the change
on finite inputs above the F16 range. All 39 finite-range cases pass after extending
coverage to larger tile geometries, as part of 237 supported matrix GPU/CPU comparisons;
185 expert comparisons and both native contracts also pass. The generic runner summary includes filtered
and unsupported cases. F16 single-column CPU conversion overflows for the large
activation fixture, so that invalid reference is excluded instead of called a GPU
pass. These checks do not resolve the separate CPU concurrency/token variation
or establish complete model, platform or performance qualification.

## Batched strided attention

The F16/F32 attention vector paths execute one to eight query columns directly from
their tensor strides. Each workgroup shares K/V reads between two queries; odd final
columns do not read a nonexistent neighbor. Grouped-query head mapping, descriptor
offsets and output columns remain explicit. Rows use the X dispatch dimension, and
layouts or dimensions beyond the specialized kernel's contract retain the general
path. Four-dimensional batches are not flattened into this three-dimensional path.

`test-backend-ops -f attention=true` selects focused correctness/performance cases.
Physical checks pass 36 attention cases, including odd column counts, padded strides
and offsets, plus 273 supported matrix and 185 expert comparisons. Both native
contracts and dense/MoE protocol subsets pass. The RX 9070 XT has four-byte storage
buffer alignment, so the old F32 offset omission is not numerically reproduced here;
larger-alignment device qualification remains separate.

Two-column operation times improve from approximately 19.7 to 15.3 microseconds for
values and 58.3 to 40.3 for keys. Development dense five-pair model timing passes all
four local workloads with unchanged bytes/counts. This is not a published-build,
MoE, other-device or distributed performance qualification.

## Verified profiler matrices

Kernel profiling encodes float operands into their actual F16/quantized storage
before upload. Completed matrix output is checked against a reference computed from
the decoded operands; valid lower-precision representations are not compared to
unquantized constants. Allocation, unsupported operations, failed compute and zero
elapsed time refuse measurement. The selected backend executes directly without a
CPU fallback scheduler or leaked scheduler/backend ownership.

`prima-profiler-contract-test` checks all twelve existing CPU profiling formats.
Its explicit `--vulkan` mode checks nine supported formats on a real selected GPU
and requires refusal for IQ2_XXS/IQ1_S/IQ1_M, whose Vulkan matrix kernels are absent.
CPU contracts run in portable build CI; the GPU mode requires physical hardware
and does not run on hosted Windows compilation workers. RX 9070 XT development
checks pass those nine numerical comparisons and the three unsupported refusals.
These kernel timings are estimates, not end-to-end model or distributed benchmarks.

The matrix repair alone did not fix full-model profiling. Follow-up now extracts
costs from the complete GGUF tensor metadata without replacing live model layers.
Qwen3 Q/K norms, actual Q/K/V/output dimensions and selected MoE experts are counted;
storage accounts for every expert. Missing roles, incompatible dimensions and
unknown tensor layouts refuse estimates rather than returning partial costs.

Structured `KLLM_PRIMA_PROFILE` output is schema2 (`typed_kernel_profile_v2`). It
records per-type CPU/selected-backend kernel rates, absolute per-layer storage and
roofline estimates, input/output costs, and serial unbuffered disk measurements.
`KLLM_PRIMA_PROFILE_BACKEND` chooses cpu/vulkan/metal/cuda explicitly. Vulkan's read
probe executes and checks a large matrix-vector graph; KV copying uses valid F16
operands and verifies completed bytes. Neither operation uses CPU fallback. Estimates
are not end-to-end decode/prefill measurements or release performance qualification.

Profiling no longer shells out to fio. Its private exclusive64MiB file is created
beside the model on the same filesystem, uses sector-aligned native unbuffered I/O,
checks every returned read byte and is removed by handle lifetime. Windows uses
NO_BUFFERING/WRITE_THROUGH/DELETE_ON_CLOSE and the reported storage alignment; Linux
uses O_DIRECT/O_SYNC, and macOS uses F_NOCACHE plus flush. Unsupported storage or an
unwritable model directory refuses the measurement. The probe is queue-depth1 with
1MiB sequential blocks and a reported aligned random block. Hardware caches remain
part of the observed device; this is not a guarantee of uncached physical media.
See [Windows buffering requirements](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
and [storage alignment fields](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_storage_info).

Native contracts cover Qwen3 dimensions/norms, GQA cache costs and active-expert
arithmetic versus full storage. Explicit --io checks real I/O and invalid alignment;
--hold-io-file supports owned-child death cleanup checks. Development RX9070XT runs
produce complete dense28-layer/Q8 and MoE48-layer/mixed-Q4K/Q6K reports, and Kluster's
production ProfileJobs consumer validates/caches both with no retained process.
Portable compilation, final immutable-build consumption and wider planner/physical
acceptance remain independently required. Unsupported IQ Vulkan kernels, broader
architectures/layouts and inference performance qualification remain open.


### Completed host/device embedding handoffs

`host_transfer_profile_v1` extends the schema2 typed report with `gpu_host_transfer`.
The `completed_f32_embedding_v1` method allocates an F32 tensor at the actual model
embedding width on the requested selected GPU backend, warms once, then measures
five completed uploads and downloads separately using a steady high-resolution
clock. Every sample changes its input and compares all returned bytes. The report
retains exact bytes and both sample arrays; CPU reports null, and absent/unsupported
GPU backends refuse instead of substituting CPU. This is distinct from device-memory
reads and GPU-to-GPU KV copies. Consumers must use only matching model payload sizes;
these synthetic handoffs do not qualify distributed generation or performance.

Windows/RX9070XT contracts verify4/4096/8192/16388-byte handoffs, invalid widths and
CPU refusal. Actual managed development Qwen3 dense/MoE kernel jobs both complete
28/48-layer reports with checked host-transfer samples. Portable native contracts
pass3/3. Physical multi-node/platform acceptance remains separate and will be tested
later through Kluster's integration qualification workflow.


### Completed logits downloads

`logits_transfer_profile_v1` extends schema2 with `gpu_logits_transfer` using method
`completed_f32_logits_v1`. It initializes/checks an F32 GPU buffer at the actual
`llama_n_vocab` width and reports five completed downloads after warmup. Exact bytes
and positive sample arrays are retained; these downloads are separate from embedding
handoffs. CPU reports null, and unavailable requested GPU backends refuse substitution.
Kluster must match the primary GGUF embedding tensor's vocabulary dimension before
using the median output download cost. A missing advertised report cannot be cached.

Physical Windows/RX9070XT contracts cover odd131073 and actual151936-element output
widths (524292/607744bytes), alongside prior embedding widths. Dense/MoE development
managed jobs both produce607744-byte checked logits samples. Native contracts3/3 and
Kluster consumer/scorer/capability checks pass. Local connected graphs keep GPU output
on device until logits download; distributed rank zero uses separate transformer and
output graphs with another host embedding upload. These are profiling/estimate receipts,
not generation timing or qualification. Physical distributed cluster validation will
be tested later through Kluster's qualification workflow.

### Selected-expert vectors for small decoding batches

Vulkan MUL_MAT_ID now uses selected token/expert vector workgroups for one through
eight tokens with F32 activations. Larger prefill batches retain grouped matrix
execution. The shader uses the actual ID row stride, independent token activation
and output offsets, and supports both one input column and one column per selected
expert. This avoids dispatching grouped work for every unselected expert during
small-batch decode. Execution placement and operand storage formats are unchanged.

The general indexed matrix shader now bounds its final output row tile and its
vectorized expert-list reads. Odd output widths previously overwrote the next
expert's rows. Regression cases compare every result against decoded stored operands,
using distinct signed inputs, padded ID rows and repeated selections across tokens.
All thirteen indexed operand formats pass 208 cases on CPU and the selected physical
RX9070XT: one/two/eight/nine tokens, one/eight selected experts, and input/down
projections with35 output rows. Existing general quantized matrix F16 decoding has
an explicit magnitude-based rounding bound; small-batch vectors retain F32 bounds.

These operation contracts do not resolve model-level generated-token differences
or establish local performance acceptance. Matched model-backed gateway, placement
and paired benchmark receipts remain required. Physical distributed cluster gates
and validations will be tested later through Kluster's qualification workflow.


### Device-memory expert aggregation

The recipe27 candidate made weighted tensors with at least four selected experts use a packed
expert-axis SUM_ROWS reduction when their projection weights live in device memory.
Host weights retain the original serial adds, including CPU prefixes in partial
offload. One through three experts keep their existing path. The resulting
embedding/token tensor remains contiguous and preserves the selected expert order.
SUM_ROWS dispatch now carries the exact output row count and returns entire surplus
workgroups before any storage access. This makes the bounds explicit even on devices
where robust descriptor access masks out-of-range workgroups.

CPU and selected RX9070XT checks cover54 signed, odd-width, multi-token cases:
embedding35/513/2048, one/two/three/four/eight/sixteen experts and one/two/nine tokens.
Canaries following the output remain unchanged. They also remained unchanged in
the unguarded development run on this driver, so they are not a reproduced corruption
claim. Existing208 indexed-matrix cases and all three portable contracts pass.
Five alternating completed operation samples after warmup show CPU serial0.0052ms
versus packed0.0485ms, and GPU serial0.0699ms versus packed0.0641ms. These measurements
explain retaining host aggregation; they do not establish model performance acceptance.
Model-backed checks and strict paired benchmarks remain required before qualification.
Physical distributed cluster gates and validations will be tested later.

### MSVC native CPU instruction detection

With `GGML_NATIVE=ON`, Windows builds now check CPU feature bits, XSAVE/OSXSAVE,
enabled XCR0 vector state and completed compiler-intrinsic probes. AVX-512 VBMI,
VNNI and BF16 are selected independently only when both the host and compiler
support them. The previous detector left these extensions disabled and used unused
vector expressions that could be optimized away when detecting base instruction sets.
Probes are refreshed on reconfiguration; detector-header changes trigger CMake.

`GGML_NATIVE=OFF` preserves explicit target ISA settings. MSVC native detection
refuses cross-compilation without a runnable target; cross builds must choose
explicit flags. A native build is specific to its observed CPU/OS and cannot be
assumed safe on another machine. CPU/OS prerequisite vectors and an actual portable
Windows CPU build exercise selection/refusal separately from model acceptance.
Local model/gateway and paired performance checks remain required before qualification.
Physical distributed cluster gates and validations will be tested later.

### CPU graph synchronization and cancellation

CPU workers skip view/reshape/permute/transpose and empty nodes that perform no
computation, while retaining barriers after real work and final worker settlement.
Cancellation records the first node not to execute so late workers finish the
current node before stopping. A prior cancellation returns before graph outputs
are mutated or workers are started. These changes preserve tensor arithmetic.

The original controlled multi-worker cancellation check timed out, and the original
single-worker check modified the first output despite prior cancellation. Regression
checks cover real work interleaved with aliases, repeated cancellation/pool reuse,
metadata-only and empty graphs, and one/two/eight workers under OpenMP and pthreads.
Operation timings and fixtures do not establish model-level performance acceptance;
managed source/model/gateway and paired checks remain required. Physical distributed
cluster gates and validations will be tested later.


### Direct Vulkan expert-stride reduction

The recipe28 candidate made Vulkan device weights with4..32 selected experts reduce directly from the
permuted expert view. One shader lane owns each embedding/token output and adds
experts in legacy serial order. It requires no packing copy and no cross-lane
reduction tree. Host weights and other-backend paths keep their previous behavior.
The F32 strided SUM_ROWS path validates supported row widths/strides, binds exact
input/output byte ranges, accounts for unaligned view offsets and bounds surplus
workgroups. Unsupported strided widths refuse GPU support. Backend selection uses
the actual Vulkan buffer interface rather than an adapter display name.

54 reduction and208 indexed cases pass on CPU and selected RX9070XT. Inputs cover
odd35/513 widths, model2048 width, multiple experts/tokens, unaligned source views,
neighboring canaries and cancellation-sensitive values. The serial/direct result1
versus packed-tree result4 is recorded, preserving observable F32 addition order.
Completed alternating operation samples report serial0.077ms, packed0.069ms and
direct0.0666ms. They do not establish full-model performance or qualification.
The published27 packed model candidate regressed and failed strict acceptance;
actual model/gateway/paired checks remain required for this replacement. Physical
distributed cluster gates and validations will be tested later.

Shader generation now tracks new .comp files and the shader-generator executable
as CMake dependencies. The initial incremental build exposed missing generated
symbols after adding the shader; reconfiguration and the dependency correction
restore ordinary incremental generation. Clean managed builds remain mandatory
when consuming an immutable published pin.

### Model aggregation rollback and observable chat prompts

Published recipe28 completes its matched five-pair assessment, but throughput
ratios0.722/0.724/0.754/0.841 remain below Kluster's0.95 requirement. The packed
recipe27 graph also regresses. Default model expert aggregation therefore returns
to the original add chain for every backend. The separately checked bounded
strided Vulkan primitive remains available; operation microtimings do not justify
selecting it in the model graph.

`POST /apply-template` shares the actual generation renderer, including forced
tool instructions and Qwen3 non-thinking suffixes. Its strict boolean
`add_generation_prompt` defaults true; false renders only history. Native contracts
cover rendering, history and invalid/internal parameters. Exact per-engine prompt
token diagnostics confirm identical Qwen3-MoE inputs for the observed short/long
requests, while long chat generation still differs (134 versus71 generated tokens).
This rules out rendered-input differences for those requests, not arithmetic,
sampling or grammar differences. API/model/performance qualification remains open.
Physical distributed cluster gates and validations will be tested later.

### Optional Vulkan timing diagnostics

`GGML_VULKAN_PERF=ON` now retains normal command batching instead of forcing
per-node contexts, which previously broke reduction correctness. Backend-owned,
bounded timestamp queries report completed GPU command ranges as
`KLLM_PRIMA_VULKAN_TIMING` JSON. Queue timestamp support, counter width, nanosecond
period, wrap ambiguity and result availability are checked. Unsupported or absent
GPU work produces an unavailable record. Limits are 32 graphs per backend and
65,536 query slots; startup may consume the budget before decoding.

Instrumented servers advertise `diagnostic_vulkan_timing_v1`. Kluster assessment
rejects them as paired acceptance inputs. These command ranges change execution
dependencies and do not establish model throughput or an acceptance pass.
Ordinary builds keep `GGML_VULKAN_PERF=OFF`.

Performance acceptance and physical distributed validation are deferred, at the
user's request, to a live Kluster cluster of three computers. Existing thresholds
and unpassed qualification remain in place; no further local performance-gate
experiments are required. Implementation and correctness checks continue.

Timestamp semantics: [vkCmdWriteTimestamp](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html)
and [vkGetQueryPoolResults](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html).

### Full Jinja chat templates

The server now uses the Jinja parser, runtime and builtins from Kluster's pinned
llama.cpp baseline7fe450e19305b828c199d602c23a8337aaa1f03b. Exact source and MIT
license provenance are retained in common/jinja. Common targets require C++17.
The server renders full embedded model templates by default, including sources
longer than the old fixed metadata buffer. A missing embedded template uses the
legacy ChatML alias. --no-jinja retains legacy rendering explicitly.

--chat-template-file reads a bounded UTF-8 Jinja file; --chat-template accepts
inline source with --jinja. Templates are limited to64KiB and validated before
readiness. Template kwargs, BOS/EOS tokens, tools and the generation-prompt flag
enter fresh execution contexts. Parsed source is cached per thread; template
kwargs cannot overwrite message/token/generation inputs. Typed text is normalized
according to the template's detected content capabilities. Generation and
/apply-template use the same renderer without duplicating non-thinking suffixes.

The server advertises jinja_template_v1. Kluster freezes preset/custom source into
the signed run manifest, binds it to qualification context, and renders only on
rank zero; CLI workers consume the resulting token stream. Interactive native CLI
Jinja rendering is not added by this server change. Input-origin token marking
and complete automatic tool-call parsing remain separate implementation work.
This change grants no performance or distributed qualification.

### Named function streams

Named forced function calls now support streaming in Chat, Responses and Messages.
The native server advertises streaming_named_tools_v1. A call owns one ID for its
entire stream. Chat uses delta.tool_calls, Responses uses function-call argument
and item events, and Messages uses tool_use/input_json_delta blocks. Text channels
remain separate. Terminal usage and native logprobs are retained, including scored
tokens with no visible argument bytes during framing or UTF-8 buffering.

A length limit preserves partial arguments and reports length/incomplete/max_tokens;
it does not claim a completed action. Aggregate terminal content is not repeated
as a delta. Empty terminal output and errors after start settle the stream once.
Automatic tool selection, parallel automatic calls and full history conversion
remain separate implementation work. This change grants no performance or physical
distributed qualification; those checks remain deferred to the user's live cluster.

### Text tool-call history

The server advertises tool_history_text_v1. Chat assistant/tool messages, Responses
function_call/function_call_output items and Messages tool_use/tool_result blocks
can carry complete text tool history into a new request. Returned IDs are retained,
consecutive historical calls share their assistant turn, and typed tool results
use the pinned baseline's newline joining. Assistant null/missing content with
calls is normalized; argument strings are checked and supplied to Jinja in the
representation supported by that template. Templates that omit calls and unsupported
non-text content refuse instead of silently discarding history.

Kluster requires the tool_turns qualification feature for each protocol. Text tool
blocks are not vision input; nested image results still require vision evidence.
Anthropic tool_choice none does not request automatic tool choice. Native invalid
Responses input retains its HTTP400 boundary. Full multimodal history, stored
Responses state and automatic tool choice remain separate implementation work.
Performance and physical distributed acceptance remain deferred to the live cluster.

### Explicit local device ownership

The `explicit_devices_v1` capability adds ordered `--device`/`-dev` selection
(`LLAMA_ARG_DEVICE`) to the server and CLI. Use `--device none` for CPU-only
execution, including a zero-work coordinator. Unknown, duplicate, CPU and
unavailable GPU selections refuse. Explicit selection cannot mix legacy RPC
servers. Vulkan devices now participate in the native backend registry without
duplicate backend initialization; pinned host buffers belong to the selected
adapter rather than implicitly creating Vulkan0.

For layer placement, `--tensor-split` proportions select the first cumulative
normalized weight strictly greater than each offloaded layer position. Manifest
mode takes the proportions from the approved rank. Device order is significant;
Kluster binds physical UUIDs, sets the ordered visibility mask and passes the
remapped native names. Row, tensor and remote-expert placement remain unsupported.

Opt-in execution evidence reports only Vulkan adapters that completed matrix
work, with physical UUID/driver identity and actual per-layer backend ownership.
An initialized device with no assigned layers is not reported as executed work.
CPU-only reports contain CPU work and an empty GPU list. Local correctness checks
do not pass performance or physical distributed gates, which remain deferred
to the user's live cluster of three computers.

Driver-unavailable startup must also be tested: an empty visibility mask still
creates a Vulkan instance and does not exercise driver initialization failure.
The exported C initializer retains failure as an empty inventory, including
repeated discovery, and returns no GPU backend or host/device buffer for missing
adapters. It does not propagate C++ exceptions across MSVC /EHsc C callers.
The Windows/Vulkan CI job exercises unavailable-driver server contracts in
addition to normal compilation/contracts. This proves refusal/recovery behavior
and does not establish physical GPU or performance acceptance.

### Compatible draft-model decoding

The draft_simple_v1 capability selects draft-simple through --spec-type (draft
is an alias); none disables it. Modern --spec-draft-* aliases coexist with the
existing draft flags. Draft contexts use one local rank and independent KV state,
clear the target manifest/network scheduling, respect separate K/V cache types
and avoid autonomous profiling. Kluster launches the verified staged draft asset
only on rank zero with explicit CPU placement. The target's signed GPU/rank
placement remains unchanged, including a zero-transformer CPU coordinator.

Draft proposals are verified by the target sampler one token at a time, retaining
grammar state, logprobs, usage, stop/EOS/output limits and bounded KV rollback.
Prefix evaluation stays within the configured batch capacity and checks decode
status. Limits refuse invalid counts/probabilities before allocation. Opt-in
KLLM_PRIMA_SPECULATION counters establish completed verification rather than
performance. Unsupported speculative variants refuse instead of falling back.

Kluster preserves draft shard basenames in complete verified manifests, requires
the native capability before staging, and prices each independent draft context
on the master. CPU draft placement and all selected assets/settings are bound
to the execution scope. Native library support and same-host multi-rank checks
do not qualify physical distributed operation, other draft strategies or
performance; those acceptance gates remain unpassed and deferred.


## Rank-wide prompt cache implementation

`--cache-ram N` enables a process-owned prompt cache with a rank-local ceiling in
MiB; zero disables it. The native standalone CLI also accepts -1 for no ceiling;
Kluster requires a bounded nonnegative value and reserves that ceiling in host
RAM on every selected participant, including GPU workers and a zero-layer head.
Manifest mode takes the budget from the immutable execution settings. Runtime
capability `prompt_cache_v1` is required before managed asset preparation.

The server saves completed prompt state and completed continuations, chooses a
matching token prefix, and uses one eviction order for all ranks. Every rank keeps
its own KV bytes. A bounded ring transaction acknowledges save/restore/remove
across the selected ranks; snapshot bytes never travel to another rank. Failed
saves remove any partial entries before retry. Failed restores invalidate the
destination sequence everywhere before normal prompt recomputation. A missing or
foreign acknowledgement makes the context unusable and terminates the serving
run rather than generating from uncertain state. System-prompt and adapter
changes clear cached snapshots. Draft contexts rebuild their own compatible
prompt state through the existing bounded speculative path.

Sequence serialization now uses actual rank-owned layers and their absolute
indices, including empty heads. Partial-rank state cannot restore under a
different layer allocation. Full-model state retains the existing serialized
layout. Cache entries include tokens and bookkeeping in their budget charge.
Measured planner estimates remain unavailable for cached execution until there
are matching cost inputs; this feature does not justify a throughput claim.

This is implementation with production compilation only. Unit/native contract
and model-backed validation is deferred until integration implementation finishes,
as requested by the user. Later validation must cover dense/MoE local cache
reuse and eviction, repeated branches, cancellation and slot reuse, independent
slot sequences, malformed/budget inputs, partial-rank serialization, zero-work
head plus worker restoration and lost-acknowledgement cleanup. Physical distributed
and performance acceptance will be tested later on the user's three computers.
