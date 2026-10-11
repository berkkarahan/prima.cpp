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
