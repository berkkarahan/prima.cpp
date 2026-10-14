# prima.cpp: Kluster LLM fork

This repository is a **fork of [OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp/tree/main)**. The `main` branch carries this integration; the original project remains at [OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp/tree/main), and the work was developed on [`codex/kluster-integration`](https://github.com/berkkarahan/prima.cpp/tree/codex/kluster-integration), proposed in [PR #1](https://github.com/berkkarahan/prima.cpp/pull/1).

Prima.cpp distributes GGUF model inference across heterogeneous devices, combining CPU, GPU, memory and storage resources. This fork keeps Prima's native execution architecture while making it easier to build, orchestrate and validate through Kluster.

## What's new in this fork

- **Windows and Vulkan support.** Reproducible server, worker and profiler builds with explicit GPU identity and execution reporting.
- **Broader inference and serving.** Additional dense, MoE and embedding model paths, compact sliding-window storage, modern text APIs, streaming tools, chat templates, adapters and draft decoding.
- **Orchestrated distributed execution.** Immutable rank/layer manifests, repeated layer segments, head-only coordination and owned readiness, cancellation and recovery protocols for Kluster.
- **Model-aware placement inputs.** Typed CPU/GPU work and memory estimates, plus scoped storage and network profiling for Kluster's planner.

## Current status

This fork is under active integration. Earlier local Windows/Vulkan runs exercised Qwen3 dense and MoE models on an AMD Radeon RX 9070 XT, but those historical results do not qualify the current source. Production compilation continues; unit, native and model tests, and test-fix work, are deferred until the full required implementation is finished.

Complete model, API, platform, accelerator and arbitrary placement parity remains unfinished. The newer model paths and capability reports are development work, with acceptance still pending. Prima remains an optional, qualification-gated Kluster engine; llama.cpp remains its default.

**Physical distributed cluster gates and validations will be tested later on the user's live three-computer cluster.** This includes mixed-platform/backend inference, network failure recovery and performance acceptance. These gates remain unpassed. Linux/macOS Vulkan hardware acceptance also remains open. This fork does not claim the original project's published speedups for its new configurations.

## Prerequisites and Building

Ensure the following build and runtime prerequisites are installed:

- `gcc` >= 9.4.0 (or Xcode Command Line Tools / MSVC C++ toolchain)
- `make` >= 4.2.1
- `cmake` >= 3.16.3
- `fio` >= 3.16
- `zeromq` (`zmq` >= 4.3.2)
- `highs` (`HiGHS` >= 1.9.0, for workload distribution / `-DUSE_HIGHS=ON`)
- `git`, `wget`, `curl`
- Vulkan SDK / MoltenVK (when building with `-DGGML_VULKAN=ON`)

**macOS (Homebrew):**

```shell
brew install gcc make cmake fio git wget highs zeromq curl molten-vk vulkan-headers vulkan-loader shaderc spirv-headers
```

**Linux (Ubuntu / Debian):**

```shell
sudo apt-get update -y && sudo apt-get install -y build-essential gcc g++ make cmake fio git wget libzmq3-dev curl pkg-config libvulkan-dev glslc spirv-headers
```

### Build on macOS or Linux

```shell
git clone https://github.com/berkkarahan/prima.cpp.git
cd prima.cpp
cmake -S . -B build -DGGML_RPC=ON -DGGML_VULKAN=ON -DUSE_HIGHS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_BLAS=OFF "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
cmake --build build --config Release --target llama-server llama-cli profile-tool -j 4
```

### Build on Windows with Vulkan

Install Git, CMake, the Visual Studio C++ toolchain and the Vulkan SDK. From a developer shell:

```powershell
git clone https://github.com/berkkarahan/prima.cpp.git
cd prima.cpp
cmake -S . -B build -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DUSE_HIGHS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
cmake --build build --config Release --target llama-server llama-cli profile-tool -j 4
```

CMake retrieves the pinned ZeroMQ and HiGHS (`v1.9.0`) dependencies when `-DUSE_HIGHS=ON` is set. Release executables are under `build/bin/` (or `build/bin/Release/` on Windows). For portable CPU build configurations, see the [CI workflow](.github/workflows/kluster-build.yml).

Inspect the available devices before choosing GPU placement:

```powershell
.\build\bin\Release\llama-server.exe --list-devices
```

For managed use, follow Kluster's installer (`uv run scripts/install.py --engine primacpp`) and runtime-library workflow with an immutable source commit. Kluster automatically installs missing prerequisite packages, supplies verified model assets and execution manifests; the standalone build above does not enable a qualified Kluster run.

## Serving and integration

The native server provides chat completions and text Responses/Messages APIs through its existing inference scheduler. Supported requests include streaming text, structured JSON, automatic/required/named function calls, parallel tool streams, supplied text tool history and token logprobs. `/apply-template` exposes the same prompt renderer used by chat generation.

Support is scoped: automatic tools require a tool-capable Jinja template and a model that emits its declared call format. Ordinary or unmarked JSON output stays text, matching the pinned baseline parser. Multimodal requests and complete template/API parity remain open. Responses rejects nonempty `previous_response_id`, as does the pinned baseline; it does not provide a stored-response service. See the [server API documentation](examples/server/README.md) for endpoint behavior and limitations.

Kluster owns membership, authenticated preparation, final placement and process lifecycle. Prima executes the assigned native work and supplies profiling measurements. Manifest mode preserves the approved ranks and placement instead of autonomously changing them.

Keep native rank traffic on the approved private network and use the orchestrator's scoped admission rules. See the [Windows/Vulkan integration guide](docs/kluster-windows-vulkan.md) for device identity, execution evidence, readiness and validation details.

## Fork implementation summaries

The documents below summarize each new implementation area the fork introduces over the original project; they are collected in [docs/fork/overview.md](docs/fork/overview.md).

| Document | Area | Summary |
|---|---|---|
| [docs/fork/windows-vulkan-platform.md](docs/fork/windows-vulkan-platform.md) | Platform | Windows build robustness, checked MSVC SIMD detection, explicit Vulkan device identity/ordering, driver-unavailable handling, native error containment |
| [docs/fork/vulkan-execution.md](docs/fork/vulkan-execution.md) | GPU kernels | Vulkan correctness and dispatch: quantized vector loops, tensor-view strides, strided MoE expert reduction, attention batching, optional timing diagnostics |
| [docs/fork/kluster-manifest-mode.md](docs/fork/kluster-manifest-mode.md) | Orchestration | Immutable execution manifests with verified assets, rank readiness acknowledgements, zero-transformer coordinators, synchronized cancellation |
| [docs/fork/segmented-placement.md](docs/fork/segmented-placement.md) | Placement | Absolute layer ownership, ordered ring executor with relay ranks, planned graph-visit reservations, output-only GPU coordinators |
| [docs/fork/model-support.md](docs/fork/model-support.md) | Models | Rank-owned/segmented execution for 40+ architectures, rank-owned BERT embeddings, classifier labels, rerank ordering and TEI format |
| [docs/fork/gpt-oss-mxfp4.md](docs/fork/gpt-oss-mxfp4.md) | Models | GPT-OSS loader, segmented graphs and Harmony tokenizer, MXFP4 storage on CPU/Vulkan, attention sinks, typed profiling |
| [docs/fork/kv-storage-context.md](docs/fork/kv-storage-context.md) | KV state | Compact sliding-window KV storage and row mapping, server/CLI KV rewind and self-extend replay, bounded prompt caching, per-slot context checkpoints |
| [docs/fork/chat-runtime.md](docs/fork/chat-runtime.md) | Templates | Vendored Jinja engine and baseline chat/parser runtime with per-model tool parsers, reasoning budgets, shared rendering, template keyword defaults |
| [docs/fork/serving-apis.md](docs/fork/serving-apis.md) | Serving | Chat/Responses/Messages protocols, automatic/required/parallel tool streams, tool history, structured JSON, token logprobs, completions and FIM infill |
| [docs/fork/lora-draft-decode.md](docs/fork/lora-draft-decode.md) | Adapters | Rank-owned managed LoRA with synchronized selection and per-request batches, embedding/tied-output adapters, isolated draft-simple decoding |
| [docs/fork/profiling-placement.md](docs/fork/profiling-placement.md) | Profiling | Typed model/kernel profiler with verified operands, exclusive native I/O probes, GPU embedding/logits measurements, opt-in telemetry |

## Documentation

- [Fork implementation summaries](docs/fork/overview.md)
- [Windows/Vulkan integration and validation](docs/kluster-windows-vulkan.md)
- [Native server API](examples/server/README.md)
- [Build and contract-test CI](.github/workflows/kluster-build.yml)
- [Original project and its documentation](https://github.com/OpenCPIL/prima.cpp/tree/main)

## Attribution and license

Prima.cpp was created by the [OpenCPIL team](https://github.com/OpenCPIL/prima.cpp) and builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). Their research results and original usage examples are available in the upstream repository. This fork retains the [MIT license](LICENSE).

When citing the original research:

```bibtex
@inproceedings{li2026primacpp,
  title={Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters},
  author={Li, Zonghang and Li, Tao and Feng, Wenjiao and Xiao, Rongxing and She, Jianshu and Huang, Hong and Guizani, Mohsen and Yu, Hongfang and Ho, Qirong and Xiang, Wei and Liu, Xue},
  booktitle={14th International Conference on Learning Representations},
  year={2026},
  url={https://iclr.cc/virtual/2026/poster/10008093}
}
```
