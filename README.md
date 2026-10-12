# prima.cpp: Kluster LLM fork

This repository is a **fork of [OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp/tree/main)**. The linked `main` branch is the original project. This fork develops Windows/Vulkan support and integration with Kluster LLM on [`codex/kluster-integration`](https://github.com/berkkarahan/prima.cpp/tree/codex/kluster-integration), proposed in [PR #1](https://github.com/berkkarahan/prima.cpp/pull/1).

Prima.cpp distributes GGUF model inference across heterogeneous devices, combining CPU, GPU, memory and storage resources. This fork keeps Prima's native execution architecture while making it easier to build, orchestrate and validate through Kluster.

## What's new in this fork

- **Windows and Vulkan integration.** Native server, worker and profiler builds, explicit GPU selection, and verification of completed GPU execution on Windows.
- **Broader model and serving support.** Qwen3 dense and MoE models, expanded text APIs and streaming, structured output, forced function calls, token logprobs and shared chat-template rendering.
- **Predictable cluster orchestration.** Explicit execution manifests and rank readiness let Kluster control placement and lifecycle, including a coordinator that delegates all transformer layers to workers.
- **Measured planning inputs.** Model-aware CPU/GPU profiling and checked memory, storage and transfer measurements provide inputs for Kluster's placement planner.
- **Reproducible development and validation.** Pinned dependencies, portable build CI, native contract tests and model-backed checks make changes traceable and testable.

These are development capabilities, with the validation boundaries described below.

## Current status

Local Windows/Vulkan execution has been exercised with Qwen3 dense and MoE models on an AMD Radeon RX 9070 XT, including serving through Kluster's gateway. CI builds and native contracts cover CPU configurations on Windows, Linux and macOS, plus Windows/Vulkan compilation.

Complete model/API/placement compatibility remains in progress. Performance acceptance will be tested later on a live Kluster cluster of three computers. Passing build or operation tests does not establish full inference qualification, and this fork does not claim the original project's published speedups for its new configurations. Prima remains an optional, qualification-gated engine in Kluster; llama.cpp remains its default.

**Physical distributed cluster gates and validations will be tested later**, including mixed-platform/backend inference, network failure recovery and distributed performance. They are not counted as passed. Linux/macOS Vulkan hardware acceptance also remains open.

## Build on Windows with Vulkan

Install Git, CMake, the Visual Studio C++ toolchain and the Vulkan SDK. From a developer shell:

```powershell
git clone --branch codex/kluster-integration https://github.com/berkkarahan/prima.cpp.git
cd prima.cpp
cmake -S . -B build -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
cmake --build build --config Release --target llama-server llama-cli profile-tool prima-contract-test prima-server-contract-test prima-profiler-contract-test -j 4
ctest --test-dir build -C Release --output-on-failure
```

CMake retrieves the pinned ZeroMQ dependencies. Release executables are under `build/bin/Release/`. For portable CPU build configurations, see the [CI workflow](.github/workflows/kluster-build.yml).

Inspect the available devices before choosing GPU placement:

```powershell
.\build\bin\Release\llama-server.exe --list-devices
```

For managed use, follow Kluster's installer and runtime-library workflow with an immutable source commit. Kluster supplies verified model assets and execution manifests; the standalone build above does not enable a qualified Kluster run.

## Serving and integration

The native server provides chat completions and text Responses/Messages APIs through its existing inference scheduler. Supported requests include streaming text, structured JSON, named nonstreaming function calls and token logprobs. `/apply-template` exposes the same prompt renderer used by chat generation.

Support is scoped: automatic or streaming tools, multimodal requests and complete template/API parity are still open. See the [server API documentation](examples/server/README.md) for endpoint behavior and limitations.

Kluster owns membership, authenticated preparation, final placement and process lifecycle. Prima executes the assigned native work and supplies profiling measurements. Manifest mode preserves the approved ranks and placement instead of autonomously changing them.

Keep native rank traffic on the approved private network and use the orchestrator's scoped admission rules. See the [Windows/Vulkan integration guide](docs/kluster-windows-vulkan.md) for device identity, execution evidence, readiness and validation details.

## Documentation

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
