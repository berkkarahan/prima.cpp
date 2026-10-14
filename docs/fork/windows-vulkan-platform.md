# Windows and Vulkan platform support

The fork makes prima.cpp build and run reliably on Windows with Vulkan, with explicit
GPU identity, checked native-feature detection and contained native failures. The
full contract list is maintained in
[docs/kluster-windows-vulkan.md](../kluster-windows-vulkan.md); this document
summarizes the platform-level implementations.

## Windows build robustness

Upstream `main` did not build cleanly for Windows server/profiler use. The fork adds:

- Quoted CMake policy versions so PowerShell-driven builds parse correctly.
- Explicit Windows performance API declarations and numeric header dependencies,
  and profiler include/clock-header fixes.
- Pagefile-backed capacity read through the supported PSAPI interface, so total
  memory accounting includes Windows page files.
- Termux memory inspection compiled only on Linux, with the environment probe
  guarded on non-Linux hosts.

## Checked MSVC native SIMD detection

`ggml/cmake/FindSIMD.cmake` was rewritten to detect host-safe MSVC native SIMD
extensions through checked probes: CPUID feature bits, `OSXSAVE`/`XCR0` state, and
runnable compiler intrinsics rather than optimized-away vectors. AVX-512
sub-extensions are enabled independently only when supported, and probes refresh on
reconfiguration. `GGML_NATIVE=OFF` with manual ISA selection remains supported.
The detection is exercised by contract tests (`prima-contract.cpp`).

## Explicit Vulkan device identity and ordering

- `llama-server --list-devices` emits `KLLM_PRIMA_CAPABILITIES`; Vulkan entries carry
  the physical device UUID, raw driver version and physical enumeration index. The
  physical index is the value accepted by `GGML_VK_VISIBLE_DEVICES`; after filtering,
  logical devices renumber from `Vulkan0`.
- Selection is by UUID and the reported identity is verified instead of assuming
  logical zero identifies a particular GPU. Logical ordinals are never treated as
  physical identities, and CPU/Metal/CUDA are never substituted in reporting.
- Ordered explicit devices (`--device`, `common/prima-devices.h`) select adapters in
  order with weighted layer ownership; unselected adapters do not initialize. The
  manifest mode requires explicit devices for any GPU-containing rank.
- An empty visibility mask is distinct from driver-unavailable startup, and each is
  reproduced and tested separately.

## Driver-unavailable handling

Vulkan initialization failures stay inside the C API boundary (MSVC `/EHsc` assumes
exported C calls do not throw), so an unavailable driver yields an empty GPU
inventory instead of an abort. CPU-only inference still works, repeated discovery
settles, and selecting an explicitly unavailable device refuses before model loading.
CI runs a driver-unavailable contract by pointing `VK_DRIVER_FILES` at a missing ICD
file.

## Native error containment across C boundaries

Model/context/backend/grammar/socket setup returns `nullptr`/status codes and refuses
readiness on failure instead of throwing across exported C calls. Backend setup
borrows its context; socket handles are closed only when distinct and owned, with
zero linger; reinitialization without closing live handles is refused. Sampler
operations are checked with thread-local error text for managed C++ callers; a null
grammar never becomes unconstrained generation. In the server, a sampler failure
fails only the owning HTTP request and clears its sequence while other requests and
valid snapshots continue.

## Execution evidence and readiness

- `KLLM_PRIMA_EXECUTION_EVIDENCE=1` records at most one completed prefill and one
  completed decode graph receipt in `KLLM_PRIMA_GPU_EXECUTION`, binding runtime,
  manifest and rank to physical device/driver identities and scheduled matrix-weight
  ownership, after graph synchronization, with warm-up excluded.
- `rank_readiness_v1` adds a separate `KLLM_PRIMA_RANK_READY` acknowledgement emitted
  only after model loading, warm-up, thread pools and control-socket setup (rank zero
  additionally requires HTTP health). `KLLM_PRIMA_RANK_CONFIGURED` alone is not
  readiness.

## Related commits

`a4a7e4d`, `43bfa68`, `70dd62b`, `847e513`, `733094a`, `aeb3afd`, `1a59a78`,
`ab2fbdd`, `acd7987`, `ad2a03c`, `a9d1099`, `c8a3744`, `9cc8b1e`, `91e5864`
