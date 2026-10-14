# Model support expansion

Upstream `main` supports a handful of architectures for distributed execution. The
fork extends rank-owned and segmented execution to more than forty model families and
adds embedding, classifier and reranking serving paths. All of this is implementation
capability; per-architecture acceptance is deferred with the rest of the test phase.

## Rank-owned and segmented architectures

Architectures newly executable with absolute layer ownership and segmented placement
(commits titled "Add … segmented/rank-owned … execution"):

- Classic families: GPT-2, GPT-NeoX, MPT, StarCoder, StarCoder2, Refact, Bloom,
  StableLM, CodeShell, Orion, Xverse, Jais, Falcon, Chameleon.
- Qwen family: Qwen (1), Qwen2-MoE (with shared-expert costs), Qwen3 dense and MoE.
- Phi and MiniCPM: Phi-2, Phi-3, MiniCPM, MiniCPM3.
- IBM Granite: dense graphs with LongRoPE, routed and shared experts, sliding-window
  attention with fused expert projections.
- Google: Gemma (dense with LongRoPE), Gemma2 and Gemma3 with compact sliding-window
  KV and manifest-owned generation.
- Other MoE/routed: Command-R, DBRX, Grok, Arctic, Bitnet, DeepSeek2, OLMoE.
- OLMo family: OLMo, OLMo2 (compact sliding attention), OLMoE.
- OpenELM with per-layer dimensions.
- Recurrent and other: ChatGLM, Mamba, RWKV6, T5.

Each implementation follows the same pattern: typed per-layer graphs under absolute
ownership, architecture-specific graph support kept explicit, and unsupported
layouts/settings failing explicitly rather than falling back.

## Embedding execution and companion tools

- BERT embedding execution is rank-owned, with coordinator pooling inputs; embedding
  modes are bound to approved rank roles so an unassigned rank cannot run pooling.
- Embedding outputs honor projected output width (`n_embd_outputs`-style GGUF
  metadata), including bounded batch metadata for companion tools
  (embedding, rerank, retrieval examples updated accordingly).
- GGUF classifier labels and multi-output rank vectors are carried through serving.

## Reranking

- Baseline rerank ordering and `top_n` limiting were added for managed serving,
  preserving each input document's index and ordering by descending relevance.
- The TEI (text-embeddings-inference) request/response format is supported:
  `texts` in place of `documents`, and `return_text: true` echoes source text.

## Tokenizer and API semantics

- `parse_special` is honored in the tokenizer API so special-token text is parsed as
  control tokens exactly where the API promises it.

## Status

Per-architecture capability reports do not equal qualification; model correctness,
placement parity and API parity checks are part of the deferred test phase, and
physical distributed/performance acceptance stays with the live three-computer
cluster.

## Related commits

`91e5864`, `f984f77`, `cec1a40`, `d69be68`, `483a90f`, `731eb61`, `953b629`,
`c7bd885`, `99e179f`, `b30588a`, `8b5cbe4`, `ec2d4f0`, plus the ~35
"Add … segmented execution support" commits (`31c1e22` … `831948f`),
`7f6a136`, `b1b5e7c`, `2a0c5ac`, `2632e0f`, `077c9c5`, `146ccc7`, `aaee8b4`
