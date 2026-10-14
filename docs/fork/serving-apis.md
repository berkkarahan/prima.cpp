# Serving APIs and tool calling

The fork's native server (`examples/server/`) gains the modern llama.cpp text API
surface — three protocols, function tools with streaming, structured JSON, logprobs
and completion routes — through its existing inference scheduler. Endpoint behavior
and limits are documented in [examples/server/README.md](../../examples/server/README.md);
new adapter modules are `protocols.hpp`, `auto-tools.hpp`, `tool-history.hpp`,
`structured.hpp` and `logprobs.hpp`.

## Three text protocols

- OpenAI-compatible Chat Completions, plus stateless **Responses** and **Messages**
  text protocols with streaming contracts, all sharing the scheduler and renderer
  (`protocols.hpp`). Responses/Messages capability is advertised; requests with
  nonempty `previous_response_id` are rejected (no stored-response service), and
  multimodal is not implemented.
- Messages maps `auto`/`any`/`tool`/`none` tool choice including
  `disable_parallel_tool_use`.

## Function tools

- The baseline tool parser and lazy grammar foundation (`chat-runtime` +
  `prima-lazy-grammar-contract`) drive **automatic, required and named** tool calls,
  with **parallel tool streams** where the model emits several calls.
- Streaming keeps text, reasoning and function-argument deltas separate with stable
  per-call/per-item identities, terminal usage, once-only terminal records and
  explicit length termination; framing/UTF-8 buffering that produces no visible text
  still emits the call identity.
- Supplied **text tool history** is accepted on all three protocols, normalized per
  protocol before Jinja rendering (assistant/tool messages, Responses
  `function_call`/`function_call_output`, Messages `tool_use`/`tool_result`).
- Generated tool grammars are prefilled with the template's existing generation
  prefix (special-token tokenized, leading-space skipped, reapplied after sampler
  reset); user-provided grammars are never prefilled. Explicit output grammars
  cannot be combined with automatic tool grammars.
- Unmarked JSON from the model remains text, matching the pinned baseline parser;
  fenced structured JSON supports streamed content parsing. Later hardening commits
  improve JSON-native tool analysis, split grammar translation units, and add
  JSON-native content fallback with preserved-token refresh after patches.

## Token logprobs

OpenAI-style choice and stream token logprobs (`logprobs.hpp`): sampled-token
logprobs, top alternatives and UTF-8 token pieces are preserved from the native
sampler — never re-derived by retokenizing output text.

## Completion routes

- `/v1/completions` (OpenAI text completions) accepts text or token IDs and multiple
  prompts, `n` choices (1–16), SSE streaming and optional `logprobs`/`top_logprobs`.
- The legacy `/completion` and `/completions` routes keep their native format;
  internal preparation parameters are reserved on raw completion/infill routes too.
- FIM infill was repaired: prefix/suffix handling, optional `prompt`, `input_extra`
  context, and literal tokenization of user FIM text.
- `/tokenize` honors `parse_special`; `/apply-template` (`apply_template_v1`) exposes
  the generation renderer without running inference.

## Status

Automatic tools require a tool-capable Jinja template and a model that emits its
declared format. Automatic-choice, parallel-call and complete tool-history support
are kept as separately gated capabilities; tool-stream, server and chat-parser
contract tests exist and run in CI, with broader parity checks deferred.

## Related commits

`f221716`, `3e41d47`, `3707d0a`, `ccc8c59`, `673cd92`, `37828fe`, `501585d`,
`83555d7`, `b18ae11`, `aaee8b4`, `0c2f229`, `6681aec`, `f4317ea`, `6c5db7d`,
`3107200`
