# Chat runtime: templates, parsers and reasoning

The fork vendors the pinned llama.cpp baseline's chat-template and tool-parsing
engines and wires them into Prima's serving stack, preserving baseline semantics
exactly.

## Vendored Jinja engine

`common/jinja/` copies the llama.cpp Jinja engine (lexer/parser/runtime/value,
upstream PR #18462 lineage) at the immutable Kluster baseline commit, with provenance
and license retained (`PRIMA-PROVENANCE.md`, `LICENSE`). The only structural
adaptation is using Prima's existing `json.hpp` instead of the newer vendor layout.
Input marking protects against special-token injection inside message content.

## Baseline chat runtime and tool parsers

`common/chat-runtime/` imports the baseline chat/template analysis, PEG parser,
chat-diff analyzer, automatic-tool parser generator, JSON-schema-to-grammar and the
model-specific parsers, adapted only to Prima's vintage model/token API and legacy
template-call signature (provenance in `PROVENANCE.md`):

- Specialized parsers cover eighteen model families, including `deepseek`,
  `gpt-oss`, `qwen3-coder`, `kimi-k2/k3`, `functionary-v3-2`, `gemma4`,
  `minimax-m3`, `cohere2moe`, `gigachat-v3`, `lfm2`, `ling3`, `minicpm5`,
  `ministral3`, `muse-glimmer`.
- `prima-auto-chat.cpp` is a narrow bridge exposing only serialized preparation and
  parsed output; the imported runtime does not replace Prima's model execution or
  HTTP adapters. Repeated per-call JSON elements must consume a call marker so
  nullable repetitions cannot create infinite grammars; transitive nullability is
  computed before recursion checks.
- Shared chat rendering (`common/prima-chat.{cpp,h}`) is used by chat generation,
  the Responses/Messages protocols, `/apply-template`, input-token counting and
  automatic tool preparation — one renderer for all of them.

## Reasoning budgets

`common/reasoning-budget.{h,cpp}` ports the baseline's reasoning budget state
machine: completion-scoped reasoning control, thinking on/off behavior, legacy
content handling and streaming contracts. Native sampling owns budget lifetime,
grammar suppression/replay, reset and clone behavior. CLI/server flags:
`--reasoning`, `--reasoning-budget`, `--reasoning-effort`, `--reasoning-format`.

## Template selection and keyword defaults

- `--jinja` / `--no-jinja` and `--chat-template-file` select the template source.
- Deployment Jinja keyword defaults (`--chat-template-kwargs`,
  `--chat-template-kwargs-file`, `template_kwargs_defaults_v1`) merge request values
  over the owned deployment object in the same renderer: bounded finite JSON,
  preserved dictionary insertion order, an owned content-addressed JSON file staged
  on rank zero, existing bytes verified without following symlinks, and native
  startup matching parsed defaults against the signed execution manifest. The file
  path avoids Windows argv limits; workers consume rendered tokens. Named legacy
  aliases never silently ignore keyword defaults.

## Contract testing

`prima-template-contract-test` and `prima-chat-parser-contract-test` (built in
portable CI) exercise the template contract and parser behavior against the pinned
baseline, including malformed-template rejection.

## Status

Template/parser behavior is compared against the pinned baseline; per-model tool-call
parity, thinking-mode coverage and broader template/API parity remain part of the
deferred test phase.

## Related commits

`cfca7ee`, `ae9ee59`, `704986f`, `c699b6f`, `59890f4`, `fb1ff38`, `400a4ff`,
`ccc8c59`, `3707d0a`
