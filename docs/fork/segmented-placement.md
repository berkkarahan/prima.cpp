# Segmented placement and ring execution

The fork adds segmented execution: a model's layers can be split into repeated
segments across ranks, where an activation travels the approved physical ring and may
visit a rank several times. This is the mechanism behind the "rank-owned segmented
execution" model sweep described in [model-support.md](model-support.md).

## Absolute layer ownership

`common/prima-layer-layout.h` carries immutable absolute layer ownership through
model, context, tensor loading, KV storage, graph callbacks and sequence snapshots:

- An `owners[]` array maps every absolute layer to its owning rank; ownership is
  immutable after construction and validated (owner within world, world ≤ 32).
- O(1) rank-local tensor/graph indexing: each owned layer gets a local index, and
  contiguous ownership runs become distinct execution graphs with precomputed
  segment boundaries.
- Head input/output stay on rank zero; adjacent ownership is merged when counting
  local visits.
- The layout alone does not enable segmented transport; it is the indexing
  foundation the executor and planner build on.

## Ordered ring executor

The ordered segmented ring executor (`src/llama.cpp` groundwork) routes each
activation through actual physical ring hops, including relay ranks and repeated
visits:

- Activation packets bind to the exact manifest, micro-batch wave, stage, row count
  and model width; malformed, foreign or expired waves are refused and make the
  affected contexts unusable.
- Local graph visit order is preserved; zero-work relay ranks wait and forward
  without executing; a complete ring acknowledgement is required before the next
  micro-batch or KV command.
- Rank-zero input/output visits and per-device visits are reserved and priced by the
  planner (see [profiling-placement.md](profiling-placement.md)).

## Planned graph visits

Segmented placement is bound to planned graph visits: the manifest's
`prima_graphs`/`prima_graph_counts` reservations freeze the planner's visit counts
alongside graph structure, and rank visit counts are independently compared in both
Kluster and native startup. Missing or altered reservations refuse noncanonical
startup. Segmentation must be negotiated (before absolute layer-count launch
translation or candidate ring-edge additions).

## Output-only GPU coordinators

Rank-zero coordinators can run output projection on a GPU with zero transformer
offload (`output_gpu`, negotiated `output_only_gpu_v1`): the output graph and its
upload/logits-download costs are reserved even when transformer offload is zero, the
selected device is required at startup, and `ngl` counts transformer layers only.
Evidence distinguishes transformer GPU work from completed output projection in both
prefill and decode.

## Explicit limitations

- Row/tensor/expert segmentation is not implemented; only contiguous layer
  segmentation exists.
- Existing one-handoff-per-rank typed costs cannot score repeated visits or relays;
  such candidates stay capacity-first and explicitly unmeasured until covered.
- `segmented_placement` is an implementation capability, not a qualification.
  Physical distributed and performance acceptance remain deferred to the live
  three-computer cluster.

## Related commits

`cfdaff4`, `a314e91`, `5c19eed`, `54470ae`, plus the per-architecture commits listed
in [model-support.md](model-support.md)
