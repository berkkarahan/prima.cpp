# Compact sliding-window row metadata

`common/prima-swa-map.h` provides the metadata foundation for compact GPT-OSS
sliding-window K/V storage. It does not enable compact execution.

Physical tensor rows map to global cells in the existing full KV cache. Positions
and sequence memberships stay in that authoritative cache, including shared
prefixes. No independent sequence table is introduced.

The caller prepares a contiguous physical reservation before reserving the full
cache. Preparation is read-only; a failed full-cache reservation can discard its
ticket. After full-cache success, commit installs the incoming global IDs and
reclaims the rows approved at preparation. Owner identity and generation reject
foreign or stale tickets, including tickets from a destroyed map. Restore and
global-cell permutations validate a complete candidate before replacing state.
Defragmentation takes an old-global-to-new-global permutation and preserves the
compact tensor rows' physical locations.

The reclaim predicate must retain a cell if any shared sequence still needs it.
An incoming prefill must retain the history needed by its earliest token.
Sequence rewind must detect unavailable historical rows and replay rather than
reuse absent data. Those runtime consumers remain to be implemented.

Compact mode must stay disabled until slot admission, tensor views, attention
masks, sequence operations, K shifts, defragmentation, snapshots and managed
memory budgets are integrated together. Full storage remains the active mode.

Only production compilation has been checked. Unit/native/model tests and test
fixes are deferred until the required integration implementation is finished.
Physical distributed and performance acceptance remain deferred to the user's
three-computer cluster. This groundwork does not qualify inference or claim a
memory/performance improvement.
