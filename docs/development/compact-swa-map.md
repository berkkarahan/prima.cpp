# Compact sliding-window row metadata

`common/prima-swa-map.h` provides the metadata foundation for compact GPT-OSS
sliding-window K/V storage. It does not enable compact execution.

Physical tensor rows map to global cells in the existing full KV cache. Positions
and sequence memberships stay in that authoritative cache, including shared
prefixes. No independent sequence table is introduced.

The decoder prepares a contiguous physical reservation before reserving the full
cache. Preparation is read-only; a failed full-cache reservation can discard its
ticket. All commit storage is allocated during preparation. After full-cache success, commit installs the incoming global IDs and
reclaims the rows approved at preparation. Owner identity and generation reject
foreign or stale tickets, including tickets from a destroyed map. Restore and
global-cell permutations validate a complete candidate before replacing state.
Defragmentation takes an old-global-to-new-global permutation and preserves the
compact tensor rows' physical locations.

The reclaim predicate must retain a cell if any shared sequence still needs it.
An incoming prefill must retain the history needed by its earliest token.
Suffix removal detects unavailable historical rows before mutation so the server's
prefix-reuse path can replay. Decoder admission also refuses missing history.

Layer-specific K/V views use compact capacity, head and active extent. The SWA
mask resolves physical rows through the map; free rows remain masked. K shifts
read deltas in that same physical order. Full-cache defragmentation remaps compact
global cell IDs without copying their tensor rows.

Snapshots distinguish compact layout with a versioned header, capacity, window
and owned-layer pattern. They save sparse compact rows by serialized global-cell
ordinal, retaining full dense metadata. Restore validates those ordinals, installs
rows in available physical slots and publishes the new map after reading tensor
data. Both transposed and non-transposed V storage use their actual pool strides.
Failed/truncated restoration clears the destination sequence (or whole cache).

The server's prefix reuse and speculative suffix removal now use an all-rank
rewind transaction. Read-only preflight traverses every physical rank; the head
then sends commit or abort and waits for settlement. Commands bind the manifest,
monotonic revision, sequence/range, phase, world and hop count. A worker with
evicted history can request replay even when the head owns no SWA layers. Missing,
foreign or incomplete settlement faults the context and requires restart.

`--swa-full` explicitly selects the currently active full-size layout. The C API
defaults to full storage and refuses premature compact activation. Kluster resolves
and prices full SWA storage, including checkpoints, and passes the flag to every
rank. This removes the mismatch between compact estimates and full allocation.

CLI session reuse now declares the exact local session token IDs and compares a
SHA-256 prefix digest on every transformer rank. The digest uses a domain label,
prefix count and little-endian uint32 token IDs. A rank also requires its cached
predecessor at the earliest decode boundary; empty relays need no tensor state.
Missing/mismatched sessions or evicted history cause all-rank clearing and replay.
The boundary includes any last token that must be decoded to rebuild logits.
Declarations are invalidated by checked mutation or actual decode.

The local draft helper replays its selected target-prompt window when prefix
rewind fails. Ordinary CLI context shift retains exact evaluated token history
independently of session-file saving, uses checked removal and position shifting,
and replays retained tokens on all ranks if either operation needs history.

Checked positional transforms simulate add/divide operations before mutation,
validate signed position/delta bounds and inspect the resulting window for every
shared sequence. They use the same prepare/commit/abort settlement, with versioned
transform operands bound to the command. Recurrent models check their actual
tail state rather than treating it as a transformer cache.

The server now keeps completed token history independently of prompt caching.
Only successful decode views extend it; speculative verification is trimmed to
the accepted prefix after settlement. Failed batch decoding trims optimistic
prompt-cache entries to completed history before a later request can reuse them.
Reused prompt prefixes initialize history from the exact incoming token IDs.

Ordinary server context shift preserves the complete system prefix and uses
checked all-rank removal and position shifting. Missing transformed SWA history
causes per-sequence retained-token replay. The history records base/active aLoRA
weight phases; replay restores the system prefix, switches the appropriate phase
for each contiguous token segment, then restores the live request selection.
Sampler state and generated output stay with the existing request. Replay failure
clears that sequence through the normal sampling-failure path.

Ordered transform programs now preflight up to eight add/divide steps together.
They validate every intermediate signed position/delta and check window history
after the final result. The entire program, revision and phase are bound to
all-rank prepare/commit/abort settlement; individual intermediate windows do not
trigger a spurious replay decision.

CLI self-extend submits its three operations as one program and tracks evaluated
token positions. Missing transformed history causes all-rank replay at those
positions. Session files omit the transform schedule, so self-extend rebuilds their
input tokens instead of treating compressed state as an ordinary reusable prefix.
Recurrent self-extend does not accumulate transformer replay history.

Compact mode stays disabled pending compact managed budgets and server self-extend
position/replay handling. The new paths remain untested and do not establish
distributed runtime acceptance. Full storage remains active.

Only production compilation has been checked. Unit/native/model tests and test
fixes are deferred until the required integration implementation is finished.
Physical distributed and performance acceptance remain deferred to the user's
three-computer cluster. This groundwork does not qualify inference or claim a
memory/performance improvement.
