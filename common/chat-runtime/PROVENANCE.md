# Baseline chat parser provenance

Imported from ggml-org/llama.cpp at
7fe450e19305b828c199d602c23a8337aaa1f03b, retaining its MIT LICENSE.
This directory contains the baseline chat/template analysis, PEG, schema/grammar
and specialized parser implementation. Its common.h adapts the vintage Prima
model/token API. Legacy template calls receive Prima's required model argument.
This import does not itself advertise automatic-tool serving support.

Adaptations are intentionally limited to the vintage model/token API, legacy
template-call signature, and the renamed PrimaBaselineBuiltinRule type (avoiding
a collision with Prima's older schema converter). The repeated per-call JSON
parser requires each repeated element to consume a call marker; optionality is
handled by the outer automatic-choice parser. This prevents a nullable repetition
from producing an infinitely expanding character grammar.

prima-auto-chat.cpp is a local bridge exposing only serialized preparation and
parsed output. The imported runtime does not replace Prima's model execution or
its existing HTTP adapters. Lazy trigger state and transitive nullable-cycle
rejection are implemented in Prima's native character grammar and sampler.
