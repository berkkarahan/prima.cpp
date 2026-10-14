# Vendored template engine provenance

Copied from ggml-org/llama.cpp common/jinja, common/json.{h,cpp} and
common/unicode.{h,cpp} at immutable commit
7fe450e19305b828c199d602c23a8337aaa1f03b (Kluster baseline).
These sources use llama.cpp's MIT license, retained verbatim in this directory's LICENSE.
The JSON backing include uses Prima's existing json.hpp instead of the newer
vendor layout. A trailing blank line in unicode.cpp is removed. Parser, runtime
and builtins retain baseline implementations.
