# Vendored distribution scope

This source snapshot pins upstream vLLM at `20cac26b197a2e9530a787a5c1dbf5163533b2ff` and retains the MARCH changes.

The downstream distribution omits upstream CI automation under `.buildkite/`
and `.github/`, plus large binary screenshots and generated HTML examples under
`docs/assets/`. Retained Markdown uses upstream asset URLs pinned to that commit.
The root repository owns CI for the supported Safin-1 integration.

Runtime Python, CUDA/C++ sources, CMake files, build requirements and relevant
tests remain available. Required LICENSE, NOTICE and source/change records are
retained. Root `third_party/dependency-manifest.json` records every retained file;
this distribution change does not renew or broaden any GPU acceptance.
