# Flash Linear Attention research runtime

Source: https://github.com/fla-org/flash-linear-attention

Version: 0.5.0; upstream tag `v0.5.0`, verified source commit
`3a9ce1c83a13994d824dbb3421e2989d330bb38b`.

The two original PyPI wheels are pinned in `PROVENANCE.json`. On 2026-10-16,
all 369 ordinary package files matched the fixed upstream Git archive byte for
byte. The remaining `fla/__init__.py` matches the namespace initializer generated
by upstream `scripts/build_packages.py`; its literal was checked without
executing upstream code. This establishes the source revision by content,
not by inferring it from the version number.

The 374 historical overlay files retain their recorded SHA-256 values. They
include 29 changed or added files relative to the upstream wheels, including
four GDN2 files. Local changes cover GDN/KDA/Transformer configuration,
cache/generation compatibility, packed attention, state-producer launch geometry
and TileLang backward paths. No research source bytes changed in this review.
The complete library is retained because its catalogs import each other. Only
the research launcher prepends it to PYTHONPATH; normal Safin-1 launchers retain
their existing dependencies.

To reproduce offline on CPU, prepare the two wheel filenames and `upstream.tar.gz`
from the URLs in `PROVENANCE.json`, then run from the repository root:

```bash
.venv/bin/python scripts/march/verify_fla_provenance.py --artifacts /path/to/fla-artifacts
.venv/bin/python scripts/march/check_vendor_manifest.py --strict-provenance
```

The verifier checks artifact hashes before reading sources and rejects
unrecorded local files or changes. Source provenance does not certify kernel
correctness, speed or a clean native build.
