# Preparing a public release

The source repository is [AI45Lab/Safin-1](https://github.com/AI45Lab/Safin-1).
Publication metadata, model IDs, licensing and version fields live in
`configs/release/publication.json`. The [supported scope](first-release-scope.md)
and [validation summary](validation.md) describe what the code supports.

## Review the source

Export a reviewed snapshot with `scripts/march/export_source.py`. Start public
Git history from that snapshot with a public contributor identity. The
development checkout's Git history, local sharing overlays, personal run logs
and build outputs are outside the public distribution. Do not copy the complete
development directory as a release.

```bash
python scripts/march/check_public_release.py . --history --report /tmp/march-source-audit.json
python scripts/march/check_vendor_manifest.py --strict-provenance
python scripts/march/check_publication.py --scope code
python scripts/march/check_operator_correctness.py
python scripts/march/check_serving_evidence.py
python scripts/march/check_installation_evidence.py
```

The targeted privacy scanner reports locations without printing secret values.
It is not a guarantee that every credential format is detected. Preserve all
upstream licenses/notices and use the [licensing guide](licensing.md).

## Validate and package

```bash
python -m pytest tests/march -q
MARCH_TEST_MEGATRON=1 PYTHONPATH="$PWD/third_party/mcore-bridge/src:$PWD" \
  python -m pytest tests/march/test_training_contract.py -q
python -m build
python scripts/march/export_source.py --output /path/to/Safin-1-source.tar.gz
```

Use a new archive path outside the checkout. Verify source-to-sdist-to-wheel
installation outside the source tree and preserve SHA256 sidecars.

The [sanitized release receipts](evidence-index.md) retain exact source hashes
and measured scopes. Operator, TP/DP serving and installation checks reject
missing or stale evidence; updating fingerprints alone cannot renew a result.
Documentation-only cleanup does not broaden any original GPU acceptance.
CPU CI does not establish GPU correctness or throughput. See the
[repeatable acceptance commands](acceptance.md) for new GPU checks.

## Publish weights separately

Follow [model delivery](model-release.md). Preserve each checkpoint's model type
and HSD/anchor runtime; training templates must not silently replace saved
metadata. Model cards and weights are published separately from source.

```bash
python scripts/march/check_publication.py --scope all
```

Published model revisions and download verification remain separate from the
code-only check. The explicit release policy controls optional evaluation and
training-data disclosure requirements without marking unknown reviews complete.
Creating local packages or passing CI does not upload weights, push source,
create a public tag or publish to PyPI.
