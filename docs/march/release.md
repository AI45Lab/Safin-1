# Preparing a public release

The source repository is [AI45Lab/Safin-1](https://github.com/AI45Lab/Safin-1).
Publication metadata, model IDs, licensing and version fields live in
`configs/release/publication.json`. The [supported scope](first-release-scope.md)
and [validation summary](validation.md) describe what the code supports.

## Review the source

Publish the reviewed tracked source directly from this Git checkout. Verify a
clean worktree, the intended public remote and public commit identities before
pushing. Local sharing overlays, personal run logs and build outputs must remain
untracked. An older private development history must not be reachable from the
public branches or tags.

```bash
git status --short
git log --oneline --decorate -5
git remote get-url origin
```

```bash
python scripts/march/check_public_release.py . --history --report /tmp/march-source-audit.json
python scripts/march/check_vendor_manifest.py --strict-provenance
python scripts/march/check_publication.py --scope code
python scripts/march/check_operator_correctness.py
python scripts/march/check_serving_evidence.py
python scripts/march/check_installation_evidence.py
```

The scanner reports file/line locations for known private paths, credential
patterns and generated artifacts. Preserve upstream licenses and notices as
described in [licensing](licensing.md).

## Validate the code

```bash
python -m pytest tests/march -q
MARCH_TEST_MEGATRON=1 PYTHONPATH="$PWD/third_party/mcore-bridge/src:$PWD" \
  python -m pytest tests/march/test_training_contract.py -q
```

Source archives and wheels are optional installation artifacts; they are not
required to publish the Git repository. If distributing them separately, build
from the reviewed source, verify installation outside the source tree and
preserve SHA256 sidecars. `scripts/march/export_source.py` can create a scanned
archive at a new path outside the checkout.

The [release receipts](evidence-index.md) record source fingerprints, build
environments and measured results. Source changes require the corresponding
checks; [acceptance](acceptance.md) lists the repeatable workloads.

## Publish weights separately

Follow [model delivery](model-release.md). Preserve each checkpoint's model type
and HSD/anchor runtime; training templates must not silently replace saved
metadata. Model cards and weights are published separately from source.

```bash
python scripts/march/check_publication.py --scope all
```

Record the published model revisions and download checks in the release
metadata. Code pushes, model uploads and release tags are explicit steps
of the publication workflow.
