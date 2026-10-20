# Contributing to MARCH / Safin-1

Open issues and pull requests in the repository hosting this checkout. MARCH
changes are maintained by this fork; upstream ms-swift is credited separately.

Include the model family, source revision, backend, complete launch command and
the report from `scripts/march/check_environment.py` in a bug report. Redact
private model/data paths and credentials before sharing logs.

Use Python 3.11 and run the checks appropriate to the change:

```bash
bash -n scripts/march/*.sh
PYTHONPATH="$PWD" python -m pytest tests/march -q
python scripts/march/check_public_release.py .
python scripts/march/check_vendor_manifest.py
```

Training changes need the Megatron contract suite and train/export/reload smoke;
kernel/cache changes need GPU forward/backward and boundary tests. Report skip
counts and hardware instead of treating skipped tests as passed.

Preserve upstream copyright and license notices. When changing a vendored file,
update its downstream change record and refresh the corresponding per-file hash
in `third_party/dependency-manifest.json`. Never replace a missing upstream commit
with a local import commit.

The supported scope is Safin-1 CPT/SFT and HF/vLLM inference, including optional
persistent state. Propose new backend or schema changes with compatibility and
validation evidence. Publication metadata is maintained in
`configs/release/publication.json`; unresolved fields remain explicit.
