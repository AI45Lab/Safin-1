# Source checks

Run these checks before publishing source:

```bash
python scripts/march/check_public_release.py . --history --report /tmp/march-source-audit.json
python scripts/march/check_vendor_manifest.py --strict-provenance
python scripts/march/check_publication.py --scope code
```

The source scanner checks known private-path, credential and generated-artifact
patterns, tracked private overlays and Git identities. Findings report file/line
locations without printing matched secrets.

The vendor checker verifies file hashes, LICENSE/NOTICE files and recorded
upstream revisions. The publication checker checks release metadata together
with the source-bound operator, installation and serving receipts.

Use [validation](validation.md) for the test matrix and
[release](release.md) for the Git publication workflow. Local model locations,
run logs and maintenance backups stay outside tracked source.
