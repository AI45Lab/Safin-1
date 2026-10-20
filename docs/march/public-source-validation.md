# Public-source preparation validation — 2026-10-06

An independent source tree was created from the scanned internal snapshot,
without its Git history or tracked checkpoint overlay. Original/internal
checkouts were not modified. Runtime code is retained; this change focuses on
public entrypoints, provenance and release controls.

- Full `tests/march` with `MARCH_TEST_MEGATRON=1`: 80 passed, 40 skipped,
  21 unittest subtests passed in 93.05 seconds.
- GPU tests were skipped because this node has no GPU. No new GPU claim follows.
- Python AST and public shell syntax checks passed.
- Installation dry-runs covered CPU, training and vLLM recipes.
- Dependency resolution for the source installation succeeded: 147 packages.
- The test used the existing prepared Python 3.11 environment, not a fresh public
  CUDA image. Clean installation/native-build acceptance remains pending.

mcore-bridge's upstream revision was recovered by exact import-tree identity.
lm-eval's imported inventory was reconciled with a pinned upstream baseline;
its original clone revision remains unknown. Detailed differences are retained
in `third_party/lm-evaluation-harness/PROVENANCE.json`.

Strict publication checks intentionally fail until public identities, licensing,
evaluation results and the remaining acceptance items are completed. Draft
artifact generation and successful CPU checks do not imply publication approval.


Final preparation checks:

- Source tree scan: no findings, including tracked-private-overlay checks.
- Strict vendored hash/license/provenance check: passed for all four components;
  evaluator reconciliation is explicitly distinguished from original clone identity.
- Wheel build and isolated `/tmp` installation passed. Imports came from the
  installed wheel; both Safin-1 model types registered successfully.
- FlashQLA LICENSE, NOTICE and UPSTREAM.md are present in the wheel.
- Documentation links and final Python/Shell syntax checks passed.
- Final release-tool tests additionally cover deleted-secret history and the
  vLLM version override required by Git-free vendored exports.
