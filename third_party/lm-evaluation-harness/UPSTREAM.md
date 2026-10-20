# Source provenance

- Upstream repository: https://github.com/EleutherAI/lm-evaluation-harness
- Reconciled upstream baseline: `6632cb788facbbbe9b47a5133bfec6a67aee5ee8`.
- Original clone revision: unknown; the baseline is not represented as proof of that revision.
- Local import commit: `21f8f43c083e3d9f6841da508d90874cd6f93066` (this is not an upstream SHA).
- Local modifications: Vendored optional evaluation tree; no MARCH evaluator validation claimed.

The exact current vendored contents are pinned by per-file SHA-256 hashes in
`third_party/dependency-manifest.json`. The original imported snapshot was compared with the public upstream Git inventory:
15,867 files match by Git blob identity, 9 RULER files differ and 33 files
are absent. PROVENANCE.json lists the complete import delta. This provides
a pinned, auditable upstream baseline while preserving the distinction
between reconciliation and an exactly recovered original clone SHA.

Public-scope cleanup: replaced an unrelated model download in BOS-handling tests with a local fast-tokenizer fixture and switched the dummy evaluator tokenizer to Qwen3.5-4B. This does not constitute end-to-end evaluator validation.

2026-10-16 evaluation integration: resolve vLLM thinking options without
overwriting template values or mutating caller mappings; explicit constructor
values take precedence and invalid thinking settings fail before engine allocation.
IFEval now validates complete local English punkt_tab resources and never
downloads from module import. The root evaluation entrypoints share the serving
profile and perform dependency/resource preflight. Synthetic task smokes are
kept separate from benchmark accuracy and production-scale acceptance.
