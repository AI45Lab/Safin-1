# Research validation scope

This workspace is independent of Safin-1 4B/35B. Its public inputs are the
model JSON files, training recipes and caller-supplied tokenizer/dataset paths.
Personal execution reports, original run identifiers and per-step training
curves are excluded from the source distribution.

CPU contracts check all seven model geometries, model-type registration,
launcher arguments, source provenance, generation masks and strict tiny-model
save/load. GPU checks have exercised dense HSD-GDN training, checkpoint
save/resume, strict reload and cached generation across a 512-token anchor
boundary. The recommended dense rank-32-query recipe has also completed its
32-GPU training schedule. These checks do not certify arbitrary-workload
bitwise resume, other hardware, language quality, every baseline's GPU
training, or a research vLLM backend.

The [model and training configuration](configs/hsd_gdn_0p8b_readq32_full32.json)
and [32-GPU guide](reference-full32.md) define the reproducible launch inputs.
The default reader is `tile_online`; the reader and backward source hashes
remain recorded in `SOURCE_MANIFEST.json`. No kernel computation is changed
by removing execution records.

```bash
python -m pytest tests/march/test_research_hsd.py tests/march/test_research_recipe.py -q
MARCH_TEST_RESEARCH=1 python -m pytest tests/march/test_research_cleanup.py -q
```

The optional dependencies must be installed for geometry/load checks. CPU
tests and dry-runs are not GPU performance evidence. FLA and FlashQLA source
licenses, notices and provenance are retained. Safin-1's operator receipt
does not cover these research kernels.
