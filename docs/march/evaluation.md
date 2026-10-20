# Safin-1 evaluation

Use the public entrypoints to keep evaluation settings consistent with serving.
HF and vLLM both require migrated Safin-1 checkpoints and use seed 123, greedy
task defaults and thinking disabled by default. The vLLM wrapper uses the same
`march/inference/vllm_profile.py` profile as `serve_vllm.py`: Graph V3 by default,
Triton GDN, prefix caching disabled, synchronous scheduling and per-family MoE
controls. `MARCH_INFERENCE_PROFILE=eager` selects the complete fallback.

## Install and prepare local resources

```bash
bash scripts/march/install.sh eval
python scripts/march/check_environment.py --backend eval --strict-versions
```

The evaluator installation includes its IFEval extra (`langdetect`,
`immutabledict` and `nltk>=3.9.1`). Prepare NLTK data outside a GPU/offline task:

```bash
python -m nltk.downloader -d /path/to/nltk_data punkt_tab
export NLTK_DATA=/path/to/nltk_data
```

Before running, prepare the task dataset in the local datasets cache as well.
The reference task definitions are:

| Task | Dataset identifier | Extra local requirements |
|---|---|---|
| `ifeval` | `google/IFEval` | Complete `punkt_tab/english` files; installed `[ifeval]` extra |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | Dataset cache |

These are the public reference task choices, not a claim of completed full
benchmark accuracy acceptance. The vendored harness retains the upstream task
catalog; other tasks may require their own extras, resources and validation.
The wrappers validate IFEval packages and all four English punkt files before
allocating the model. Importing IFEval never downloads resources, including
when `OFFLINE=0`; missing resources produce preparation instructions.

## HF and vLLM commands

```bash
NLTK_DATA=/path/to/nltk_data MODEL=/path/to/Safin-1-4B \
TASKS=ifeval OUTPUT_DIR=/path/to/results/hf LIMIT=16 \
bash scripts/march/eval_hf.sh

NLTK_DATA=/path/to/nltk_data MODEL=/path/to/Safin-1-4B \
TASKS=ifeval OUTPUT_DIR=/path/to/results/vllm LIMIT=16 \
bash scripts/march/eval_vllm.sh

MODEL=/path/to/Safin-1-35B-A3B TASKS=mmlu_pro \
OUTPUT_DIR=/path/to/results/35b TP_SIZE=4 MAX_NUM_SEQS=32 LIMIT=16 \
bash scripts/march/eval_vllm.sh
```

The wrappers default to `OFFLINE=1`, so model/configuration and dataset lookup
also remain offline. `LIMIT=16` is a usage smoke, not a final benchmark result.
`TP_SIZE=4` is an interface example; production TP/DP acceptance remains a
separate gate. Set `TASKS=ifeval,mmlu_pro` for both reference tasks.

Use `DRY_RUN=1` to inspect the command, structured model arguments and kernel
controls without importing vLLM or loading weights. Checkpoint paths containing
spaces or commas are preserved through the harness JSON argument parser.
Additional evaluator options, such as `--gen_kwargs '{"max_gen_toks":64}'`, are
forwarded; model/profile/output arguments are managed by the wrapper.

Thinking is explicit:

```bash
MODEL=/path/to/Safin-1-4B TASKS=ifeval OUTPUT_DIR=/path/to/results/thinking \
NLTK_DATA=/path/to/nltk_data ENABLE_THINKING=true THINK_END_TOKEN='</think>' \
bash scripts/march/eval_vllm.sh
```

For direct vLLM adapter users, an explicit `enable_thinking` constructor value wins
over the corresponding chat-template setting. If absent, the template setting
is retained, including `False`; the caller's dictionary is not modified.
Enabling thinking requires `think_end_token` and is checked before engine
allocation. Thinking and non-thinking metrics must be reported separately.

## Validation boundary

The [current integration record](validation.md#inference) includes
completed HF and vLLM fixture evaluations on the 4B SFT export, with two effective rows per task.
Full-model cross-process numerical checks remain conditional, including an
unchanged-source 4B control. The small evaluation scores do not establish
backend equivalence or benchmark accuracy.

Small synthetic evaluation smokes test task loading, prompt handling,
generation, metric execution and result writing. They do not estimate model
quality. Preserve software/source fingerprints, seeds, templates, backend
profiles and generation limits with full benchmark results.

The existing [Graph/eager numerical diagnosis](validation.md#numerical-limits)
remains applicable. Sharing a profile fixes configuration drift between serving
and evaluation; it does not establish HF/vLLM or Graph/eager numerical parity.
