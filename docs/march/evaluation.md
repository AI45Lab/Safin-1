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

The vendored harness also contains upstream task definitions. Prepare each
task's datasets and optional extras before running. IFEval checks its packages
and English punkt files at startup and reports missing local resources.

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
also use the local cache. `LIMIT=16` selects a small first run; remove `LIMIT`
for the full task. Set `TASKS=ifeval,mmlu_pro` to evaluate both reference tasks.

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

## Result records

Keep task and dataset versions, seed, chat template, thinking settings,
backend profile and generation limits with each result. Use the same inputs
and settings when comparing HF and vLLM. Runtime precision notes are in
[validation](validation.md#numerical-limits).
