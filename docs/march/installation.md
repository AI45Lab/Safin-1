# Installation

Use Linux and Python 3.11 in a dedicated virtual environment. This fork retains
the `ms_swift` distribution and the `swift` / `megatron` commands, so installing
it replaces an upstream ms-swift installation in that environment. Clone or
extract the actual MARCH source checkout; installing upstream ms-swift or vLLM
alone does not provide Safin-1.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
```

Run the installer from the repository root. `DRY_RUN=1` prints commands without
installing packages or importing CUDA:

```bash
DRY_RUN=1 bash scripts/march/install.sh hf
bash scripts/march/install.sh hf
python scripts/march/check_environment.py --backend hf --strict-versions
```

The CUDA recipe explicitly selects the public PyTorch `cu129` wheel index to
match the recorded CUDA 12.9 runtime. Package constraints pin the observed core
versions. Fresh isolated HF/training installation, TE/vLLM native source
builds, both full-size one-step SFT/export/strict HF reload chains and TP1/DP1
Graph service checks passed within the [installation acceptance
record](installation-acceptance-20261017.json). The vLLM environment required the
recorded incremental FLA dependency repair. This is not a complete transitive
dependency lock or separate certification of a newly built container image;
other toolchains, hardware and larger workloads need their own validation.
The [official wheel index](https://download.pytorch.org/whl/cu129/torch/) lists
Python 3.11 Linux wheels for the pinned torch version.

CPU fixtures need no weight downloads:

```bash
bash scripts/march/install.sh cpu
PYTHONPATH="$PWD" python -m pytest tests/march -q
```

For training, provide a CUDA 12.9 development toolchain with `nvcc`, C++ compiler,
Ninja and enough RAM/disk for extension builds. The installer builds Transformer
Engine's PyTorch bindings and installs the local bridge before this fork.
It forces the bindings' source build, including in offline mode, so their wheel
builder does not probe GitHub for a precompiled package. CUDA/cuDNN headers come
from the installed NVIDIA wheels; the complete system CUDA toolkit supplies
`nvcc` and compiler headers such as `crt/host_defines.h`. The installer keeps
that SDK include directory on `CPATH` while compiling the bindings:

```bash
bash scripts/march/install.sh training
python scripts/march/check_environment.py --backend training --strict-versions
# Before reserving GPUs, exercise the actual processor/tokenizer loader on CPU:
CUDA_VISIBLE_DEVICES='' python scripts/march/check_environment.py \
  --backend training --strict-versions --model /path/to/safin1-checkpoint
MARCH_TEST_MEGATRON=1 PYTHONPATH="$PWD/third_party/mcore-bridge/src:$PWD" \
  python -m pytest tests/march/test_training_contract.py -q
```

The vLLM installer sets an explicit fork version with `VLLM_VERSION_OVERRIDE`
because a vendored source export has no upstream `.git` metadata. It also
disables upstream precompiled-wheel reuse for this source-build recipe.

vLLM native extensions must be built from this checkout. They are not shipped in
the public source archive. Prefer a separate environment for serving if its
transitive dependencies conflict with training:

```bash
# Set architectures for the target GPUs; 9.0 is the recorded H200/Hopper target.
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 bash scripts/march/install.sh vllm
python scripts/march/check_environment.py --backend vllm --strict-versions
```

For evaluation, install the vendored evaluator rather than an unrelated current
PyPI snapshot:

```bash
bash scripts/march/install.sh eval
python scripts/march/check_environment.py --backend eval --strict-versions
```

The installer runs `pip check` after installation and records the resolved
package list and core environment report under `.local/environment/`. Those
files are local only and are excluded from publication. Override
`ENVIRONMENT_REPORT_DIR` to keep them elsewhere.

The recipes download public dependencies. For offline use, prepare a compatible
wheelhouse first and set pip's normal offline/index options. Model weights,
tokenizers and evaluation datasets must also be provisioned independently.
See [validation](validation.md) for tested GPUs and incomplete coverage.

The public source home is [AI45Lab/Safin-1](https://github.com/AI45Lab/Safin-1).
Use a released source revision or a reviewed source archive for installation.
Record that revision together with the resolved environment.

## Offline installation

A wheelhouse is a directory containing downloaded Python wheels and all their
transitive dependencies. It is separate from a preinstalled Python environment.
Use a new virtual environment and point the installer at local artifacts:

```bash
WHEELHOUSE=/path/to/wheelhouse bash scripts/march/install.sh training
```

For vLLM source builds, additionally provide the CMake source caches for CUTLASS,
vLLM FlashAttention, FlashMLA, DeepGEMM, QuTLASS and Triton kernels. These are the
revisions pinned in the vendored `CMakeLists.txt` and `cmake/external_projects/`.
Set `FETCHCONTENT_BASE_DIR` to a populated cache, or set the source overrides
listed by the preflight tool:

```bash
python scripts/march/check_offline_build.py
WHEELHOUSE=/path/to/wheelhouse FETCHCONTENT_BASE_DIR=/path/to/cmake-cache \
  TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 bash scripts/march/install.sh vllm
```

With a wheelhouse, pip runs with `PIP_NO_INDEX=1`. The vLLM build also sets
`FETCHCONTENT_FULLY_DISCONNECTED=ON` and rejects missing source caches before
starting. Do not provision dependencies from a GPU validation allocation.
A missing wheel/source cache is an incomplete clean-build acceptance, even if
an existing training image passes the GPU regressions.

The eval installer includes IFEval Python extras. Prepare NLTK/dataset caches
explicitly before offline tasks; see [evaluation](evaluation.md).

The vLLM profile also installs `flash-linear-attention==0.5.0`: MARCH varlen
prefill imports its helpers in addition to the operations vendored inside vLLM.
The environment check requires both the FLA distribution and `fla` module before
serving.

The inherited Qwen3.5 training processor imports `qwen-vl-utils`, `decord` and
`torchvision` even for text examples. The training profile pins these dependencies
and installs torchvision from the same CUDA wheel index as PyTorch. Supplying
`--model` to the training environment check also loads that processor and
tokenizes a short string on CPU, without loading model weights.
