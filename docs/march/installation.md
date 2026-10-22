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

The CUDA profiles use the public PyTorch `cu129` wheel index and the
versions in `requirements/march-constraints.txt`. Build requirements and
tested configurations are collected in [validation](validation.md).

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

The vLLM profile builds native extensions from this checkout and assigns
the fork version through `VLLM_VERSION_OVERRIDE`. Use a dedicated serving
environment when keeping training and serving dependencies separate.

```bash
# Set the target GPU compute capability; 9.0 is an example.
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
See [validation](validation.md) for tested GPU configurations.

The public source home is [AI45Lab/Safin-1](https://github.com/AI45Lab/Safin-1).
Use a released source revision or a reviewed source archive for installation.
Record that revision together with the resolved environment.

## Offline installation

A wheelhouse contains downloaded Python wheels and their dependencies.
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
starting. Prepare the wheelhouse and source caches before running the build.

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
