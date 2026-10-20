"""Opt-in acceptance plugin; snapshots state at the actual MCore save/load boundary.

Loaded only by smoke_training_resume.py through --external_plugins. Optimizer
moments/master parameters are sampled; RNG tensors are fingerprinted in full.
The separate checker compares every final HF weight tensor.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def fingerprint(value, *, sample=False):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().reshape(-1)
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            if sample and tensor.numel() > 96:
                middle = tensor.numel() // 2
                tensor = torch.cat((tensor[:32], tensor[middle:middle + 32], tensor[-32:]))
            digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def record(event, args, models, optimizer, scheduler, iteration):
    import random
    from megatron.core import tensor_parallel

    if optimizer is None or scheduler is None:
        raise RuntimeError("Resume acceptance requires an optimizer and scheduler")
    children = getattr(optimizer, "chained_optimizers", [optimizer])
    state = []
    for child in children:
        inner = child.optimizer
        state.append({"adam": inner.state_dict(),
                      "masters": [[param.detach() for param in group["params"]]
                                  for group in inner.param_groups]})
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
           "tracker": tensor_parallel.get_cuda_rng_tracker().get_states()}
    row = {"event": event, "rank": torch.distributed.get_rank(), "iteration": iteration,
           "consumed_train_samples": args.consumed_train_samples,
           "optimizer_sample_sha256": fingerprint(state, sample=True),
           "rng_sha256": fingerprint(rng), "scheduler": scheduler.state_dict()}
    target = Path(os.environ["MARCH_RESUME_PROBE_DIR"])
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"rank-{row['rank']}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def install():
    from functools import wraps
    import swift.megatron.utils as public_utils
    import swift.megatron.utils.megatron_lm_utils as checkpoint_utils
    import swift.megatron.trainers.base as trainer_base

    original_save = checkpoint_utils.save_mcore_checkpoint
    original_load = checkpoint_utils.load_mcore_checkpoint

    @wraps(original_save)
    def save(args, models, optimizer=None, opt_param_scheduler=None, *positional, **kwargs):
        record("before_save", args, models, optimizer, opt_param_scheduler, kwargs["iteration"])
        return original_save(args, models, optimizer, opt_param_scheduler, *positional, **kwargs)

    @wraps(original_load)
    def load(args, models, optimizer=None, opt_param_scheduler=None, *positional, **kwargs):
        result = original_load(args, models, optimizer, opt_param_scheduler, *positional, **kwargs)
        iteration = result[0] if isinstance(result, tuple) else result
        record("after_load", args, models, optimizer, opt_param_scheduler, iteration)
        return result

    # BaseMegatronTrainer imports these functions directly. Patch both the
    # definition and its explicit public references, without changing kernels.
    for module in (checkpoint_utils, public_utils, trainer_base):
        module.save_mcore_checkpoint = save
        module.load_mcore_checkpoint = load


if os.environ.get("MARCH_RESUME_PROBE_DIR"):
    install()
