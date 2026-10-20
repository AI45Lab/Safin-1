# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang


def profile(func, inputs, wait: int = 50, warmup: int = 50, rep: int = 100):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=rep),
    ) as prof:
        for idx in range(wait + warmup + rep):
            func(*inputs)
            prof.step()

    cuda_events = [
        evt for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.device_time > 0
    ]

    kernels_per_iter = None
    for n_kernels in range(1, len(cuda_events) + 1):
        if len(cuda_events) % n_kernels == 0:
            chunk = [e.name for e in cuda_events[:n_kernels]]
            ok = True
            for i in range(n_kernels, len(cuda_events), n_kernels):
                if [e.name for e in cuda_events[i:i + n_kernels]] != chunk:
                    ok = False
                    break
            if ok:
                kernels_per_iter = n_kernels
                break

    if kernels_per_iter is None:
        kernels_per_iter = len(cuda_events) // rep if rep > 0 else len(cuda_events)

    if kernels_per_iter == 0:
        result = {}
        result["total"] = tilelang.profiler.do_bench(
            lambda: func(*inputs), warmup=warmup, rep=rep
        )
        return result

    num_iters = len(cuda_events) // kernels_per_iter
    sums = {}
    order = []
    for i in range(kernels_per_iter):
        name = cuda_events[i].name
        count = sum(1 for j in range(i) if cuda_events[j].name == name)
        key = f"{name}#{count}" if count > 0 else name
        order.append(key)
        sums[key] = 0.0

    for it in range(num_iters):
        base = it * kernels_per_iter
        for i, key in enumerate(order):
            sums[key] += cuda_events[base + i].device_time * 1e-3

    result = {k: sums[k] / num_iters for k in order}
    result["total"] = tilelang.profiler.do_bench(
        lambda: func(*inputs), warmup=warmup, rep=rep
    )
    return result
