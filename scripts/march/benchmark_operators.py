#!/usr/bin/env python3
"""Offline MARCH operator timings, peak allocation and numerical checks.

No model weights/datasets are needed. Timings include public Python wrappers;
they are operator microbenchmarks, not model training or serving throughput.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


class BackendUnavailable(RuntimeError):
    pass


def integer_list(value):
    result = [int(x) for x in value.split(",")]
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


@contextlib.contextmanager
def environment(**values):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class Benchmark:
    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.rows = []
        self.root = Path(__file__).resolve().parents[2]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU timings are substituted")
        self.device = torch.device("cuda")
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        props = torch.cuda.get_device_properties(0)
        packages = {}
        for name in ("torch", "triton", "flash-linear-attention", "tilelang", "megatron-core", "transformer-engine"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        sources = {}
        for directory in ("march/kernels", "march/inference", "third_party/mcore-bridge/src"):
            for path in sorted((self.root / directory).rglob("*.py")):
                sources[str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        revision = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True)
        self.metadata = dict(
            started_at=datetime.now(timezone.utc).isoformat(), revision=revision.stdout.strip(),
            gpu=props.name, compute_capability=list(torch.cuda.get_device_capability()),
            total_memory_gib=props.total_memory / 2**30, cuda_runtime=torch.version.cuda,
            packages=packages, dtype="bfloat16", state_dtype="float32", seed=args.seed,
            warmup=args.warmup, samples=args.samples, source_sha256=sources,
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            configuration=dict(lengths=args.lengths, heads=args.heads, head_dim=args.head_dim,
                               topk=args.topk, decode_batches=args.decode_batches,
                               decode_anchors=args.decode_anchors,
                               paged_persistent_memory=args.paged_persistent_memory,
                               graph_read=args.graph_read),
            methodology="CUDA events around complete warmed-up calls; no compile/warmup in timings. "
                        "Forward+backward includes forward and all input gradients. "
                        "Peak allocation includes fixture/reference tensors; incremental peak is reported separately. "
                        "The process owns one leased GPU; other GPUs on the node may run other jobs.",
        )
        self.save()
        print(json.dumps({"event": "started", "gpu": props.name, "revision": self.metadata["revision"]}), flush=True)

    def save(self):
        path = self.args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps({"metadata": self.metadata, "results": self.rows}, indent=2, allow_nan=False) + "\n")
        temp.replace(path)

    def record(self, row):
        self.rows.append(row)
        self.save()
        print(json.dumps(row, allow_nan=False), flush=True)

    def clean(self):
        gc.collect()
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()

    def measure(self, fn):
        t = self.torch
        for _ in range(self.args.warmup):
            fn()
        t.cuda.synchronize()
        t.cuda.reset_peak_memory_stats()
        allocated = t.cuda.memory_allocated()
        timings = []
        for _ in range(self.args.samples):
            start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        timings.sort()
        return dict(median_ms=statistics.median(timings), min_ms=timings[0], max_ms=timings[-1],
                    p20_ms=timings[int((len(timings) - 1) * .2)],
                    p80_ms=timings[int((len(timings) - 1) * .8)],
                    peak_allocated_mib=t.cuda.max_memory_allocated() / 2**20,
                    incremental_peak_mib=(t.cuda.max_memory_allocated() - allocated) / 2**20)

    def compare(self, actual, expected, atol, rtol):
        t = self.torch
        errors = []
        passed = True
        for value, reference in zip(actual, expected, strict=True):
            value, reference = value.detach().float(), reference.detach().float()
            finite = bool(t.isfinite(value).all() and t.isfinite(reference).all())
            delta = (value - reference).abs()
            errors.append(dict(max_abs=float(delta.max()) if finite else None,
                               rms=float(delta.square().mean().sqrt()) if finite else None,
                               finite=finite))
            passed &= finite and bool(t.allclose(value, reference, atol=atol, rtol=rtol))
        return dict(passed=passed, atol=atol, rtol=rtol, errors=errors)

    def run_backend(self, base, fn, inputs, reference_outputs, reference_grads, grad_outputs):
        t = self.torch
        row = dict(base)
        started = time.monotonic()
        try:
            outputs = fn()
            if outputs is None:
                raise BackendUnavailable("requested backend returned None; no fallback timing")
            outputs = (outputs,) if isinstance(outputs, t.Tensor) else tuple(outputs)
            checks = self.compare(outputs, reference_outputs, atol=2e-3, rtol=2e-2)
            grads = t.autograd.grad(outputs, inputs, grad_outputs=grad_outputs)
            gradient_checks = self.compare(grads, reference_grads, atol=6e-3, rtol=4e-2)
            del outputs, grads

            def forward():
                with t.no_grad():
                    value = fn()
                    if value is None:
                        raise BackendUnavailable("backend missed during measurement")

            def forward_backward():
                value = fn()
                value = (value,) if isinstance(value, t.Tensor) else tuple(value)
                t.autograd.grad(value, inputs, grad_outputs=grad_outputs)

            row.update(status="ok" if checks["passed"] and gradient_checks["passed"] else "numerical_failed",
                       outputs=checks, gradients=gradient_checks,
                       forward=self.measure(forward), forward_backward=self.measure(forward_backward))
            row["tokens_per_second_fb"] = base["length"] / (row["forward_backward"]["median_ms"] / 1000)
        except BackendUnavailable as exc:
            row.update(status="unsupported", reason=str(exc))
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(limit=4))
        row["setup_and_measure_seconds"] = time.monotonic() - started
        self.record(row)
        self.clean()

    def chunk(self):
        t = self.torch
        from march.kernels import march_chunk_gdn as chunk
        from march.kernels.chunk_reference import _run_segmented_packed_reference
        for length in self.args.lengths:
            if length % 512:
                raise ValueError("chunk benchmark lengths must be multiples of 512")
            for heads in self.args.heads:
                t.manual_seed(self.args.seed + length + heads * 100003)
                dim, anchors = self.args.head_dim, length // 512
                shape = (1, length, heads, dim)
                q, k = [t.nn.functional.normalize(t.randn(shape, device="cuda"), dim=-1).to(t.bfloat16)
                        for _ in range(2)]
                v = (t.randn(shape, device="cuda") * .1).to(t.bfloat16)
                beta = t.rand(1, length, heads, device="cuda") * .5
                g = -t.rand(1, length, heads, device="cuda") * .08
                aq = t.nn.functional.normalize(t.randn(1, anchors, heads, dim, device="cuda"), dim=-1).to(t.bfloat16)
                inputs = [x.detach().requires_grad_() for x in (q, k, v, beta, g, aq)]
                kwargs = dict(zip(("query", "key", "value", "beta", "g", "anchor_q"), inputs))
                kwargs.update(boundary_counts=t.arange(512, length + 1, 512, device="cuda"),
                              text_cu_seqlens=t.tensor([0, length], device="cuda"),
                              anchor_cu_seqlens=t.tensor([0, anchors], device="cuda"))
                reference = lambda kwargs=kwargs, dim=dim: _run_segmented_packed_reference(**kwargs, scale=dim**-.5)
                expected = reference()
                grad_outputs = tuple(t.randn_like(x) * (.01 if i == 2 else 1) for i, x in enumerate(expected))
                expected_grads = tuple(x.detach() for x in t.autograd.grad(expected, inputs, grad_outputs))
                expected = tuple(x.detach() for x in expected)
                variants = [
                    ("segmented_fla_reference", reference),
                    ("triton_chunk", lambda kwargs=kwargs: chunk.hsd_chunk_gated_delta_rule_with_anchor_states(**kwargs)),
                    ("flashqla_optimized", lambda kwargs=kwargs: chunk.hsd_flashqla_tile_gated_delta_rule_with_anchor_states(
                        **kwargs, backward_backend="optimized")),
                ]
                for backend, fn in variants:
                    self.run_backend(dict(group="chunk", backend=backend, length=length, heads=heads,
                                          head_dim=dim, anchors=anchors, packed=True),
                                     fn, inputs, expected, expected_grads, grad_outputs)
                del inputs, kwargs, expected, expected_grads, grad_outputs, variants, fn, reference
                self.clean()

    def history(self):
        t = self.torch
        from mcore_bridge.model.modules.march import MarchGatedDeltaNet
        layer = object.__new__(MarchGatedDeltaNet)
        t.nn.Module.__init__(layer)
        layer.hsd_use_route_scale = True
        layer.hsd_route_logit_scale = 1.0
        layer.hsd_route_qk_l2norm = False
        layer.hsd_read_topk = self.args.topk
        layer.hsd_route_head_dim = layer.key_head_dim = self.args.head_dim
        layer.persistent_memory_state = None
        for length in self.args.lengths:
            for heads in self.args.heads:
                t.manual_seed(self.args.seed + length + heads * 100003 + 1)
                dim, anchors = self.args.head_dim, max(1, length // 512)
                shape = (1, length, heads, dim)
                read_q, route_q = [t.nn.functional.normalize(t.randn(shape, device="cuda"), dim=-1).to(t.bfloat16)
                                   for _ in range(2)]
                states = t.randn(1, anchors, heads, dim, dim, device="cuda") * .1
                keys = t.nn.functional.normalize(t.randn(1, anchors, heads, dim, device="cuda"), dim=-1).to(t.bfloat16)
                null = t.randn(1, length, heads, 1, device="cuda") * .1
                inputs = [x.detach().requires_grad_() for x in (read_q, route_q, states, keys, null)]
                kwargs = dict(zip(("read_q", "router_q", "states", "anchor_k", "null_logits"), inputs))
                kwargs.update(boundary_counts=t.arange(512, anchors * 512 + 1, 512, device="cuda"), text_len=length)
                fn = lambda kwargs=kwargs: layer._history_read(**kwargs)
                layer.hsd_topk_backend = "reference"
                expected = fn()
                grad_out = (t.randn_like(expected),)
                expected_grads = tuple(x.detach() for x in t.autograd.grad(expected, inputs, grad_out))
                expected = (expected.detach(),)
                for backend, mode in [("reference", "grouped"), ("sparse_grouped", "grouped"), ("sparse_token", "token")]:
                    layer.hsd_topk_backend = "reference" if backend == "reference" else "sparse"
                    with environment(HSD_TOPK_EXECUTION_MODE=mode, HSD_TOPK_STATE_PRECISION_MODE="high_residual",
                                     HSD_TOPK_GLOBAL_READ_STRICT="1"):
                        self.run_backend(dict(group="training_history", backend=backend, length=length, heads=heads,
                                              head_dim=dim, anchors=anchors, topk=self.args.topk,
                                              includes="routing/topk/null gate/state read"),
                                         fn, inputs, expected, expected_grads, grad_out)
                del inputs, kwargs, expected, expected_grads, grad_out, fn
                self.clean()

    def paged(self):
        t = self.torch
        from march.inference import paged_history_read as p
        for batch in self.args.decode_batches:
            for anchors in self.args.decode_anchors:
                t.manual_seed(self.args.seed + batch * 100003 + anchors * 101 + 2)
                dim, heads, page_m = self.args.head_dim, 32, 16
                pages_per_state = math.ceil(anchors / page_m)
                cache = p.HSDAnchorPageCache(page_m=page_m)
                cache.preallocate(spec=p.HSDHistoryCacheSpec(
                    page_m=page_m, num_pages=batch * pages_per_state, heads=heads, value_dim=dim,
                    key_dim=dim, states_dtype=t.float32, keys_dtype=t.bfloat16), device=self.device)
                cache.preallocate_metadata(state_rows=batch, max_pages_per_state=pages_per_state, device=self.device)
                for index in range(batch):
                    cache.append(index, t.randn(1, anchors, heads, dim, dim, device="cuda") * .1,
                                 t.randn(1, anchors, heads, dim, device="cuda", dtype=t.bfloat16),
                                 (t.arange(anchors, device="cuda") +
                                  (0 if self.args.paged_persistent_memory else 1)) * 512)
                q = t.randn(batch, heads, dim, device="cuda", dtype=t.bfloat16)
                rq, null = t.randn_like(q), t.randn(batch, heads, device="cuda", dtype=t.bfloat16)
                indices = t.arange(batch, device="cuda")
                starts = t.full((batch,), (anchors + 1) * 512, device="cuda", dtype=t.long)
                kwargs = dict(cache=cache, read_q=q, router_q=rq, state_indices=indices, start_indices=starts,
                              route_scale=dim**-.5, read_scale=dim**-.5, null_logits=null,
                              persistent_memory_enabled=self.args.paged_persistent_memory)
                with t.no_grad():
                    expected = p.paged_batched_decode_history_read(**kwargs, backend="torch")
                    if expected is None:
                        raise RuntimeError("paged reference unavailable")
                    graph = static = stream = output = zero = scale = valid = None
                    graph_modes = ("split", "fused") if self.args.graph_read == "both" else (self.args.graph_read,)
                    backends = ["torch", "triton", "fused_triton", *["static_graph_" + mode for mode in graph_modes]]
                    for backend in backends:
                        base = dict(group="paged_decode", backend=backend, batch=batch, anchors=anchors,
                                    heads=heads, head_dim=dim, persistent_memory=self.args.paged_persistent_memory)
                        try:
                            if backend.startswith("static_graph_"):
                                if graph is not None:
                                    graph.reset()
                                output = t.empty_like(q)
                                zero = t.zeros_like(q)
                                scale = t.tensor(1., device="cuda")
                                valid = t.ones(batch, dtype=t.bool, device="cuda")

                                def static(kwargs=kwargs, output=output, zero=zero, valid=valid, scale=scale):
                                    output.copy_(zero)
                                    return p.paged_static_decode_history_read(
                                        **kwargs, valid_mask=valid, base_out=output, history_scale=scale)

                                with environment(VLLM_HSD_GRAPH_V3_SPLIT_HISTORY_READ=
                                                 "1" if backend.endswith("_split") else "0"):
                                    stream = t.cuda.Stream()
                                    stream.wait_stream(t.cuda.current_stream())
                                    with t.cuda.stream(stream):
                                        for _ in range(3):
                                            static()
                                    t.cuda.current_stream().wait_stream(stream)
                                    t.cuda.synchronize()
                                    graph = t.cuda.CUDAGraph()
                                    with t.cuda.graph(graph):
                                        static()
                                fn = graph.replay
                                fn()
                                actual = output
                            else:
                                fn = lambda kwargs=kwargs, backend=backend: p.paged_batched_decode_history_read(**kwargs, backend=backend,
                                                                                 validate_fused=False)
                                actual = fn()
                            if actual is None:
                                raise BackendUnavailable("requested backend returned None")
                            check = self.compare((actual,), (expected,), atol=2e-2, rtol=2e-2)
                            base.update(status="ok" if check["passed"] else "numerical_failed",
                                        outputs=check, forward=self.measure(fn))
                            base["requests_per_second"] = batch / (base["forward"]["median_ms"] / 1000)
                        except BackendUnavailable as exc:
                            base.update(status="unsupported", reason=str(exc))
                        except Exception as exc:
                            base.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                                        traceback=traceback.format_exc(limit=4))
                        self.record(base)
                    if graph is not None:
                        graph.reset()
                    del graph, static, stream, output, zero, scale, valid
                    del cache, kwargs, expected, actual, fn, q, rq, null
                self.clean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="chunk,history,paged")
    parser.add_argument("--lengths", type=integer_list, default=[512, 2048, 8192])
    parser.add_argument("--heads", type=integer_list, default=[16, 32])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--decode-batches", type=integer_list, default=[1, 16, 32])
    parser.add_argument("--decode-anchors", type=integer_list, default=[16, 64])
    parser.add_argument("--paged-persistent-memory", action="store_true")
    parser.add_argument("--graph-read", choices=("split", "fused", "both"), default="split",
                        help="split matches the public serving profile; both compares capture paths")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    groups = args.groups.split(",")
    if set(groups) - {"chunk", "history", "paged"}:
        parser.error("groups must be chunk,history,paged")
    if args.samples < 3 or args.warmup < 1 or args.head_dim <= 0 or args.topk not in (1, 2, 4, 8, 16):
        parser.error("positive dimensions, warmup>=1, samples>=3 and topk in 1/2/4/8/16 required")
    if "chunk" in groups and any(length % 512 for length in args.lengths):
        parser.error("chunk benchmark lengths must be multiples of 512")
    if args.dry_run:
        print(json.dumps({"groups": groups, "lengths": args.lengths, "heads": args.heads,
                          "head_dim": args.head_dim, "topk": args.topk,
                          "decode_batches": args.decode_batches, "decode_anchors": args.decode_anchors,
                          "paged_persistent_memory": args.paged_persistent_memory,
                          "graph_read": args.graph_read,
                          "requires_gpu": True, "requires_models_or_datasets": False}, indent=2))
        return
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "third_party/mcore-bridge/src"), str(root)]
    with environment(HSD_DEBUG_STATS="0", HSD_DEBUG_VERBOSE_STATS="0", HSD_PROFILE_TIMING="0",
                     HSD_GRAD_DEBUG_ACTIVATIONS="0", HSD_PRODUCTION_HEALTH_STATS="0"):
        bench = Benchmark(args)
        for group in groups:
            try:
                getattr(bench, group)()
            except Exception as exc:
                bench.record(dict(group=group, status="group_error", reason=f"{type(exc).__name__}: {exc}",
                                  traceback=traceback.format_exc(limit=5)))
        bench.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        bench.save()
        failed = sum(r["status"] in ("error", "group_error", "numerical_failed") for r in bench.rows)
        print(json.dumps({"event": "complete", "cases": len(bench.rows), "failed": failed}), flush=True)
        return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
