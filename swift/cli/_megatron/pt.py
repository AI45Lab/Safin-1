# Copyright (c) ModelScope Contributors. All rights reserved.
import os


def _configure_per_rank_compile_cache() -> None:
    cache_root = os.getenv('SWIFT_TORCHRUN_PER_RANK_CACHE_ROOT')
    if not cache_root:
        return
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is None or not local_rank.isdigit():
        raise RuntimeError(
            'SWIFT_TORCHRUN_PER_RANK_CACHE_ROOT requires torchrun LOCAL_RANK.')
    rank_root = os.path.join(cache_root, f'r{local_rank}')
    if not os.path.abspath(rank_root).startswith('/tmp/'):
        raise RuntimeError(
            'Per-rank compilation caches must remain below /tmp, got '
            f'{rank_root!r}.')
    paths = {
        'TMPDIR': 'tmp',
        'TRITON_CACHE_DIR': 'triton',
        'TILELANG_CACHE_DIR': 'tilelang',
        'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
        'TORCH_EXTENSIONS_DIR': 'torch_extensions',
        'FLASHINFER_WORKSPACE_BASE': 'flashinfer',
        'XDG_CACHE_HOME': 'xdg',
    }
    for env_name, leaf in paths.items():
        path = os.path.join(rank_root, leaf)
        os.makedirs(path, exist_ok=True)
        os.environ[env_name] = path
    if len(os.environ['TMPDIR']) > 64:
        raise RuntimeError(
            'Per-rank TMPDIR is too long for Python multiprocessing AF_UNIX '
            f'sockets: {os.environ["TMPDIR"]}')


if __name__ == '__main__':
    _configure_per_rank_compile_cache()
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from swift.megatron import megatron_pretrain_main
    megatron_pretrain_main()
