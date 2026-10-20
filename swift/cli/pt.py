# Copyright (c) ModelScope Contributors. All rights reserved.

import os
from pathlib import Path


def configure_rank_private_triton_cache() -> None:
    """Avoid DeepSpeed Triton autotune races on shared cache directories."""
    disable = os.environ.get('SWIFT_DISABLE_RANK_PRIVATE_TRITON_CACHE', '').lower()
    if disable in {'1', 'true', 'yes'}:
        return

    base = (os.environ.get('MARCH_TRITON_CACHE_BASE')
            or os.environ.get('LOGLINEAR_TRITON_CACHE_BASE')  # Legacy compatibility.
            or os.environ.get('TRITON_CACHE_DIR'))
    if not base:
        return

    local_rank = os.environ.get('LOCAL_RANK')
    global_rank = os.environ.get('RANK')
    node_rank = os.environ.get('NODE_RANK') or os.environ.get('GROUP_RANK')
    if node_rank is not None and local_rank is not None:
        rank_suffix = f'node_{node_rank}_local_{local_rank}'
    else:
        rank_suffix = f'rank_{global_rank or local_rank or "0"}'

    base_path = Path(base)
    cache_dir = base_path if base_path.name == rank_suffix else base_path / rank_suffix
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(cache_dir)

if __name__ == '__main__':
    configure_rank_private_triton_cache()
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import pretrain_main
    pretrain_main()
