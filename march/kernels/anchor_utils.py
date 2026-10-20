"""Compatibility import for MARCH training anchor batches."""

import sys
from march.training import anchor_batch as _implementation

sys.modules[__name__] = _implementation
