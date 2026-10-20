"""Compatibility import for the MARCH packed inference producer."""

import sys
from march.inference import varlen_segments as _implementation

sys.modules[__name__] = _implementation
