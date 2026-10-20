"""Compatibility import for the MARCH paged inference runtime."""

import sys
from march.inference import paged_history_read as _implementation

sys.modules[__name__] = _implementation
