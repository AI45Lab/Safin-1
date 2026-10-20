"""Upstream model-download integrations are opt-in and honor assigned GPUs."""
import os

collect_ignore_glob = [] if os.getenv("MCORE_BRIDGE_RUN_UPSTREAM_TESTS") == "1" else ["test_llm.py", "test_mllm.py"]
