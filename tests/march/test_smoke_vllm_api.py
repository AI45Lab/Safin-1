import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "smoke_vllm_api", ROOT / "scripts/march/smoke_vllm_api.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_endpoint_keeps_base_path_and_normalizes_slashes():
    assert smoke.endpoint("http://127.0.0.1:8000/api", "/health") == "http://127.0.0.1:8000/api/health"


def test_validate_completion_returns_reviewable_fields():
    result = smoke.validate_completion({
        "id": "chatcmpl-1",
        "model": "Safin-1",
        "choices": [{
            "message": {"role": "assistant", "content": "MARCH uses memory anchors."},
            "finish_reason": "stop",
        }],
    })
    assert result == {
        "id": "chatcmpl-1",
        "model": "Safin-1",
        "finish_reason": "stop",
        "content": "MARCH uses memory anchors.",
    }


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}])
def test_validate_completion_rejects_incomplete_response(payload):
    with pytest.raises(ValueError):
        smoke.validate_completion(payload)
