#!/usr/bin/env python3
"""Wait for a Safin-1 vLLM server and exercise its chat-completions API."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def endpoint(base_url, path):
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def wait_until_ready(base_url, wait_seconds, request_timeout):
    health = endpoint(base_url, "/health")
    deadline = time.monotonic() + wait_seconds
    last_error = None
    while True:
        try:
            with urlopen(health, timeout=request_timeout) as response:
                if 200 <= response.status < 300:
                    return
                last_error = RuntimeError(f"health endpoint returned HTTP {response.status}")
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
        if time.monotonic() >= deadline:
            raise RuntimeError(f"vLLM server was not ready after {wait_seconds:g}s: {last_error}")
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def validate_completion(payload):
    if not isinstance(payload, dict):
        raise ValueError("chat completion response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion response has no choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("first chat completion choice has no text content")
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "finish_reason": first.get("finish_reason"),
        "content": content,
    }


def request_completion(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }, ensure_ascii=False).encode()
    request = Request(
        endpoint(base_url, "/v1/chat/completions"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion returned HTTP {error.code}: {detail}") from error
    return validate_completion(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Safin-1")
    parser.add_argument("--prompt", default="用一句话介绍 MARCH。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--wait-seconds", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.wait_seconds < 0 or args.timeout <= 0:
        parser.error("timeouts must be non-negative and request timeout must be positive")
    wait_until_ready(args.base_url, args.wait_seconds, min(args.timeout, 10.0))
    report = request_completion(
        args.base_url, args.model, args.prompt, args.max_tokens, args.timeout)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
