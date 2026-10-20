import hashlib
import json
import logging
import os
import re
import shutil
import time
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]


def get_num_samples(default: int = 500) -> int:
    value = os.environ.get("RULER_NUM_SAMPLES")
    if value is None:
        return default
    num_samples = int(value)
    if num_samples <= 0:
        raise ValueError("RULER_NUM_SAMPLES must be a positive integer.")
    return num_samples


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)




def _truthy_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _cache_root() -> Path:
    root = os.environ.get("RULER_DATASET_CACHE_DIR")
    if root is None or root.strip() == "":
        root = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "ruler_dataset_cache",
        )
    return Path(root)


def _tokenizer_fingerprint(pretrained) -> str:
    raw = getattr(pretrained, "name_or_path", pretrained)
    raw = str(raw)
    path = Path(raw)
    hasher = hashlib.sha256()
    found_files = False
    if path.exists():
        candidates = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "tokenizer.model",
        ]
        for name in candidates:
            file_path = path / name if path.is_dir() else None
            if file_path is not None and file_path.exists():
                found_files = True
                hasher.update(name.encode("utf-8"))
                with file_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
    if not found_files:
        hasher.update(raw.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _cache_path(task_name: str, seq_lengths: list[int], num_samples: int, pretrained) -> Path:
    payload = {
        "version": 1,
        "task": task_name,
        "seq_lengths": [int(x) for x in seq_lengths],
        "num_samples": int(num_samples),
        "tokenizer": _tokenizer_fingerprint(pretrained),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return _cache_root() / task_name / digest


def get_or_build_dataset_cache(
    task_name: str,
    seq_lengths: list[int],
    pretrained,
    build_rows: Callable[[], list[dict]],
) -> dict[str, datasets.Dataset]:
    """Build or reuse cached RULER synthetic datasets.

    The cache is shared across eval output directories and keyed by task, lengths,
    sample count, and tokenizer files. Set RULER_DATASET_CACHE=0 to disable.
    """

    if not _truthy_env("RULER_DATASET_CACHE", default=True):
        return {"test": datasets.Dataset.from_list(build_rows(), split=datasets.Split.TEST)}

    num_samples = get_num_samples()
    path = _cache_path(task_name, seq_lengths, num_samples, pretrained)
    if path.exists():
        eval_logger.info("Loading RULER dataset cache: %s", path)
        return {"test": datasets.load_from_disk(str(path))}

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if path.exists():
                eval_logger.info("Loading RULER dataset cache: %s", path)
                return {"test": datasets.load_from_disk(str(path))}
            eval_logger.info("Waiting for RULER dataset cache lock: %s", lock_path)
            time.sleep(5)

    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        if path.exists():
            return {"test": datasets.load_from_disk(str(path))}
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        eval_logger.info("Building RULER dataset cache: %s", path)
        dataset = datasets.Dataset.from_list(build_rows(), split=datasets.Split.TEST)
        dataset.save_to_disk(str(tmp_path))
        tmp_path.rename(path)
        return {"test": dataset}
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
