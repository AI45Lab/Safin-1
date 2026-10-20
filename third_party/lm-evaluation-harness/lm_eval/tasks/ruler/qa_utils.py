# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import hashlib
import itertools  # noqa: I001
import json
import os
import random
from urllib.parse import urlparse
from functools import cache

import datasets
import requests
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import (
    DEFAULT_SEQ_LENGTHS,
    get_num_samples,
    get_or_build_dataset_cache,
    get_tokenizer,
)

CONFIG = {
    "tokens_to_generate": 32,
    "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
    "answer_prefix": """Answer:""",
}
SEED = 42
TEMPLATE = CONFIG["template"]
DOCUMENT_PROMPT = "Document {i}:\n{document}"


@cache
def download_json(url) -> dict:
    source_dir = os.environ.get("RULER_SOURCE_DIR")
    if source_dir:
        local_name = None
        if "SQuAD" in url or "squad" in url or "dev-v2.0.json" in url:
            local_name = "squad.json"
        elif "hotpot" in url or "hotpotqa" in url:
            local_name = "hotpotqa.json"
        if local_name:
            local_path = os.path.join(source_dir, local_name)
            if os.path.exists(local_path):
                with open(local_path, encoding="utf-8") as f:
                    return json.load(f)

    cache_dir = os.environ.get("RULER_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "ruler",
        )
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path) or "data.json"
    cache_name = f"{hashlib.sha256(url.encode()).hexdigest()[:12]}-{basename}"
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, cache_path)
    return data


@cache
def read_squad(
    url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
) -> tuple[list[dict], list[str]]:
    data = download_json(url)
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs


@cache
def read_hotpotqa(
    url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
) -> tuple[list[dict], list[str]]:
    data = download_json(url)
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q: str = qas[index]["query"]
    curr_a: list[str] = qas[index]["outputs"]
    curr_docs: list[int] = qas[index]["context"]
    curr_more: list[int] = qas[index].get("more_context", [])
    if num_docs <= len(curr_docs):
        all_docs = [docs[idx] for idx in curr_docs]
    elif num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(SEED).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a


def generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = 500,
    tokens_to_generate: int = 32,
    pre_samples: int = 0,
    incremental: int = 10,
    remove_newline_tab=False,
) -> list[dict]:
    write_jsons = []

    # Find the perfect num_docs
    num_docs = incremental

    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(0, num_docs, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(tokenizer(input_text + f" {answer}").input_ids)
        # print(
        #     f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
        # )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs = max(1, num_docs - incremental)
            break

        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break
    # print("Number of documents:", num_docs)

    if len(qas) == 0:
        raise ValueError("No QA samples are available for RULER QA generation.")

    # Generate samples. Some short contexts (for example 1k) cannot fit a
    # particular QA item even after reducing to the minimum document count, so
    # skip to the next QA item instead of retrying forever.
    cursor = pre_samples
    max_attempts_per_sample = max(len(qas), num_samples * 20)
    for index in tqdm(
        range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"
    ):
        attempts = 0
        while attempts < max_attempts_per_sample:
            qa_index = cursor % len(qas)
            used_docs = num_docs
            while True:
                try:
                    input_text, answer = generate_input_output(
                        qa_index, used_docs, qas=qas, docs=docs
                    )
                    length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                    assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                    cursor += 1
                    break
                except Exception:  # noqa: BLE001
                    if used_docs > 1:
                        used_docs = max(1, used_docs - incremental)
                        continue
                    cursor += 1
                    attempts += 1
                    input_text = None
                    break
            if input_text is not None:
                break
        else:
            raise RuntimeError(
                "Unable to build a RULER QA sample that fits "
                f"max_seq_length={max_seq_length} after {attempts} attempts. "
                "Use a larger max_seq_length or fewer/shorter QA source examples."
            )

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": "Answer:",
        }
        write_jsons.append(formatted_output)

    return write_jsons


def get_dataset(pretrained, docs, qas, max_seq_length=None, **kwargs) -> list[dict]:
    tokenizer = get_tokenizer(pretrained)
    write_jsons = generate_samples(
        tokenizer=tokenizer,
        docs=docs,
        qas=qas,
        num_samples=get_num_samples(),
        tokens_to_generate=32,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def get_qa_dataset(ds, **kwargs) -> dict[str, datasets.Dataset]:
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    if ds == "squad":
        qas, docs = read_squad()
    else:
        qas, docs = read_hotpotqa()
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)

    def build_rows():
        df = (
            get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq)
            for seq in seq_lengths
        )
        return list(itertools.chain.from_iterable(df))

    return get_or_build_dataset_cache(f"ruler_qa_{ds}", seq_lengths, pretrained, build_rows)


def get_squad(**kwargs):
    return get_qa_dataset("squad", **kwargs)


def get_hotpotqa(**kwargs):
    return get_qa_dataset("hotpotqa", **kwargs)
