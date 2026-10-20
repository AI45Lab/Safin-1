# Training data and checkpoint ancestry

The [sanitized inventory](training-data-20261017.json) was reconstructed from the
four original SFT checkpoints and their CPT initializations. All four record the
same ten SFT file specifications. The corpus snapshot, filtering, actual unique
sample counts and measured non-padding tokens are not established by these args.

## CPT

The initialization records identify an Intern-S2-derived internal text corpus,
CPT step 5961, global batch 256 and packing length 32768. The maintainer supplied
no public provenance or permission record. No public dataset ID or dataset
license is assigned here. Applied preprocessing and filtering remain unverified.
Publishing a recipe name does not establish permission to release data or weights.

The packed-position upper bound is 50,004,492,288; this is not a measured text-token
count. Upstream Qwen model-family attribution is recorded, while the exact
training-time upstream commit remains unknown.

## SFT

Recorded filenames identify [NVIDIA Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data),
whose data card specifies the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
The observed upstream card revision is recorded separately from the unknown
training snapshot. Matching names do not prove local content identity.

| Relative upstream filename | Recorded sampling request |
|---|---:|
| `math/math_notool.jsonl` | 780000 |
| `math/math_proof.jsonl` | 195000 |
| `science/science.jsonl` | 1245000 |
| `chat/chat_part_1.jsonl` | 846000 |
| `chat/chat_part_2.jsonl` | 846000 |
| `chat/chat_part_3.jsonl` | 846000 |
| `chat/chat_part_4.jsonl` | 846000 |
| `instruction_following/instruction_following.jsonl` | no suffix |
| `safety/safety.jsonl` | no suffix |
| `swe/swe_agentless.jsonl` | 215000 |

`#N` records the loader sampling request, not an observed number of distinct
examples. A missing suffix is not a zero count. All recipes use seed/data seed 42,
packing length 32768 and no validation split. The 4B models save step 3600 with
global batch 256; 35B-A3B saves step 7200 with global batch 128. Each has the same
packed-position upper bound of 30,198,988,800. Padding, truncation, resampling,
and repeated tokens prevent treating this as measured unique training text.

## Publication boundary

Weights are assigned Apache-2.0 by the maintainer; training data retains its
own terms. No corpus contents, internal paths or original args.json are included
in the upload candidates. CPT provenance/permissions and a reviewed snapshot and
preprocessing manifest remain undocumented. The maintainer chose to disclose
these gaps while sharing the weights; no dataset contents are redistributed.
