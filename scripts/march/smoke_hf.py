#!/usr/bin/env python3
"""Train/save/reload a tiny Safin-1 fixture using real PyTorch reference ops.

This checks the HF model and remote-code export, not Megatron conversion or
production-size checkpoints. No weights or datasets are downloaded.
"""
import argparse
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


@contextmanager
def reference_kernels():
    # Transformers chooses installed CUDA extensions even on CPU. Select its
    # own differentiable reference implementations for this fixture only.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as dense
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
    with ExitStack() as stack:
        for module in (dense, moe):
            for name in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
                         "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
                stack.enter_context(patch.object(module, name, None))
        yield


@contextmanager
def remote_registry():
    # Explicit local AutoConfig registrations override trust_remote_code in
    # Transformers 5.5. Isolate them so this check really loads exported code.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    retained = {key: value for key, value in CONFIG_MAPPING._extra_content.items()
                if not key.startswith("safin1_")}
    with patch.dict(CONFIG_MAPPING._extra_content, retained, clear=True):
        yield


def tiny_config(variant, chunk_size=512):
    from march.safin1.configuration_safin1 import Safin1Config
    from march.safin1.configuration_safin1_moe import Safin1MoeConfig
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                linear_key_head_dim=16, linear_value_head_dim=16,
                layer_types=["linear_attention", "full_attention"],
                max_position_embeddings=2048, anchor_chunk_size=chunk_size,
                anchor_token_num=1, anchor_bos_token_id=1,
                hsd_read_block_size=chunk_size,
                hsd_global_read_layers="1-2", hsd_read_topk=4, hsd_null_read_gate=True,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                 "partial_rotary_factor": 1., "mrope_section": [2, 3, 3]})
    if variant == "safin1_35b_a3b":
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                    shared_expert_intermediate_size=32)
        text.pop("intermediate_size")
    cls = Safin1Config if variant == "safin1_4b" else Safin1MoeConfig
    config = cls(text_config=text, vision_config=dict(depth=1, hidden_size=32,
                 intermediate_size=64, num_heads=2, out_hidden_size=32,
                 num_position_embeddings=16, patch_size=2, temporal_patch_size=1),
                 image_token_id=60, video_token_id=61, vision_start_token_id=62,
                 vision_end_token_id=63)
    config._attn_implementation = "eager"
    return config


def run_variant(variant, output, chunk_size=512):
    import torch
    from transformers import AutoModelForCausalLM
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    from march.safin1.config_utils import normalize_safin1_config, copy_safin1_remote_code

    torch.manual_seed(123)
    torch.set_num_threads(2)
    with reference_kernels(), remote_registry():
        cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
        model = cls(tiny_config(variant, chunk_size))
        ids = torch.randint(3, 59, (1, chunk_size + 2))
        model.train()
        embedding = model.model.language_model.anchor_embedding.weight
        before = embedding.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        assert out.logits.shape == (1, ids.shape[1], 64), "anchor tokens leaked into logits"
        assert torch.isfinite(out.loss), "non-finite training loss"
        out.loss.backward()
        assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
        assert embedding.grad.abs().sum() > 0, "anchor embedding has no learning signal"
        optimizer.step()
        assert not torch.equal(before, embedding), "training did not update anchor embedding"
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=ids, use_cache=False).logits
        model.save_pretrained(output, safe_serialization=True)
        path = Path(output, "config.json")
        saved = normalize_safin1_config(json.loads(path.read_text()))
        path.write_text(json.dumps(saved, indent=2) + "\n")
        copy_safin1_remote_code(output, variant)
        loaded, info = AutoModelForCausalLM.from_pretrained(
            output, trust_remote_code=True, local_files_only=True,
            attn_implementation="eager", output_loading_info=True)
        assert not info.get("missing_keys"), info
        assert not info.get("unexpected_keys"), info
        assert not info.get("mismatched_keys"), info
        assert loaded.config.model_type == variant
        assert type(loaded).__module__.startswith("transformers_modules."), "local registration masked remote loading"
        assert loaded.config.text_config.anchor_chunk_size == chunk_size
        loaded.eval()
        errors = {}
        with torch.no_grad():
            actual = loaded(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            # Independently prefill at both sides of the anchor boundary.
            for prefix in (chunk_size - 1, chunk_size, chunk_size + 1):
                cached = loaded(input_ids=ids[:, :prefix], use_cache=True)
                prepared = loaded.prepare_inputs_for_generation(
                    ids[:, :prefix+1], past_key_values=cached.past_key_values, use_cache=True)
                step = loaded(**prepared)
                full = loaded(input_ids=ids[:, :prefix+1], use_cache=False)
                torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1],
                                           atol=2e-5, rtol=2e-4)
                errors[str(prefix)] = (step.logits[:, -1] - full.logits[:, -1]).abs().max().item()
        report = dict(variant=variant, backend="hf_cpu_torch_reference", seed=123,
                      parameters=sum(p.numel() for p in model.parameters()),
                      loss=out.loss.item(), anchor_embedding_updated=True,
                      reload_max_abs_error=(actual-expected).abs().max().item(),
                      cached_decode_max_abs_error=errors)
        Path(output, "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=["both", "safin1_4b", "safin1_35b_a3b"], default="both")
    args = parser.parse_args()
    variants = ["safin1_4b", "safin1_35b_a3b"] if args.variant == "both" else [args.variant]
    with tempfile.TemporaryDirectory(prefix="safin1-smoke-") as temp:
        root = args.output_dir or Path(temp)
        for variant in variants:
            print(json.dumps(run_variant(variant, root / variant), indent=2), flush=True)


if __name__ == "__main__":
    main()
