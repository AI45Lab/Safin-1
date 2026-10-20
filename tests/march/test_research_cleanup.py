"""Retired research variants cannot alter the supported dense recipe."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
RESEARCH=ROOT/'research/hsd_gdn'


def policy():
    spec=importlib.util.spec_from_file_location('research_cleanup_policy', RESEARCH/'src/march_research/runtime_policy.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inactive_checkpoint_metadata_loads_without_restoring_retired_fields():
    config=json.loads((RESEARCH/'configs/hsd_gdn_0p8b_readq32.json').read_text())
    saved=dict(config, backbone_profile=None, block_norm_type='prenorm', hsd_sequence_profile='stage1_8k',
               true_vocab_size=32000, vocab_z_loss_coeff=0.0, olmo3_weight_decay=False)
    clean=policy().sanitize_research_config(saved)
    assert clean==config
    assert clean['hsd_history_read_q_mode']=='per_head_residual_lora'
    assert clean['hsd_history_read_q_lora_rank']==32
    assert saved['block_norm_type']=='prenorm'  # caller's checkpoint is not mutated


@pytest.mark.parametrize('name,value', [
    ('backbone_profile','olmo3_1b_all_hsd'), ('block_norm_type','olmo3_reordered'),
    ('vocab_z_loss_coeff',1e-5), ('olmo3_weight_decay',True),
    ('true_vocab_size',100278), ('hsd_sequence_profile','stage3_65k'),
    ('hsd_fused_global_read_profile','olmo3_1b'),
])
def test_enabled_retired_checkpoint_settings_fail(name,value):
    with pytest.raises(ValueError,match='[Rr]etired|OLMo3'):
        policy().sanitize_research_config({'vocab_size':32000,name:value})


@pytest.mark.parametrize('name,value', [
    ('HSD_HISTORY_STATE_FORMAT','low_rank'), ('HSD_HISTORY_STATE_FORMAT','unknown'),
    ('HSD_HISTORY_STATE_RANK','48'), ('HSD_HISTORY_FACTOR_POWER_ITERS','1'),
])
@pytest.mark.parametrize('entrypoint', ['run.py','train_recipe.py'])
def test_retired_storage_fails_before_gpu_launch(tmp_path,name,value,entrypoint):
    env=dict(os.environ)
    for key in ('MODEL','MODEL_CONFIG','DATASET','CACHED_DATASET','OUTPUT_DIR','NODE_RANK','MASTER_ADDR',
                'HSD_HISTORY_STATE_FORMAT','HSD_HISTORY_STATE_RANK','HSD_HISTORY_FACTOR_POWER_ITERS'):
        env.pop(key,None)
    env.update(DATASET=str(tmp_path/'data.jsonl'),OUTPUT_DIR=str(tmp_path/'output'),MODEL=str(tmp_path/'model'),
               NODE_RANK='0',MASTER_ADDR='192.0.2.10')
    env[name]=value
    command=[sys.executable,str(RESEARCH/entrypoint)]
    if entrypoint=='run.py': command+=['train','--profile','hsd_gdn_readq32']
    command+=['--dry-run']
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    assert result.returncode!=0
    assert name in result.stderr
    assert not (tmp_path/'output').exists()


def test_disabled_storage_environment_and_query_lora_are_independent():
    policy().validate_research_environment({'HSD_HISTORY_STATE_FORMAT':'dense', 'HSD_HISTORY_READ_Q_LORA_RANK':'32'})


def test_frozen_online_kernel_sources_still_match_the_restoration_snapshot():
    import hashlib
    manifest=json.loads((RESEARCH/'SOURCE_MANIFEST.json').read_text())
    for name in ('hsd_global_read.py','hsd_global_read_tile.py','hsd_global_read_tile_online_bwd.py','hsd_chunk_gdn.py'):
        expected=manifest['hsd_sources']['loglinear/hsd/'+name]['restored_sha256']
        assert hashlib.sha256((RESEARCH/'src/march_research/hsd'/name).read_bytes()).hexdigest()==expected


@pytest.mark.skipif(os.getenv('MARCH_TEST_RESEARCH')!='1',reason='Optional research dependencies are required')
def test_real_research_profiles_keep_shapes_and_old_checkpoint_loading(tmp_path):
    code=r'''
import json,sys
from pathlib import Path
import torch
from march_research.runtime import model_classes
root,output=map(Path,sys.argv[1:])
counts={
    'hsd_gdn_0p8b.json':900909300,
    'hsd_gdn_0p8b_readq32.json':845367540,
    'hsd_gdn_0p8b_topk4.json':843892980,
    'gdn_0p8b.json':792785532,
    'delta_net_0p8b.json':718273920,
    'kda_0p8b.json':814292094,
    'transformer_0p8b.json':692912640,
}
for name,count in counts.items():
    data=json.loads((root/'configs'/name).read_text())
    config_cls,model_cls=model_classes(data['model_type'])
    with torch.device('meta'):
        model=model_cls(config_cls(**data))
    assert sum(p.numel() for p in model.parameters())==count,name
    if name=='hsd_gdn_0p8b_readq32.json':
        downs=[p for key,p in model.named_parameters() if key.endswith('history_read_q_down_weight')]
        ups=[p for key,p in model.named_parameters() if key.endswith('history_read_q_up_weight')]
        assert len(downs)==len(ups)==20
        assert all(tuple(p.shape)==(6,32,192) for p in downs)
        assert all(tuple(p.shape)==(6,192,32) for p in ups)
from march_research.hsd.configuration_hsd_gated_deltanet import HSDGatedDeltaNetConfig
from march_research.hsd.modeling_hsd_gated_deltanet import HSDGatedDeltaNetForCausalLM
config=HSDGatedDeltaNetConfig(hidden_size=32,num_hidden_layers=2,num_heads=2,num_v_heads=2,head_dim=8,
    max_position_embeddings=64,summary_chunk_size=4,vocab_size=32,
    hsd_route_head_dim=4,hsd_route_key_heads=1,hsd_history_read_q_mode='per_head_residual_lora',
    hsd_history_read_q_lora_rank=4,hsd_read_layers=[0,1],use_short_conv=False,
    fuse_norm=False,fuse_swiglu=False,fuse_cross_entropy=False)
model=HSDGatedDeltaNetForCausalLM(config)
model.save_pretrained(output)
path=output/'config.json'; saved=json.loads(path.read_text())
saved.update(backbone_profile=None,block_norm_type='prenorm',hsd_sequence_profile='stage1_8k',
             true_vocab_size=32,vocab_z_loss_coeff=0.0,olmo3_weight_decay=False)
path.write_text(json.dumps(saved))
loaded,info=HSDGatedDeltaNetForCausalLM.from_pretrained(output,local_files_only=True,output_loading_info=True)
assert not any(info.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
assert model.state_dict().keys()==loaded.state_dict().keys()
assert all(torch.equal(value,loaded.state_dict()[key]) for key,value in model.state_dict().items())
assert 'block_norm_type' not in loaded.config.to_dict()
report={'meta_profiles':len(counts),'tiny_strict_reload_tensors':len(model.state_dict())}
(output.parent/'model-checks.json').write_text(json.dumps(report))
print(json.dumps(report))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OFFLINE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
             PYTHONPATH=os.pathsep.join([str(ROOT/'third_party/flash-linear-attention'),
                                       str(RESEARCH/'src'),str(ROOT)]))
    result=subprocess.run([sys.executable,'-c',code,str(RESEARCH),str(tmp_path/'checkpoint')],
                          env=env,capture_output=True,text=True,check=True)
    assert json.loads((tmp_path/'model-checks.json').read_text())['meta_profiles']==7
