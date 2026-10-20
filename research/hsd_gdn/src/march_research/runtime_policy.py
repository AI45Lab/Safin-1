"""Reject retired research variants while loading inactive historical metadata."""
import os


def validate_research_environment(environ=None):
    env = os.environ if environ is None else environ
    if env.get('HSD_HISTORY_STATE_FORMAT', 'dense').strip().lower() not in {'', 'dense'}:
        raise ValueError('Only dense history storage is supported; unset HSD_HISTORY_STATE_FORMAT')
    for name in ('HSD_HISTORY_STATE_RANK', 'HSD_HISTORY_FACTOR_POWER_ITERS'):
        if env.get(name, '').strip():
            raise ValueError(f'Low-rank history storage was removed; unset {name}')


def sanitize_research_config(config):
    """Consume old disabled fields; never reinterpret an enabled retired model."""
    validate_research_environment()
    values = dict(config)
    defaults = {
        'backbone_profile': None,
        'block_norm_type': 'prenorm',
        'hsd_sequence_profile': 'stage1_8k',
        'vocab_z_loss_coeff': 0.0,
        'olmo3_weight_decay': False,
    }
    for name, default in defaults.items():
        if name not in values:
            continue
        value = values.pop(name)
        if isinstance(default, str) and isinstance(value, str):
            value = value.strip().lower()
        if value != default:
            raise ValueError(f'Retired research option {name} is enabled; this checkpoint requires its original implementation')
    if 'true_vocab_size' in values:
        value = values.pop('true_vocab_size')
        if value is not None and int(value) != int(values.get('vocab_size', 32000)):
            raise ValueError('Retired true_vocab_size must match vocab_size')
    profile = values.get('hsd_fused_global_read_profile')
    if isinstance(profile, str) and profile.strip().lower().startswith('olmo3'):
        raise ValueError('The OLMo3 model profile is not supported by this research entrypoint')
    return values
