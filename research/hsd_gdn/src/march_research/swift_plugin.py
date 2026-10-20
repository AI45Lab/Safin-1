"""ms-swift plugin loaded only by research/hsd_gdn/run.py."""
import os

from march_research.runtime import MODELS, assert_runtime, model_classes
from swift.model import ModelLoader, ModelMeta, register_model
from swift.template import TemplateType

assert_runtime()


class ResearchModelLoader(ModelLoader):
    def _update_attn_impl(self, config):
        # The packing argument selects the collator layout; these models own
        # their FLA/FlashAttention calls and do not use HF backend injection.
        return None

    def _postprocess_config(self, config):
        config = super()._postprocess_config(config)
        raw = os.environ.get('FUSE_CROSS_ENTROPY')
        if raw is not None and hasattr(config, 'fuse_cross_entropy'):
            config.fuse_cross_entropy = raw.lower() in {'1', 'true', 'yes', 'on'}
        return config


for model_type in MODELS:
    _, cls = model_classes(model_type)
    register_model(ModelMeta(model_type=model_type, model_groups=[], loader=ResearchModelLoader,
                             template=TemplateType.default, architectures=[cls.__name__]), exist_ok=True)

from swift.pipelines.train.pretrain import SwiftPretrain  # noqa: E402
from march_research.hsd.summary_utils import HSDDataCollator  # noqa: E402

if not getattr(SwiftPretrain, '_research_hsd_collator_patched', False):
    original_prepare = SwiftPretrain._prepare_template

    def prepare_template(self):
        original_prepare(self)
        config = getattr(self.model, 'config', None)
        if getattr(config, 'model_type', None) == 'hsd_gated_deltanet':
            if not isinstance(self.template.data_collator, HSDDataCollator):
                self.template.data_collator = HSDDataCollator(self.template.data_collator, config)

    SwiftPretrain._prepare_template = prepare_template
    SwiftPretrain._research_hsd_collator_patched = True

arrow = os.environ.get('MARCH_RESEARCH_ARROW_DATASET')
if arrow:
    from pathlib import Path
    from swift.dataset import AutoPreprocessor, DatasetMeta, register_dataset
    if Path(arrow).is_dir():
        arrow = str(Path(arrow) / 'data-*.arrow')
    register_dataset(DatasetMeta(dataset_name='march_research_pretrain_arrow', dataset_path=arrow,
                                preprocess_func=AutoPreprocessor(), huge_dataset=True), exist_ok=True)
