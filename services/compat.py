"""Transformers compatibility patches for services running on newer transformers.

Many model libraries (IndexTTS, MOSS, GPT-SoVITS, etc.) import symbols that were
removed or renamed in transformers 4.55+. Call apply() once before loading models.
"""
from __future__ import annotations


def apply() -> None:
    """Apply all transformers compat patches. Idempotent — safe to call multiple times."""
    import transformers.cache_utils as _cu
    if not hasattr(_cu, 'QuantizedCacheConfig'):
        _cu.QuantizedCacheConfig = _cu.QuantizedCache

    # _crop_past_key_values removed in transformers 4.50+
    import transformers.generation.candidate_generator as _cg
    if not hasattr(_cg, '_crop_past_key_values'):
        _cg._crop_past_key_values = lambda model, output, max_length: output

    # NEED_SETUP_CACHE_CLASSES_MAPPING renamed in transformers 4.55+
    import transformers.generation.configuration_utils as _gcu
    if not hasattr(_gcu, 'NEED_SETUP_CACHE_CLASSES_MAPPING'):
        from transformers.cache_utils import DynamicCache
        _gcu.NEED_SETUP_CACHE_CLASSES_MAPPING = {
            "dynamic": DynamicCache,
            "static": _cu.StaticCache if hasattr(_cu, 'StaticCache') else DynamicCache,
            "offloaded_static": DynamicCache,
            "quantized": _cu.QuantizedCache if hasattr(_cu, 'QuantizedCache') else DynamicCache,
        }
    if not hasattr(_gcu, 'QUANT_BACKEND_CLASSES_MAPPING'):
        _gcu.QUANT_BACKEND_CLASSES_MAPPING = {}

    # SequenceSummary removed from transformers.modeling_utils
    import transformers.modeling_utils as _mu
    if not hasattr(_mu, 'SequenceSummary'):
        import torch.nn as nn

        class _SequenceSummary(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.summary_type = getattr(config, 'summary_type', 'last')

            def forward(self, hidden_states, **kwargs):
                return hidden_states[:, -1]

        _mu.SequenceSummary = _SequenceSummary

    # transformers.initialization removed in 4.50+ (MOSS model uses it)
    import transformers
    import sys
    if not hasattr(transformers, 'initialization'):
        import torch.nn
        transformers.initialization = torch.nn.init
        # Also inject into sys.modules so `from transformers import initialization` works
        sys.modules['transformers.initialization'] = torch.nn.init

    # forced_decoder_ids removed from GenerationConfig in newer transformers
    # (IndexTTS references it)
    from transformers import GenerationConfig
    if not hasattr(GenerationConfig, 'forced_decoder_ids'):
        GenerationConfig.forced_decoder_ids = property(
            lambda self: None,
            lambda self, val: None,
        )

    # begin_suppress_tokens may also be removed
    if not hasattr(GenerationConfig, 'begin_suppress_tokens'):
        GenerationConfig.begin_suppress_tokens = property(
            lambda self: None,
            lambda self, val: None,
        )

    # watermarking_config may also be missing
    if not hasattr(GenerationConfig, 'watermarking_config'):
        GenerationConfig.watermarking_config = property(
            lambda self: None,
            lambda self, val: None,
        )

    # PreTrainedConfig renamed to PretrainedConfig in transformers 5.x
    import transformers.configuration_utils as _cfu
    if not hasattr(_cfu, 'PreTrainedConfig'):
        _cfu.PreTrainedConfig = _cfu.PretrainedConfig

    # MODALITY_TO_BASE_CLASS_MAPPING renamed to AUTO_TO_BASE_CLASS_MAPPING in transformers 5.x
    # MOSS processor writes to this dict at module import time
    import transformers.processing_utils as _pu
    if not hasattr(_pu, 'MODALITY_TO_BASE_CLASS_MAPPING'):
        _pu.MODALITY_TO_BASE_CLASS_MAPPING = getattr(
            _pu, 'AUTO_TO_BASE_CLASS_MAPPING', {}
        )

    # ProcessorMixin.check_argument_for_proper_class rejects custom model classes
    # (e.g. MossAudioTokenizerModel) that don't match the registered AutoModel base.
    # Wrap it to skip the TypeError for these cases.
    import types
    _orig_check = _pu.ProcessorMixin.check_argument_for_proper_class
    if not getattr(_orig_check, '_compat_patched', False):
        def _permissive_check(self, attribute, arg):
            try:
                return _orig_check(self, attribute, arg)
            except TypeError:
                return type(arg).__name__
        _permissive_check._compat_patched = True
        _pu.ProcessorMixin.check_argument_for_proper_class = _permissive_check
