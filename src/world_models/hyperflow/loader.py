# Portions adapted from Video-Rebirth/hyperflow-h3-flash, Apache-2.0; see provenance.json and NOTICE.txt.
"""Offline, single-GPU Flash loader with an explicit memory bound.

The backbone exposes image/text conditioning, not vehicle-action or 3D control.
Use it to propose visual observations; physical state and admission remain external.
"""
from pathlib import Path

import torch
from torch import nn
from diffusers import MiniMaxH3ModularPipeline

from .blocks import flash_blocks
from .int8 import set_kernel_options
from .offload import apply_offload, model_components
from .transformer import HyperFlowH3FlashTransformer3DModel, set_block_chunk, set_ff_chunk, set_lowrank_rank_multiple


class _ShortWindowDiagnosticPipeline(MiniMaxH3ModularPipeline):
    """Explicit out-of-distribution probe; production retains the native 5 s minimum.

    The native decoder needs at least one complete temporal chunk (22 frames).
    Relaxing the public duration property lets us measure this execution path;
    it does not make the pretrained model a
    validated short-window restoration model or an online world simulator.
    """

    @property
    def min_duration(self):
        return 22 / self.fps


def load_pipeline(base_model, weights, *, device='cuda:0', memory_fraction=.40,
                  blocks=None, diagnostic_short_window=False):
    """Load only local reviewed components; stream weights on the shared GPU."""
    if not 0 < memory_fraction <= .40:
        raise ValueError('Shared-GPU pilot memory fraction must be in (0, 0.40]')
    base_model, weights = Path(base_model).resolve(strict=True), Path(weights).resolve(strict=True)
    if 'clones' in base_model.parts or 'clones' in weights.parts:
        raise ValueError('Runtime assets must not come from reference clones')
    torch.cuda.set_device(torch.device(device))
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    set_kernel_options(int8_gemm='triton', rotate_quant='factored')
    # Optional owned sampling blocks support dense geometry guidance in offline demos.
    pipeline_blocks = flash_blocks('fl2va') if blocks is None else blocks
    # init_pipeline() deep-copies blocks. The public constructor preserves the
    # caller's projection handle, which is configured after VAE reference encoding.
    pipeline_class = _ShortWindowDiagnosticPipeline if diagnostic_short_window else MiniMaxH3ModularPipeline
    pipe = pipeline_class(blocks=pipeline_blocks, pretrained_model_name_or_path=str(base_model))
    names = [name for name in pipe.pretrained_component_names if name != 'transformer']
    pipe.load_components(names=names, pretrained_model_name_or_path=str(base_model),
                         local_files_only=True, dtype=torch.bfloat16)
    if missing := [name for name in names if getattr(pipe, name, None) is None]:
        raise RuntimeError(f'Missing local base components: {missing}')
    # The conditioner reads one intermediate hidden state; upper decoder layers
    # and the language head cannot affect it. Match the upstream trimming rule.
    encoder = pipe.text_encoder
    keep = pipe.text_encoder_layer + 1
    language = encoder.model.language_model
    if len(language.layers) > keep:
        language.layers = language.layers[:keep]
        config = encoder.config.text_config
        config.num_hidden_layers = keep
        if getattr(config, 'layer_types', None):
            config.layer_types = config.layer_types[:keep]
    if isinstance(getattr(encoder, 'lm_head', None), nn.Linear):
        encoder.lm_head = nn.Identity()
    transformer = HyperFlowH3FlashTransformer3DModel.from_pretrained(
        str(weights), subfolder='fl2va_int8_convrot', local_files_only=True, torch_dtype=torch.bfloat16)
    set_lowrank_rank_multiple(transformer, 8)
    set_ff_chunk(transformer, 4096)
    set_block_chunk(transformer, 4096)
    transformer.set_attention_backend('native')
    pipe.update_components(transformer=transformer)
    pipe.audio_vae.set_attention_backend('native')
    apply_offload(model_components(pipe), torch.device(device), 'stream')
    return pipe
