"""Dense GS-video guidance for Flash, implemented as a sampling-time projection.

The model has no trained pose-control input. After each native Flash step, blend
only generated video rows toward the same-time noisy GS latent sequence. This
retains dense trajectory evidence while leaving a generative residual. It is an
experimental visual constraint, not a guarantee of 3D or physical consistency.
"""
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep


def projected_rows(sample, reference, noise, sigma, strength):
    """Mix a sample with an equally noised reference on H3's sigma axis."""
    if sample.shape != reference.shape or reference.shape != noise.shape:
        raise ValueError('Generated, reference and noise rows must have identical shapes')
    if not 0 <= strength <= 1:
        raise ValueError('Projection strength must be in [0, 1]')
    target = (1 - sigma) * reference + sigma * noise
    return torch.lerp(sample, target, strength)


class TrajectoryProjectionStep(MiniMaxH3LoopSchedulerStep):
    """Retain the native video/audio update, then constrain generated video rows."""

    def __init__(self, strength=.65):
        super().__init__()
        self.strength = strength
        self.reference = self.noise = None
        self.trace = []

    def configure(self, latents, noise, patch_size):
        if latents.shape != noise.shape or latents.shape[0] != 1:
            raise ValueError('Expected matching single-video latent tensors')
        self.reference = patchify_video_latents(latents, patch_size)
        self.noise = patchify_video_latents(noise, patch_size)
        self.trace.clear()

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        components, block_state = super().__call__(components, block_state, i, t)
        if self.reference is None:
            raise RuntimeError('Dense GS reference was not encoded before sampling')
        count = block_state.num_condition_video_rows
        rows = block_state.latents[count:]
        reference = self.reference.to(rows.device, rows.dtype)
        noise = self.noise.to(rows.device, rows.dtype)
        sigma = components.scheduler.sigmas[i + 1].to(rows.device, rows.dtype)
        projected = projected_rows(rows, reference, noise, sigma, self.strength)
        self.trace.append(dict(step=i, sigma=float(sigma), strength=self.strength,
            mean_projection_change=float((projected - rows).abs().mean()),
            generated_rows=len(rows), untouched_condition_rows=count))
        block_state.latents[count:] = projected
        return components, block_state


def install_projection(blocks, strength=.65):
    """Replace exactly one native scheduler update before pipeline construction."""
    installed = []

    def walk(block):
        for name, child in list((getattr(block, 'sub_blocks', None) or {}).items()):
            if isinstance(child, MiniMaxH3LoopSchedulerStep):
                replacement = TrajectoryProjectionStep(strength)
                block.sub_blocks[name] = replacement
                installed.append(replacement)
            else:
                walk(child)

    walk(blocks)
    if len(installed) != 1:
        raise RuntimeError(f'Expected exactly one sampling update, found {len(installed)}')
    return installed[0]
