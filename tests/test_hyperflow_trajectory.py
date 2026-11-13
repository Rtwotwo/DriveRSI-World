"""Check the noise convention and that projection cannot overwrite anchor/audio rows."""
from types import SimpleNamespace
import numpy as np
import torch

from driversi_world.world_models.hyperflow.blocks import flash_blocks
from driversi_world.world_models.hyperflow.trajectory import projected_rows, install_projection, TrajectoryProjectionStep
from driversi_world.simulation.diagnostics.trajectory_video import edited_path


def test_noise_endpoint_convention():
    sample, reference, noise = torch.zeros(3, 4), torch.ones(3, 4), torch.full((3, 4), 7.)
    assert torch.equal(projected_rows(sample, reference, noise, 1., 1.), noise)
    assert torch.equal(projected_rows(sample, reference, noise, 0., 1.), reference)
    assert torch.equal(projected_rows(sample, reference, noise, .6, 0.), sample)


def test_native_workflow_has_one_projection():
    from diffusers import MiniMaxH3ModularPipeline
    blocks = flash_blocks('fl2va')
    projection = install_projection(blocks)
    pipe = MiniMaxH3ModularPipeline(blocks=blocks)
    assert isinstance(projection, TrajectoryProjectionStep)
    assert pipe._blocks is blocks  # The reference handle must reach the executing instance.


def test_condition_and_audio_are_not_projected():
    class Scheduler:
        sigmas = torch.tensor([1., 0.])
        def step(self, prediction, timestep, sample, return_dict=False):
            return (sample + prediction,)
    step = TrajectoryProjectionStep(.5)
    step.reference, step.noise = torch.full((3, 4), 4.), torch.full((3, 4), 6.)
    state = SimpleNamespace(num_condition_video_rows=2, num_condition_audio_rows=1,
        latents=torch.full((5, 4), 2.), audio_latents=torch.full((2, 4), 3.),
        noise_pred=torch.ones(1, 5, 4), audio_noise_pred=torch.ones(1, 2, 4),
        audio_timesteps=torch.tensor([0.]))
    components = SimpleNamespace(scheduler=Scheduler(), audio_scheduler=Scheduler())
    _, state = step(components, state, 0, torch.tensor(0.))
    assert torch.equal(state.latents[:2], torch.full((2, 4), 2.))
    assert torch.equal(state.latents[2:], torch.full((3, 4), 3.5))
    assert torch.equal(state.audio_latents[0], torch.full((4,), 3.))
    assert torch.equal(state.audio_latents[1], torch.full((4,), 4.))


def test_three_meter_path_has_no_longitudinal_jump():
    times = np.linspace(0, 5.125, 124)
    ego = np.zeros((124, 6))
    ego[:, 0] = times * 2
    positions, yaw, offset = edited_path(ego, times, 3.)
    assert np.allclose(positions[:, 0], ego[:, 0])
    assert np.isclose(offset[0], 0) and np.isclose(offset[-1], 3)
    assert np.all(np.diff(offset) >= 0)
    assert np.array_equal(yaw, ego[:, -1])
