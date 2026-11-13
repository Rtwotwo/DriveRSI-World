"""Regressions for repeatable, same-time edits of MTGS input state."""
from types import SimpleNamespace

import numpy as np
import torch

from driversi_world.simulation.render.mtgs.state import (
    reconstruction_agent_states, camera_state_key,
)
from driversi_world.simulation.render.mtgs.mtgs import MTGSRenderEngine


def test_coordinate_conversion_preserves_caller_and_repeated_render_inputs():
    original = np.array([101.0, 202.0, 4.5, 2.0, 1.7, 0.2])
    states = {'ego': original.copy()}
    first = reconstruction_agent_states(states, [100, 200, 0])
    second = reconstruction_agent_states(states, [100, 200, 0])
    np.testing.assert_array_equal(states['ego'], original)
    np.testing.assert_array_equal(first['ego'], second['ego'])
    np.testing.assert_allclose(first['ego'][:2], [1, 2])
    first['ego'][0] = -10
    np.testing.assert_array_equal(states['ego'], original)


def test_actual_world_cache_rebuilds_for_same_timestamp_actor_edit():
    renderer = MTGSRenderEngine.__new__(MTGSRenderEngine)
    renderer.device = 'cpu'
    renderer.node_types = {'car': 'na'}
    renderer.submodel_names = {'car': 'model'}
    calls = []

    def gaussians(quat, trans, timestamp):
        calls.append(trans.copy())
        return {key: torch.tensor(trans.copy()).reshape(1, -1)
                for key in ('means', 'scales', 'quats', 'opacities')}

    renderer.gaussian_models = {'model': SimpleNamespace(get_global_gaussians=gaussians)}
    renderer.get_agent_pose = lambda name, state: (None, state)
    states = {'car': np.array([1., 2., 0.])}
    renderer.update_world(100, states)
    renderer.update_world(100, states)
    assert len(calls) == 1
    states['car'][0] = 3.
    renderer.update_world(100, states)
    assert len(calls) == 2
    assert renderer.collected_gaussians['means'][0, 0] == 3.


def test_sensor_cache_key_detects_in_place_calibration_and_resolution_edits():
    camera = dict(sensor2ego_rotation=[1, 0, 0, 0], sensor2ego_translation=[0, 0, 0],
                  intrinsic=np.eye(3), distortion=np.zeros(5), height=1080, width=1920)
    sensors = {'CAM_F0': camera}
    initial = camera_state_key(sensors)
    camera['intrinsic'][0, 0] = 2.
    assert camera_state_key(sensors) != initial
    second = camera_state_key(sensors)
    camera['width'] = 480
    assert camera_state_key(sensors) != second
