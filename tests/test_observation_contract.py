"""Protect PPO observations from future information and cross-camera time mixing."""
import pytest

from driversi_world.world_models.observation_contract import (
    ObservationCapabilities,
    validate_observation_timestamps,
    past_window_indices,
)


def test_same_step_cameras_allow_past_conditioning():
    validate_observation_timestamps(2.0, {'front': 2.0, 'left': 2.0, 'right': 2.0}, 1.5)


@pytest.mark.parametrize('times,condition', [
    ({'front': 2.0, 'left': 2.0}, 2.5),
    ({'front': 2.0, 'left': 1.5}, 2.0),
    ({}, 2.0),
    ({'front': float('nan')}, 2.0),
    ({'front': 2.0}, float('nan')),
])
def test_invalid_rollout_observations_are_rejected(times, condition):
    with pytest.raises(ValueError):
        validate_observation_timestamps(2.0, times, condition)


def test_clip_throughput_does_not_replace_causal_latency():
    # Even a future-free, synchronized implementation still needs a latency measurement.
    candidate = ObservationCapabilities(False, True, True)
    assert any('latency' in reason for reason in candidate.rejection_reasons())
    assert len(ObservationCapabilities(True, False, False).rejection_reasons()) == 4


def test_history_selection_is_invariant_to_future_pixels():
    values = list(range(12))
    indices = past_window_indices(list(range(12)), 6, 5)
    before = [values[i] for i in indices]
    values[7:] = [-999] * 5
    assert [values[i] for i in indices] == before == [2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        past_window_indices([0, 1, 2], 1, 3)
