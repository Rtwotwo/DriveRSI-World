"""Coordinate isolation and content keys for editable Gaussian render states."""
import numpy as np


def reconstruction_agent_states(agent_states, translation):
    """Return private reconstruction-space arrays; never mutate simulator state."""
    shifted = {name: np.array(state, dtype=np.float64, copy=True)
               for name, state in agent_states.items()}
    for state in shifted.values():
        state[:2] -= np.asarray(translation)[:2]
    return shifted


def agent_state_key(timestamp, agent_states):
    """An edit at a fixed timestamp must invalidate cached world geometry."""
    return (timestamp, tuple((name, tuple(np.asarray(state).reshape(-1)))
                             for name, state in sorted(agent_states.items())))


def camera_state_key(sensors):
    """Include camera order, calibration and resolution in the sensor cache key."""
    fields = ('sensor2ego_rotation', 'sensor2ego_translation', 'intrinsic',
              'distortion', 'height', 'width')
    return tuple((name, tuple(tuple(np.asarray(camera[field]).reshape(-1))
                             for field in fields)) for name, camera in sensors.items())
