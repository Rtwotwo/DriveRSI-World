"""Small, explicit perturbations of recorded center trajectories for wiring checks."""
import numpy as np


def lateral_reference(positions, headings, offset_m, history=4, frames=12, dt=0.5):
    """Preserve history, then smoothly reach a bounded offset in the log's left normal.

    This uses the recorded future as a diagnostic reference, not an autonomous
    driving policy. It tests the action-to-render path under ideal tracking.
    """
    positions = np.asarray(positions, dtype=np.float64)[:frames, :2].copy()
    headings = np.asarray(headings, dtype=np.float64).reshape(-1)[:frames].copy()
    if positions.shape != (frames, 2) or len(headings) != frames or not 2 <= history < frames:
        raise ValueError('A complete history/future window is required')
    if not np.isfinite(offset_m) or abs(offset_m) > 0.5:
        raise ValueError('This wiring experiment limits lateral displacement to 0.5 m')
    phase = np.clip((np.arange(frames) - (history - 1)) / (frames - history), 0, 1)
    offsets = offset_m * phase * phase * (3 - 2 * phase)
    normal = np.column_stack((-np.sin(headings), np.cos(headings)))
    positions += offsets[:, None] * normal
    # Keeping log yaw isolates translation effects; this is not a dynamics claim.
    velocities = np.gradient(positions, dt, axis=0)
    angular = np.gradient(np.unwrap(headings), dt)
    return positions, velocities, headings, angular


def center_step_action(agent, reference, next_index):
    """Issue one centered-frame step through the environment's native action API."""
    from driversi_world.simulation.common.dataclasses import Trajectory
    positions, velocities, headings, angular = reference
    return Trajectory(
        waypoints=np.stack((agent.current_position.copy(), positions[next_index])),
        velocities=np.stack((np.asarray(agent.current_velocity).copy(), velocities[next_index])),
        headings=np.array([agent.current_heading, headings[next_index]]),
        angular_velocities=np.array([agent.current_angular_velocity, angular[next_index]]),
    )
