"""Admission checks for an observation enhancer attached to a PPO environment.

These checks are necessary, not sufficient: simulation dynamics, rewards and
policy optimization remain the responsibility of the existing environment.
"""
from dataclasses import dataclass
import math
from bisect import bisect_right


@dataclass(frozen=True)
class ObservationCapabilities:
    requires_future_frames: bool
    updates_from_executed_state: bool
    multiview_validated: bool
    measured_next_observation_latency_seconds: float | None = None

    def rejection_reasons(self):
        reasons=[]
        if self.requires_future_frames: reasons.append('requires future trajectory/video frames')
        if not self.updates_from_executed_state: reasons.append('not driven by executed simulator state each step')
        if not self.multiview_validated: reasons.append('multi-camera consistency not validated')
        latency = self.measured_next_observation_latency_seconds
        if latency is None:
            reasons.append('causal next-observation latency is unmeasured; clip throughput is not latency')
        elif not math.isfinite(latency) or latency < 0:
            reasons.append('causal next-observation latency is invalid')
        return reasons


def validate_observation_timestamps(step_time, camera_times, newest_condition_time, tolerance=1e-6):
    """Reject future leakage and unsynchronized camera observations before rollout storage."""
    values = [step_time, newest_condition_time, tolerance, *camera_times.values()]
    if not all(math.isfinite(value) for value in values) or tolerance < 0:
        raise ValueError('Observation timestamps and tolerance must be finite and valid')
    if newest_condition_time>step_time+tolerance: raise ValueError('Future conditioning is forbidden')
    if not camera_times or any(abs(t-step_time)>tolerance for t in camera_times.values()):
        raise ValueError('All cameras must describe the same executed simulator step')


def past_window_indices(timestamps, current_time, length):
    """Select only observed history; startup padding must be an explicit caller policy."""
    if length < 1 or not math.isfinite(current_time):
        raise ValueError('A valid current time and positive history length are required')
    if not all(math.isfinite(t) for t in timestamps):
        raise ValueError('History timestamps must be finite')
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('History timestamps must strictly increase')
    stop = bisect_right(timestamps, current_time)
    if stop < length:
        raise ValueError('Insufficient observed history; never pad with future frames')
    return tuple(range(stop - length, stop))
