"""Render a sealed physical branch with released WorldEngine assets (diagnostic).

Missing appearance is reported, never inferred from token counts or low alpha.
This exporter cannot grant visual admission.
"""

import numpy as np

def physical_delta(state, source_pose):
    """Both contracts begin x, y, z, yaw; elevation is not planar yaw."""
    state, source_pose = np.asarray(state), np.asarray(source_pose)
    if (
        state.ndim != 1
        or len(state) < 4
        or source_pose.shape != (4,)
        or not np.isfinite(state[:4]).all()
        or not np.isfinite(source_pose).all()
    ):
        raise ValueError("Finite x/y/z/yaw poses required")
    xy = state[:2] - source_pose[:2]
    yaw = float((state[3] - source_pose[3] + np.pi) % (2 * np.pi) - np.pi)
    return [float(xy[0]), float(xy[1]), yaw]
