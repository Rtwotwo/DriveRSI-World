"""Write the pinned PufferDrive binary ABI from a validated canonical scene."""

import numpy as np

from driversi_world.contracts.scene import Scene

def native_initial_center(scene: Scene):
    """Reproduce pinned set_means float operations, not an approximate mean.

    Native initialization uses present actors at t=0 followed by every road
    vertex in binary order. Future annotations do not affect this origin.
    """
    mean = np.zeros(2, dtype=np.float32)
    count = 0
    positions = scene.poses[0, scene.presence_at(0), :2]
    for points in (positions, scene.road_points[:, :2]):
        for point in np.asarray(points, dtype=np.float32):
            count += 1
            mean += (point - mean) / np.float32(count)
    if not count or not np.isfinite(mean).all():
        raise ValueError("Finite native initialization center required")
    return mean

def native_observed_pose(scene: Scene, pose):
    """Exact binary float32 -> centered state -> world readback representation."""
    value = np.asarray(pose, dtype=np.float32).copy()
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Finite source pose required")
    center = native_initial_center(scene)
    value[:2] = (value[:2] - center) + center
    return value.astype(float)
