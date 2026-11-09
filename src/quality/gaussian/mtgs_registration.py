"""Transport physical motion into the matched source reconstruction basis.

This corrects a coordinate convention, not reconstruction quality or global
terrain registration. Source-frame refinements may vary along the trajectory.
"""

import numpy as np

def source_rotation(source):
    from driversi_world.data.assets.mtgs_binding import _matrix

    original = _matrix(source["ego2global_original"], "original source pose")
    refined = _matrix(source["ego2global"], "refined source pose")
    return refined[:3, :3] @ original[:3, :3].T

def motion_in_source(source, delta_xy, delta_yaw):
    xy = np.asarray(delta_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite([*xy, delta_yaw]).all():
        raise ValueError("Finite physical planar motion required")
    basis = source_rotation(source)
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = basis @ yaw @ basis.T
    translation = basis @ np.array([*xy, 0.0])
    half = delta_yaw / 2
    quaternion = np.r_[np.cos(half), basis[:, 2] * np.sin(half)]
    return translation, rotation, quaternion

def move_camera(source, pose, pivot, delta_xy, delta_yaw):
    pose, pivot = np.asarray(pose, dtype=float), np.asarray(pivot, dtype=float)
    if (
        pose.shape != (3, 4)
        or pivot.shape != (3,)
        or not np.isfinite([*pose.ravel(), *pivot]).all()
    ):
        raise ValueError("Finite camera and physical-centre pivot required")
    translation, rotation, _ = motion_in_source(source, delta_xy, delta_yaw)
    if np.array_equal(delta_xy, [0, 0]) and delta_yaw == 0:
        return pose.copy()
    result = pose.copy()
    result[:, :3] = rotation @ pose[:, :3]
    result[:, 3] = pivot + rotation @ (pose[:, 3] - pivot) + translation
    return result
