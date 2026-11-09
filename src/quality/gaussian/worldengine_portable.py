"""Portable WorldEngine asset renderer with explicit source-frame pose transport.

Uses owned copies of upstream Gaussian parameter/color functions. It does not
invent missing objects, extrapolate object lifetime, or certify pixel quality.
"""

import numpy as np

def ego_center_pivot(source):
    """nuPlan physical ego origin is 1.461 m ahead of the rear axle."""
    pose = np.asarray(source["ego2global"], dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Finite refined ego transform required")
    return pose[:3, 3] + pose[:3, 0] * 1.461

def camera_pose(source, camera):
    colmap = camera.get("colmap_param", {})
    if "sensor2global_rotation" not in colmap or "sensor2global_translation" not in colmap:
        raise ValueError("Reconstruction-refined camera pose is required")
    rotation = colmap["sensor2global_rotation"]
    rotation = (
        rotation.rotation_matrix if hasattr(rotation, "rotation_matrix") else np.asarray(rotation)
    )
    pose = np.c_[rotation, np.asarray(colmap["sensor2global_translation"])]
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        raise ValueError("Malformed refined camera pose")
    return pose
