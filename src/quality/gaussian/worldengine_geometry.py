"""Bounded reconstruction geometry; these facts are not image admission.

Released road maps use Sim2 p_grid = s * (R @ p_reconstruction + t).
Unlike the upstream nearest/clamping alternatives, all interpolation support
must be finite and inside the map. A planar simulation is lifted by preserving
source clearance while using the road-height difference at the new position.
"""

import json

from pathlib import Path

import numpy as np

class UnsupportedTerrain(ValueError):
    pass

class RoadHeightMap:
    def __init__(self, grid, transform):
        self.grid = np.asarray(grid, dtype=float)
        self.rotation = np.asarray(transform["R"], dtype=float)
        self.translation = np.asarray(transform["t"], dtype=float)
        self.scale = float(transform["s"])
        if (
            self.grid.ndim != 2
            or min(self.grid.shape) < 2
            or self.rotation.shape != (2, 2)
            or self.translation.shape != (2,)
            or not np.isfinite([*self.rotation.ravel(), *self.translation, self.scale]).all()
            or self.scale <= 0
            or not np.allclose(self.rotation.T @ self.rotation, np.eye(2), atol=1e-8)
            or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-8)
        ):
            raise ValueError("Malformed released terrain map or Sim2 transform")

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        return cls(
            np.load(folder / "road_height_map.npy", allow_pickle=False),
            json.loads((folder / "sim2.json").read_text()),
        )

    def sample(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite reconstruction xy required")
        uv = self.scale * (self.rotation @ xy + self.translation)
        col, row = map(int, np.floor(uv))
        if row < 0 or col < 0 or row + 1 >= self.grid.shape[0] or col + 1 >= self.grid.shape[1]:
            raise UnsupportedTerrain("Outside finite height-map interpolation domain")
        values = self.grid[row : row + 2, col : col + 2]
        if not np.isfinite(values).all():
            raise UnsupportedTerrain("Incomplete finite height-map interpolation support")
        x, y = uv - [col, row]
        value = (
            values[0, 0] * (1 - x) * (1 - y)
            + values[0, 1] * x * (1 - y)
            + values[1, 0] * (1 - x) * y
            + values[1, 1] * x * y
        )
        return {
            "height_m": float(value),
            "grid_xy": uv.tolist(),
            "support_cells": [[row, col], [row, col + 1], [row + 1, col], [row + 1, col + 1]],
            "support_min_m": float(values.min()),
            "support_max_m": float(values.max()),
        }

    def lift(self, source_center, planar_translation):
        center, delta = (
            np.asarray(source_center, dtype=float),
            np.asarray(planar_translation, dtype=float),
        )
        if center.shape != (3,) or delta.shape != (3,) or not np.isfinite([*center, *delta]).all():
            raise ValueError("Finite source center and reconstruction translation required")
        source = self.sample(center[:2])
        target = self.sample((center + delta)[:2])
        corrected = delta.copy()
        corrected[2] = target["height_m"] - source["height_m"]
        return corrected, {
            "status": "supported",
            "method": "finite_bilinear_source_clearance_v1",
            "source": source,
            "target": target,
            "source_clearance_m": float(center[2] - source["height_m"]),
            "vertical_correction_m": float(corrected[2] - delta[2]),
            "vertical_displacement_m": float(corrected[2]),
            "orientation": "source_pitch_roll_with_transported_physical_yaw",
            "scope": "vertical_road_support_not_surface_orientation_certificate",
        }

def box_frustum_support(corners, camera_to_world, ray_bounds):
    """Conservative OBB/frustum half-space exclusion; no occlusion assumption.

    ray_bounds must enclose every distorted output-pixel ray. An intersecting
    box remains unresolved; this never treats a small/distant object as absent.
    """
    corners = np.asarray(corners, dtype=float)
    pose = np.asarray(camera_to_world, dtype=float)
    bounds = np.asarray(ray_bounds, dtype=float)
    if (
        corners.shape != (8, 3)
        or pose.shape != (3, 4)
        or bounds.shape != (4,)
        or not np.isfinite([*corners.ravel(), *pose.ravel(), *bounds]).all()
        or not bounds[0] < bounds[1]
        or not bounds[2] < bounds[3]
    ):
        raise ValueError("Finite box, camera and enclosing ray bounds required")
    camera = (corners - pose[:, 3]) @ pose[:, :3]
    xmin, xmax, ymin, ymax = bounds
    planes = np.array([[0, 0, 1], [1, 0, -xmin], [-1, 0, xmax], [0, 1, -ymin], [0, -1, ymax]])
    maxima = (camera @ planes.T).max(axis=0)
    excluded = np.flatnonzero(maxima < -1e-6).tolist()
    return {
        "outside_frustum": bool(excluded),
        "separating_planes": excluded,
        "plane_support_maxima": maxima.tolist(),
        "ray_bounds": bounds.tolist(),
        "method": "all_obb_corners_halfspace_separation_v1",
        "scope": "camera_frustum_only_not_occlusion_or_appearance_certificate",
    }
