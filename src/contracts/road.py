"""Versioned lane topology and bounded local routes, independent of log futures."""

from dataclasses import dataclass

import numpy as np

LANE_MATCH_DISTANCE_LIMIT_M = 4.0

LANE_MATCH_HEADING_LIMIT_RAD = np.pi / 3

LANE_MATCH_HEADING_WEIGHT_M_PER_RAD = 2.0

LANE_MATCH_SCORE_TOLERANCE_M = .25

@dataclass(frozen=True)
class LaneMatchCandidate:
    node_id: str
    distance_m: float
    heading_error_rad: float
    score_m: float
    arc_length_m: float
    length_m: float
    outgoing: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "distance_m": self.distance_m,
                "heading_error_rad": self.heading_error_rad, "score_m": self.score_m,
                "arc_length_m": self.arc_length_m,
                "length_m": self.length_m, "outgoing": list(self.outgoing)}

@dataclass(frozen=True)
class LaneMatch:
    """Current-pose evidence; a match never infers an unobserved turn intention."""

    status: str
    reason: str
    selected_node_id: str | None
    candidates: tuple[LaneMatchCandidate, ...] = ()
    ambiguous_node_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"schema_version": 1, "status": self.status, "reason": self.reason,
                "selected_node_id": self.selected_node_id,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "ambiguous_node_ids": list(self.ambiguous_node_ids),
                "matching_policy": {"version": "pose-distance-heading-v1",
                                    "score": "distance_m + heading_weight_m_per_rad * heading_error_rad",
                                    "distance_limit_m": LANE_MATCH_DISTANCE_LIMIT_M,
                                    "heading_limit_rad": LANE_MATCH_HEADING_LIMIT_RAD,
                                    "heading_weight_m_per_rad": LANE_MATCH_HEADING_WEIGHT_M_PER_RAD,
                                    "ambiguity_score_tolerance_m": LANE_MATCH_SCORE_TOLERANCE_M,
                                    "seam_projection_limit_m": 1.0, "seam_geometry_gap_limit_m": .25}}

class LaneMatchError(ValueError):
    def __init__(self, diagnostic: LaneMatch):
        self.diagnostic = diagnostic
        message = {"ambiguous_initial_lane": "Ambiguous initial lane match",
                   "no_compatible_lane": "No direction-compatible lane within 4 m",
                   "outside_loaded_region": "Pose outside loaded map region",
                   "explicit_node_not_candidate": "Explicit initial node is not a direction-compatible lane within 4 m"}
        super().__init__(message.get(diagnostic.reason, diagnostic.reason))

@dataclass(frozen=True)
class Route:
    points: np.ndarray
    node_ids: tuple[str, ...] = ()
    end_reason: str = "explicit_endpoint"
    speed_limit_mps: float | None = None

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float).copy()
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Route requires at least two finite xy points")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-6):
            raise ValueError("Route contains duplicate consecutive points")
        if self.speed_limit_mps is not None and (not np.isfinite(self.speed_limit_mps) or self.speed_limit_mps <= 0):
            raise ValueError("Known speed limit must be positive and finite")
        points.setflags(write=False)
        object.__setattr__(self, "points", points)

    @property
    def distances(self) -> np.ndarray:
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]

    def project(self, xy) -> tuple[float, float, float]:
        """Closest arc length, unsigned distance and tangent heading."""
        edges = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        fractions = np.clip(np.sum((np.asarray(xy) - self.points[:-1]) * edges, axis=1) / lengths**2, 0, 1)
        distances = np.linalg.norm(self.points[:-1] + fractions[:, None] * edges - xy, axis=1)
        index = int(np.argmin(distances))
        return (float(self.distances[index] + fractions[index] * lengths[index]),
                float(distances[index]), float(np.arctan2(*edges[index, ::-1])))

    def project_many(self, xy) -> np.ndarray:
        """Same projection/tie rule as project(), batched over query positions."""
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Expected Nx2 positions")
        if not len(xy):
            return np.empty((0, 3), dtype=float)
        edges = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        fractions = np.clip(np.sum((xy[:, None, :] - self.points[None, :-1, :]) * edges[None, :, :], axis=2) / lengths[None, :]**2, 0, 1)
        distances = np.linalg.norm(self.points[None, :-1, :] + fractions[:, :, None] * edges[None, :, :] - xy[:, None, :], axis=2)
        index = np.argmin(distances, axis=1)
        rows = np.arange(len(xy))
        return np.column_stack((self.distances[index] + fractions[rows, index] * lengths[index],
            distances[rows, index], np.arctan2(edges[index, 1], edges[index, 0])))

    def at(self, distance: float) -> np.ndarray:
        return np.asarray([np.interp(distance, self.distances, self.points[:, i]) for i in (0, 1)])

    def to_dict(self) -> dict:
        return {"points": self.points.tolist(), "node_ids": self.node_ids,
                "end_reason": self.end_reason, "speed_limit_mps": self.speed_limit_mps}
