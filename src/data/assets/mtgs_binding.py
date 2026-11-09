"""Exact NAVSIM/MTGS provenance matching; never infer a global registration.

MTGS retains both original and refined poses. Matching tokens and original
poses establishes shared source data, not equality of refined geometry.
This module stays independent of torch and the optional renderer runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError(f"{label} must be a finite rigid 4x4 transform")
    return matrix
