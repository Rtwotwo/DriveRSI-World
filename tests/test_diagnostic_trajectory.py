"""Keep intervention history and the stated camera/trajectory bounds honest."""
import numpy as np
import pytest

from driversi_world.simulation.diagnostics.trajectories import lateral_reference
from driversi_world.simulation.diagnostics.evidence import scaled_cameras


def test_lateral_reference_preserves_history_and_has_declared_bound():
    positions = np.column_stack((np.arange(12), np.zeros(12)))
    reference, _, _, _ = lateral_reference(positions, np.zeros(12), .5)
    np.testing.assert_array_equal(reference[:4], positions[:4])
    np.testing.assert_allclose(reference[-1], [11, .5])
    assert np.max(np.linalg.norm(reference - positions, axis=1)) <= .5
    with pytest.raises(ValueError):
        lateral_reference(positions, np.zeros(12), 1)


def test_scaled_calibration_preserves_normalized_rays_and_input():
    intrinsic = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
    camera = dict(width=1920, height=1080, intrinsic=intrinsic, distortion=np.zeros(5))
    scaled = scaled_cameras({'front': camera}, ['front'], 480)['front']
    ray = np.array([.1, .2, 1.])
    np.testing.assert_allclose((scaled['intrinsic'] @ ray)[:2], (intrinsic @ ray)[:2] / 4)
    np.testing.assert_array_equal(camera['intrinsic'], intrinsic)
    assert (scaled['width'], scaled['height']) == (480, 270)
