"""Check that image diagnostics do not mix endpoint fidelity and interior drift."""
import numpy as np
import pytest
from PIL import Image
from driversi_world.world_models.hyperflow.evaluation import compare_clip


def test_interior_is_separate_from_conditioned_endpoints(tmp_path):
    reference = tmp_path / 'reference.png'
    Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(reference)
    video = np.zeros((5, 8, 16, 3), dtype=np.uint8)
    video[2] = 255
    scores = compare_clip(video, [reference] * 3)
    assert [r['generated_index'] for r in scores['anchors']] == [0, 2, 4]
    assert [r['rgb_mae'] for r in scores['anchors']] == [0., 1., 0.]
    assert scores['interior_rgb_mae'] == 1.
    assert scores['temporal_rgb_delta_mean'] == .5
    assert not scores['geometry_verified'] and not scores['ppo_admitted']


def test_float_video_is_not_silently_treated_as_bytes():
    with pytest.raises(ValueError, match='uint8'):
        compare_clip(np.zeros((2, 8, 8, 3), dtype=np.float32), [])
