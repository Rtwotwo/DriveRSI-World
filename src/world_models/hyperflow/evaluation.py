"""Image-space diagnostics only; these scores do not certify physical validity."""
import numpy as np
from PIL import Image


def compare_clip(frames, references):
    """Compare equally spaced GS anchors with normalized-time generated frames.

    GS anchors are renderings, not held-out camera ground truth. The mapping
    deliberately makes no claim that the generated clip follows ego dynamics.
    """
    video = np.asarray(frames)
    if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError('Expected uint8 video [time, height, width, RGB]')
    rows = []
    indices = np.rint(np.linspace(0, len(video) - 1, len(references))).astype(int)
    for i, (path, index) in enumerate(zip(references, indices)):
        with Image.open(path) as image:
            target = np.asarray(image.convert('RGB').resize((video.shape[2], video.shape[1]),
                Image.Resampling.LANCZOS), dtype=np.float32) / 255
        actual = video[index].astype(np.float32) / 255
        error = np.abs(actual - target)
        rows.append(dict(gs_anchor=i, generated_index=int(index), rgb_mae=float(error.mean()),
            is_endpoint=i in (0, len(references) - 1)))
    interior = [row['rgb_mae'] for row in rows[1:-1]]
    delta = np.abs(np.diff(video.astype(np.float32) / 255, axis=0))
    return dict(reference_kind='GS renders; no novel-view ground truth',
        alignment='normalized time, not a vehicle-dynamics timestamp match',
        anchors=rows, interior_rgb_mae=float(np.mean(interior)) if interior else None,
        temporal_rgb_delta_mean=float(delta.mean()), geometry_verified=False,
        action_conditioning_verified=False, ppo_admitted=False)
