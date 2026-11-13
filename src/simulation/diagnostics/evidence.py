"""Image artifacts and measured comparisons for the action/render diagnostic."""
from pathlib import Path

import cv2
import numpy as np


def scaled_cameras(cameras, names, width):
    """Resize image-plane calibration together with resolution; retain distortion."""
    import copy
    result = {}
    for name in names:
        camera = copy.deepcopy(cameras[name])
        scale = width / int(camera['width'])
        height = round(int(camera['height']) * scale)
        intrinsic = np.asarray(camera['intrinsic'], dtype=float).copy()
        intrinsic[0, :] *= width / int(camera['width'])
        intrinsic[1, :] *= height / int(camera['height'])
        camera.update(intrinsic=intrinsic, width=width, height=height)
        result[name] = camera
    return result


def image_difference(first, second):
    difference = np.abs(first.astype(np.float32) - second.astype(np.float32))
    return {'mae_0_255': float(difference.mean()),
            'fraction_pixels_changed': float(np.any(difference > 0, axis=-1).mean())}


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError('Could not write image: ' + str(path))


def write_comparison(output, branch_images, camera_names, fps=2):
    """Rows = intervention branches; columns = calibrated camera views."""
    output = Path(output)
    names = list(branch_images)
    panels = []
    for frame in range(len(branch_images[names[0]])):
        rows = []
        for name in names:
            tiles = []
            for camera in camera_names:
                tile = branch_images[name][frame][camera].copy()
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
                cv2.putText(tile, f'{name} | {camera} | t={frame / fps:.1f}s', (7, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            rows.append(np.concatenate(tiles, axis=1))
        panels.append(np.concatenate(rows, axis=0))
    write_image(output / 'comparison.png', panels[-1])
    size = (panels[0].shape[1], panels[0].shape[0])
    writer = cv2.VideoWriter(str(output / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError('Video encoder failed to open')
    try:
        for panel in panels:
            writer.write(panel)
    finally:
        writer.release()
    return {'comparison_image': str(output / 'comparison.png'),
            'comparison_video': str(output / 'comparison.mp4')}
