"""Build exact-pixel GS sequence review material without assigning admission."""

import itertools

import numpy as np

def clipped_box_projection(corners, evidence, *, near=0.1):
    """Project the actual box edges after camera-near-plane clipping.

    A partially clipped box is still a required actor. This only supplies its
    geometric overlay; it never certifies appearance, occlusion or admission.
    """
    import cv2

    pose = np.asarray(evidence["camera_to_world"], dtype=float)
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("Finite eight-corner physical box required")
    xyz = (corners - pose[:3, 3]) @ pose[:3, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite camera-space box")
    segments, clip_count = [], 0
    for i, j in itertools.combinations(range(8), 2):
        if (i ^ j) not in (1, 2, 4):
            continue
        a, b = xyz[i].copy(), xyz[j].copy()
        if max(a[2], b[2]) < near:
            continue
        if min(a[2], b[2]) < near:
            cut = a + (b - a) * ((near - a[2]) / (b[2] - a[2]))
            if a[2] < near:
                a = cut
            else:
                b = cut
            clip_count += 1
        # Sample the distorted edge rather than drawing one undistorted chord.
        samples = a + np.linspace(0, 1, 17)[:, None] * (b - a)
        pixels = cv2.projectPoints(
            samples,
            np.zeros(3),
            np.zeros(3),
            np.asarray(evidence["distorted_intrinsic"], dtype=float),
            np.asarray(evidence["distortion"], dtype=float),
        )[0].reshape(-1, 2)
        if not np.isfinite(pixels).all():
            raise ValueError("Nonfinite clipped lens projection")
        segments.append(pixels.tolist())
    return dict(
        method="physical_box_near_plane_clipped_edges_v1",
        near_m=near,
        clipped_edges=clip_count,
        pixel_edges=segments,
        original_camera_depth=xyz[:, 2].tolist(),
        projection_supported=bool(segments),
        appearance_certified=False,
    )

def portable_actor_corners(frame, item, token, scene, states, canonical):
    """Independent observed 3D box transported by the executed physical delta."""
    from driversi_world.quality.gaussian.mtgs_bindings import observed_3d_box_corners
    from driversi_world.quality.gaussian.mtgs_registration import motion_in_source
    from driversi_world.quality.gaussian.worldengine_replay import physical_delta

    anchor = canonical[frame["source_token"]]
    j, tick = scene.actor_ids.index(token), frame["tick"]
    corners, annotation = observed_3d_box_corners(anchor, token, scene.dimensions[j])
    delta = physical_delta(states[tick, j], scene.poses[tick, j])
    if not np.allclose(delta, frame["motions"][token], rtol=0, atol=1e-5):
        raise ValueError("Portable audit motion differs from executed physical branch")
    terrain = item["terrain_evidence"][token]
    if terrain["status"] != "supported":
        raise ValueError("Portable bound actor has no supported 3D height")
    translation, rotation, _ = motion_in_source(anchor, delta[:2], delta[2])
    translation[2] = terrain["vertical_displacement_m"]
    center = corners.mean(0)
    return (corners - center) @ rotation.T + center + translation, annotation, terrain

def required_portable_actors(
    frame, item, scene, states, presence, canonical, *, allow_unknown_height=False
):
    """Recompute the required review set at final consumption, not from VLM text."""
    from driversi_world.quality.gaussian.worldengine_geometry import box_frustum_support

    moved = set(frame["motions"]) - {"ego"}
    missing = set(frame.get("unresolved_physical_boxes", {}))
    expected = {
        t for j, t in enumerate(scene.actor_ids) if t != "ego" and presence[frame["tick"], j]
    }
    if moved | missing != expected or not moved <= set(item["active_object_tokens"]):
        raise ValueError("Portable audit physical population is incomplete")
    required = set()
    for token in sorted(expected):
        if (
            allow_unknown_height
            and token in moved
            and item["terrain_evidence"][token]["status"] != "supported"
        ):
            required.add(token)  # Potentially visible; no invented Z or projected box.
            continue
        corners = (
            portable_actor_corners(frame, item, token, scene, states, canonical)[0]
            if token in moved
            else np.asarray(frame["unresolved_physical_boxes"][token], dtype=float)
        )
        if box_frustum_support(corners, item["camera_to_world"], item["output_ray_bounds"])[
            "outside_frustum"
        ]:
            continue
        projection = clipped_box_projection(corners, item)
        if not projection["projection_supported"]:
            raise ValueError("Portable actor has no supported clipped box projection")
        required.add(token)
    return required
