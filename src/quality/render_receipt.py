"""Sealed teacher branch -> bounded WorldEngine diagnostic images and window.

A complete rendering receipt is not a visual certificate. Its typed output
cannot be consumed by training services requiring an admitted dataset.
"""

import math

from driversi_world.runtime.files import digest, path_at

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")

HISTORY = [5, 10, 15, 20]

def validate_render(report, root, expected_inputs, *, future_points=8):
    """Fail closed for stale source, omitted/repeated ticks or camera frames."""
    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Expected eight primary or ten native future points")
    end_tick = 20 + future_points * 5
    future = list(range(25, end_tick + 1, 5))
    contract = report.get("image_contract", "history_future_rgb_v1")
    if contract not in {"history_future_rgb_v1", "history_rgb_future_states_v1", "anchor_rgb_history_future_states_v1"}:
        raise ValueError("Unknown image contract")
    ticks = ([20] if contract == "anchor_rgb_history_future_states_v1" else HISTORY if contract == "history_rgb_future_states_v1" else list(range(0, end_tick + 1, 5)))
    image_count = len(ticks) * len(CAMERAS)
    if report.get("status") != "rendered_diagnostic" or report.get("image_count") != image_count:
        raise ValueError("Incomplete diagnostic render")
    if report.get("history_ticks") != HISTORY or report.get("future_ticks") != future:
        raise ValueError("History/future window differs")
    if report.get("quality_admitted") is not False or report.get("trainable") is not False:
        raise ValueError("Renderer cannot grant visual or training admission")
    if report.get("inputs") != expected_inputs:
        raise ValueError("Renderer did not consume the sealed branch and asset")
    output_size = report.get("output_image_size")
    if output_size is not None and output_size not in ([960, 540], [1920, 1080]):
        raise ValueError("Invalid sealed output image size")
    rows = report.get("frames", [])
    if [row["tick"] for row in rows] != ticks:
        raise ValueError("Missing or repeated physical tick")
    images = {}
    support_flags = []
    for row in rows:
        if set(row["cameras"]) != set(CAMERAS):
            raise ValueError("Camera coverage differs")
        for camera, item in row["cameras"].items():
            name = f"{row['tick']:04d}-{camera}.png"
            if item["file"] != name or item["source_token"] != row["source_token"]:
                raise ValueError("Camera and source identity differ")
            terrain = item.get("terrain_evidence", {})
            if not terrain or set(terrain) != set(row.get("motions", {})):
                raise ValueError("Missing or mismatched terrain motion evidence")
            supported = all(
                value.get("status") in {"supported", "source_identity"}
                for value in terrain.values()
            )
            if item.get("terrain_fully_supported") is not supported:
                raise ValueError("Camera terrain summary disagrees with evidence")
            for token, value in terrain.items():
                if value.get("status") == "source_identity":
                    from driversi_world.quality.gaussian.mtgs_bindings import exact_float32_identity

                    binding = item.get("object_bindings", {}).get(token)
                    if (
                        report.get("renderer_backend") != "mtgs_native"
                        or not binding
                        or value.get("method") != "source_identity_transport_v1"
                        or value.get("source_token") != binding.get("source_token")
                        or value.get("reference_tick") != binding.get("reference_tick")
                    ):
                        raise ValueError("Native source identity transport lacks exact binding")
                    identity = exact_float32_identity(
                        binding["reference_physical_pose"], binding["target_physical_pose"]
                    )
                    if not identity["equal"] or value.get("identity") != identity:
                        raise ValueError("Native source identity transport actually moves geometry")
                if value.get("status") == "supported":
                    if value.get("method") not in {
                        "finite_bilinear_source_clearance_v1",
                        "measured_road_source_clearance_v1",
                    }:
                        raise ValueError("Unsupported terrain lifting method")
                    if value.get("method") == "measured_road_source_clearance_v1":
                        for side in ("source", "target"):
                            sample = value.get(side, {})
                            independent = sample.get("independent_surface_validation", {})
                            if independent.get("method") == "traversal_consensus_measured_road_v1":
                                from driversi_world.quality.gaussian.mtgs_road_surface import (
                                    validate_traversal_consensus_evidence,
                                )

                                validate_traversal_consensus_evidence(sample)
                                continue
                            if (
                                sample.get("status") != "supported"
                                or not independent
                                or independent.get("method")
                                != "dominant_measured_road_layer_disjoint_traversal_validation_v1"
                                or any(
                                    type(independent.get(key)) not in (int, float)
                                    or not math.isfinite(independent[key])
                                    or independent[key] < 0
                                    for key in (
                                        "independent_center_difference_m",
                                        "independent_max_plane_difference_m",
                                    )
                                )
                                or independent.get("independent_center_difference_m", math.inf)
                                > 0.03
                                or independent.get("independent_max_plane_difference_m", math.inf)
                                > 0.08
                            ):
                                raise ValueError(
                                    "Measured terrain lacks independent traversal validation"
                                )
                            from driversi_world.quality.gaussian.mtgs_road_surface import (
                                validate_independent_surface_evidence,
                            )

                            validate_independent_surface_evidence(sample)
                    numbers = [
                        value.get("source_clearance_m"),
                        value.get("vertical_correction_m"),
                        value.get("vertical_displacement_m"),
                    ]
                    if any(
                        type(number) not in (int, float) or not math.isfinite(number)
                        for number in numbers
                    ):
                        raise ValueError("Nonfinite terrain lifting evidence")
            support_flags.append(supported)
            path = root / name
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Missing or foreign frame")
            if output_size is not None:
                from PIL import Image

                with Image.open(path) as pixels:
                    if list(pixels.size) != output_size or item.get("output_image_size") != output_size:
                        raise ValueError("Rendered camera pixels and declared output size differ")
            images[str(path)] = digest(path)
    if report.get("terrain_fully_supported") is not all(support_flags):
        raise ValueError("Render terrain summary disagrees with per-camera evidence")
    return images
