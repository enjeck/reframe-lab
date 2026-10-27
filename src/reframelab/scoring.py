from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import TextRegion

FloatArray = NDArray[np.float32]


def candidate_positions(frame_width: int, crop_width: int, step: int) -> NDArray[np.int32]:
    """Return candidate offsets along the active crop axis.

    The historical argument names are retained for API compatibility. ``frame_width``
    and ``crop_width`` mean the frame span and crop span on either the x or y axis.
    """

    crop_width = min(frame_width, max(1, crop_width))
    last = frame_width - crop_width
    if last <= 0:
        return np.array([0], dtype=np.int32)
    positions = np.arange(0, last + 1, max(1, step), dtype=np.int32)
    if positions[-1] != last:
        positions = np.append(positions, last).astype(np.int32)
    return positions


def candidate_statistics(
    importance_map: FloatArray,
    x_positions: NDArray[np.int32],
    crop_width: int,
    boundary_fraction: float = 0.08,
    *,
    axis: str = "x",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Score candidate windows along the active axis.

    For an x-axis crop, evidence is projected into columns. For a y-axis crop it is
    projected into rows. This keeps the temporal planner one-dimensional while
    supporting arbitrary source and output aspect ratios.
    """

    height, width = importance_map.shape
    if axis not in {"x", "y", "none"}:
        raise ValueError(f"Unsupported crop axis: {axis}")
    if axis == "y":
        frame_span = height
        crop_span = min(crop_width, height)
        masses = np.sum(np.maximum(importance_map, 0.0), axis=1, dtype=np.float64)
    else:
        frame_span = width
        crop_span = min(crop_width, width)
        masses = np.sum(np.maximum(importance_map, 0.0), axis=0, dtype=np.float64)

    total = float(masses.sum())
    count = len(x_positions)
    if total <= 1e-8:
        zeros = np.zeros(count, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()

    prefix = np.concatenate(([0.0], np.cumsum(masses)))
    weighted_prefix = np.concatenate(([0.0], np.cumsum(masses * np.arange(frame_span))))
    boundary_width = max(1, int(round(crop_span * boundary_fraction)))

    coverage = np.empty(count, dtype=np.float32)
    centering = np.empty(count, dtype=np.float32)
    boundary = np.empty(count, dtype=np.float32)

    for index, offset_value in enumerate(x_positions):
        start = int(offset_value)
        end = min(frame_span, start + crop_span)
        mass = prefix[end] - prefix[start]
        coverage[index] = float(mass / total)

        if mass <= 1e-8:
            centering[index] = 0.0
            boundary[index] = 0.0
            continue

        moment = weighted_prefix[end] - weighted_prefix[start]
        centroid = moment / mass
        crop_center = (start + end - 1) / 2.0
        half_span = max((end - start) / 2.0, 1.0)
        centering[index] = float(
            np.clip(1.0 - abs(centroid - crop_center) / half_span, 0.0, 1.0)
        )

        first_end = min(end, start + boundary_width)
        last_start = max(start, end - boundary_width)
        boundary_mass = (prefix[first_end] - prefix[start]) + (prefix[end] - prefix[last_start])
        boundary[index] = float(np.clip(boundary_mass / mass, 0.0, 1.0))

    return coverage, centering, boundary


def text_region_statistics(
    regions: list[TextRegion],
    x_positions: NDArray[np.int32],
    crop_width: int,
    frame_width: int,
    *,
    safety_margin_fraction: float = 0.035,
    axis: str = "x",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Score complete text boxes along the active crop axis.

    ``coverage`` is the weighted fraction of each box retained on the cropped axis.
    The other axis is fully retained by the maximum-aspect crop. ``boundary`` is high
    when a crop slices through a box or leaves it too close to an output edge.
    """

    count = len(x_positions)
    if not regions:
        zeros = np.zeros(count, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()
    if axis not in {"x", "y", "none"}:
        raise ValueError(f"Unsupported crop axis: {axis}")

    crop_start = x_positions.astype(np.float32)
    crop_end = crop_start + float(crop_width)
    crop_centers = (crop_start + crop_end) / 2.0
    coverage_total = np.zeros(count, dtype=np.float64)
    centering_total = np.zeros(count, dtype=np.float64)
    boundary_total = np.zeros(count, dtype=np.float64)
    total_weight = 0.0
    safety_margin = max(2.0, crop_width * safety_margin_fraction)

    for region in regions:
        if axis == "y":
            region_start_norm = region.y
            region_extent_norm = region.height
            region_center_norm = region.center_y
        else:
            region_start_norm = region.x
            region_extent_norm = region.width
            region_center_norm = region.center_x

        text_start = float(np.clip(region_start_norm, 0.0, 1.0)) * frame_width
        text_end = float(
            np.clip(region_start_norm + region_extent_norm, 0.0, 1.0)
        ) * frame_width
        text_span = max(1.0, text_end - text_start)
        text_center = float(np.clip(region_center_norm, 0.0, 1.0)) * frame_width
        area = max(region.width * region.height, 1e-6)
        weight = float(np.clip(region.score, 0.1, 1.0)) * (0.35 + np.sqrt(area) * 4.5)
        overlap = np.maximum(
            0.0,
            np.minimum(crop_end, text_end) - np.maximum(crop_start, text_start),
        )
        fraction = np.clip(overlap / text_span, 0.0, 1.0)
        coverage_total += fraction * weight

        distance = np.abs(crop_centers - text_center)
        centering = np.clip(1.0 - distance / max(crop_width * 0.62, 1.0), 0.0, 1.0)
        centering_total += centering * fraction * weight

        partial = ((fraction > 1e-4) & (fraction < 0.999)).astype(np.float32)
        fully_inside = fraction >= 0.999
        leading_margin = text_start - crop_start
        trailing_margin = crop_end - text_end
        edge_margin = np.minimum(leading_margin, trailing_margin)
        close_to_edge = np.where(
            fully_inside,
            np.clip((safety_margin - edge_margin) / safety_margin, 0.0, 1.0),
            0.0,
        )
        outside = (fraction <= 1e-4).astype(np.float32)
        boundary_total += (0.92 * partial + 0.34 * close_to_edge + 0.08 * outside) * weight
        total_weight += weight

    if total_weight <= 1e-8:
        zeros = np.zeros(count, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()
    coverage = np.clip(coverage_total / total_weight, 0.0, 1.0)
    centering = np.clip(centering_total / total_weight, 0.0, 1.0)
    boundary = np.clip(boundary_total / total_weight, 0.0, 1.0)
    return (
        coverage.astype(np.float32),
        centering.astype(np.float32),
        boundary.astype(np.float32),
    )


def combine_signals(
    x_positions: NDArray[np.int32],
    crop_width: int,
    frame_width: int,
    weights: dict[str, float],
    coverage: dict[str, FloatArray],
    centering: dict[str, FloatArray],
    boundary: dict[str, FloatArray],
    reliability: dict[str, float],
) -> tuple[FloatArray, FloatArray]:
    """Combine evidence for candidate positions on either active axis."""

    candidate_count = len(x_positions)
    score = np.zeros(candidate_count, dtype=np.float32)
    content_coverage = np.zeros(candidate_count, dtype=np.float32)
    weight_sum = 0.0

    for name, configured_weight in weights.items():
        if configured_weight <= 0.0 or name not in coverage:
            continue
        active_weight = configured_weight * float(
            np.clip(reliability.get(name, 0.0), 0.0, 1.0)
        )
        if active_weight <= 1e-6:
            continue
        if name == "text_complete":
            signal_score = 1.10 * coverage[name] + 0.05 * centering[name] - 0.78 * boundary[name]
        elif name == "text":
            signal_score = coverage[name] + 0.06 * centering[name] - 0.34 * boundary[name]
        else:
            signal_score = coverage[name] + 0.10 * centering[name] - 0.18 * boundary[name]
        score += active_weight * signal_score
        content_coverage += active_weight * coverage[name]
        weight_sum += active_weight

    if weight_sum > 1e-8:
        score /= weight_sum
        content_coverage /= weight_sum
    else:
        score.fill(0.0)
        content_coverage.fill(0.0)

    candidate_centers = x_positions.astype(np.float32) + crop_width / 2.0
    frame_center = frame_width / 2.0
    distance = np.abs(candidate_centers - frame_center) / max(frame_width / 2.0, 1.0)
    score -= 0.015 * distance.astype(np.float32)

    return score.astype(np.float32), content_coverage.astype(np.float32)
