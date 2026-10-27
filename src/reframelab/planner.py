from __future__ import annotations

from collections import defaultdict

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .corrections import FramingCorrection
from .geometry import (
    choose_layout_split_axis,
    crop_geometry,
)
from .geometry import (
    panel_crop_fraction as layout_panel_crop_fraction,
)
from .models import AnalysisBundle, CropPlan, PlanPoint, Profile
from .runtime import opencv_runtime_info
from .scoring import combine_signals
from .semantic import speaker_region_strength
from .targets import TargetObservation, TargetTrack

FloatArray = NDArray[np.float32]


def _active_axis(bundle: AnalysisBundle) -> str:
    axis = getattr(bundle, "crop_axis", None) or getattr(bundle.frames[0], "crop_axis", "x")
    return axis if axis in {"x", "y", "none"} else "x"


def _frame_span(bundle: AnalysisBundle) -> int:
    return bundle.analysis_height if _active_axis(bundle) == "y" else bundle.analysis_width


def _crop_span(frame) -> int:
    if getattr(frame, "crop_axis", "x") == "y":
        return int(frame.crop_height or 0) or 1
    return int(frame.crop_width)


def _axis_center_xy(axis: str, center: float) -> tuple[float, float]:
    value = float(np.clip(center, 0.0, 1.0))
    if axis == "y":
        return 0.5, value
    if axis == "x":
        return value, 0.5
    return 0.5, 0.5


def _point_axis_center(point: PlanPoint, axis: str) -> float:
    if axis == "y":
        return float(point.center_y_norm)
    if axis == "x":
        return float(point.center_x_norm)
    return 0.5


def _point_secondary_axis_center(point: PlanPoint, axis: str) -> float | None:
    if axis == "y":
        return point.secondary_center_y_norm
    if axis == "x":
        return point.secondary_center_x_norm
    return None


def _observation_axis_values(observation: TargetObservation, axis: str) -> tuple[float, float]:
    if axis == "y":
        return float(observation.center_y), float(observation.height)
    return float(observation.center_x), float(observation.width)


def _region_axis_values(region, axis: str) -> tuple[float, float]:
    if axis == "y":
        return float(region.center_y), float(region.height)
    return float(region.center_x), float(region.width)


def _local_confidence(scores: FloatArray, chosen_index: int) -> float:
    if len(scores) <= 1:
        return 1.0
    best_two = np.partition(scores, -2)[-2:]
    best = float(best_two.max())
    second = float(best_two.min())
    spread = float(np.max(scores) - np.min(scores))
    margin = (best - second) / max(spread, 1e-6)
    chosen_gap = max(0.0, best - float(scores[chosen_index])) / max(spread, 1e-6)
    return float(np.clip(0.35 + 1.8 * margin - 0.7 * chosen_gap, 0.0, 1.0))


def _dynamic_programming_path(
    score_matrix: FloatArray,
    x_positions: NDArray[np.int32],
    crop_width: int,
    movement_penalty: float,
    maximum_transition_fraction: float,
) -> NDArray[np.int32]:
    time_count, candidate_count = score_matrix.shape
    if time_count == 0:
        return np.empty(0, dtype=np.int32)
    if candidate_count == 1:
        return np.zeros(time_count, dtype=np.int32)

    centers = x_positions.astype(np.float32) + crop_width / 2.0
    distance = np.abs(centers[:, None] - centers[None, :])
    transition_cost = movement_penalty * distance / max(float(crop_width), 1.0)
    maximum_shift = maximum_transition_fraction * crop_width
    transition_cost[distance > maximum_shift] = np.inf

    backpointers = np.zeros((time_count, candidate_count), dtype=np.int32)
    previous_value = score_matrix[0].astype(np.float64)

    for time_index in range(1, time_count):
        possibilities = previous_value[:, None] - transition_cost
        previous_state = np.argmax(possibilities, axis=0)
        best_transition = possibilities[previous_state, np.arange(candidate_count)]
        backpointers[time_index] = previous_state
        previous_value = score_matrix[time_index] + best_transition

    path = np.zeros(time_count, dtype=np.int32)
    path[-1] = int(np.argmax(previous_value))
    for time_index in range(time_count - 1, 0, -1):
        path[time_index - 1] = backpointers[time_index, path[time_index]]
    return path


def _stabilize_centers(
    desired_centers: NDArray[np.float32],
    times: NDArray[np.float64],
    shot_ids: NDArray[np.int32],
    frame_width: int,
    crop_width: int,
    profile: Profile,
) -> NDArray[np.float32]:
    """Apply a dead zone and jerk-limited virtual-camera controller.

    The previous controller limited speed and acceleration, but it could still change
    acceleration instantaneously at every analysis sample. That is perceived as a jump
    even when the crop coordinates remain continuous. This controller also limits jerk
    (change in acceleration) and eases into a stop near the target.
    """

    if len(desired_centers) == 0:
        return desired_centers

    minimum_center = crop_width / 2.0
    maximum_center = frame_width - crop_width / 2.0
    result = np.empty_like(desired_centers, dtype=np.float32)
    position = float(np.clip(desired_centers[0], minimum_center, maximum_center))
    velocity = 0.0
    acceleration = 0.0
    result[0] = position

    dead_zone = crop_width * profile.dead_zone_fraction / 2.0
    maximum_speed = crop_width * profile.max_speed_crop_widths_per_second
    maximum_acceleration = crop_width * profile.max_acceleration_crop_widths_per_second2
    maximum_jerk = crop_width * profile.max_jerk_crop_widths_per_second3

    for index in range(1, len(desired_centers)):
        if shot_ids[index] != shot_ids[index - 1]:
            position = float(np.clip(desired_centers[index], minimum_center, maximum_center))
            velocity = 0.0
            acceleration = 0.0
            result[index] = position
            continue

        dt = max(float(times[index] - times[index - 1]), 1e-3)
        error = float(desired_centers[index] - position)
        if abs(error) <= dead_zone:
            target_position = position
        else:
            target_position = float(desired_centers[index] - np.sign(error) * dead_zone)
        target_position = float(np.clip(target_position, minimum_center, maximum_center))
        remaining = target_position - position

        # A proportional velocity target is calmer than trying to complete every move
        # in a single analysis interval. The stopping-speed term prevents overshoot.
        proportional_speed = 2.2 * remaining
        stopping_speed = np.sqrt(max(0.0, 2.0 * maximum_acceleration * abs(remaining)))
        desired_speed = min(abs(proportional_speed), stopping_speed, maximum_speed)
        desired_velocity = float(np.sign(remaining) * desired_speed) if abs(remaining) > 1e-6 else 0.0
        desired_acceleration = float(
            np.clip((desired_velocity - velocity) / dt, -maximum_acceleration, maximum_acceleration)
        )
        acceleration += float(
            np.clip(
                desired_acceleration - acceleration,
                -maximum_jerk * dt,
                maximum_jerk * dt,
            )
        )
        acceleration = float(np.clip(acceleration, -maximum_acceleration, maximum_acceleration))
        velocity += acceleration * dt
        velocity = float(np.clip(velocity, -maximum_speed, maximum_speed))
        next_position = position + velocity * dt

        if remaining != 0.0 and remaining * (target_position - next_position) <= 0.0:
            next_position = target_position
            velocity = 0.0
            acceleration = 0.0
        elif abs(remaining) <= max(1.0, crop_width * 0.004) and abs(velocity) < crop_width * 0.03:
            next_position = target_position
            velocity = 0.0
            acceleration = 0.0

        position = float(np.clip(next_position, minimum_center, maximum_center))
        if position in (minimum_center, maximum_center):
            velocity = 0.0
            acceleration = 0.0
        result[index] = position

    return result


def _difference_matrix(length: int, order: int, dt: float) -> NDArray[np.float64]:
    if length <= order:
        return np.zeros((0, length), dtype=np.float64)
    coefficients = {
        1: np.array([-1.0, 1.0], dtype=np.float64),
        2: np.array([1.0, -2.0, 1.0], dtype=np.float64),
        3: np.array([-1.0, 3.0, -3.0, 1.0], dtype=np.float64),
    }[order] / max(dt**order, 1e-9)
    matrix = np.zeros((length - order, length), dtype=np.float64)
    for row in range(length - order):
        matrix[row, row : row + order + 1] = coefficients
    return matrix


def _dead_zone_reference(
    desired: NDArray[np.float64],
    dead_zone: float,
    minimum_center: float,
    maximum_center: float,
) -> NDArray[np.float64]:
    reference = np.empty_like(desired)
    reference[0] = np.clip(desired[0], minimum_center, maximum_center)
    for index in range(1, len(desired)):
        error = desired[index] - reference[index - 1]
        if abs(error) <= dead_zone:
            reference[index] = reference[index - 1]
        else:
            reference[index] = desired[index] - np.sign(error) * dead_zone
        reference[index] = np.clip(reference[index], minimum_center, maximum_center)
    return reference


def _minimum_jerk_shot(
    desired: NDArray[np.float64],
    times: NDArray[np.float64],
    crop_width: int,
    frame_width: int,
    profile: Profile,
) -> NDArray[np.float64]:
    """Solve each movement episode with lookahead and a minimum-jerk objective.

    Stable sections are pinned exactly. Around a meaningful target change, a short
    lookahead window is solved as one curve. This keeps the benefit of planning ahead
    without allowing a change near the end of a long shot to move the camera early.
    """

    count = len(desired)
    if count <= 1:
        return desired.copy()
    minimum_center = crop_width / 2.0
    maximum_center = frame_width - crop_width / 2.0
    dead_zone = crop_width * profile.dead_zone_fraction / 2.0
    reference = _dead_zone_reference(desired, dead_zone, minimum_center, maximum_center)
    dt = max(float(np.median(np.diff(times))), 1e-3)

    change_threshold = max(1.0, crop_width * 0.005)
    changes = np.flatnonzero(np.abs(np.diff(reference)) > change_threshold) + 1
    if len(changes) == 0:
        return reference

    active = np.zeros(count, dtype=bool)
    lookahead = max(1, int(round(0.9 / dt)))
    settle = max(1, int(round(0.75 / dt)))
    for change in changes:
        active[max(0, change - lookahead) : min(count, change + settle + 1)] = True

    # Join nearby movement windows. Then include one pinned sample on each side so the
    # solved curve meets the stable path with no position discontinuity.
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, value in enumerate(np.append(active, False)):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    result = reference.copy()
    for raw_start, raw_end in runs:
        start = max(0, raw_start - 1)
        end = min(count - 1, raw_end + 1)
        local_reference = reference[start : end + 1]
        local_times = times[start : end + 1]
        local_count = len(local_reference)
        if local_count <= 2:
            result[start : end + 1] = local_reference
            continue
        local_dt = max(float(np.median(np.diff(local_times))), 1e-3)
        normalized = local_reference / max(float(crop_width), 1.0)
        d1 = _difference_matrix(local_count, 1, local_dt)
        d2 = _difference_matrix(local_count, 2, local_dt)
        d3 = _difference_matrix(local_count, 3, local_dt)
        data_weights = np.full(
            local_count,
            profile.minimum_jerk_position_weight,
            dtype=np.float64,
        )
        data_weights[0] *= 500.0
        data_weights[-1] *= 500.0
        system = np.diag(data_weights)
        if len(d1):
            system += profile.minimum_jerk_velocity_weight * (d1.T @ d1)
        if len(d2):
            system += profile.minimum_jerk_acceleration_weight * (d2.T @ d2)
        if len(d3):
            system += profile.minimum_jerk_jerk_weight * (d3.T @ d3)
        system += np.eye(local_count, dtype=np.float64) * 1e-8
        right = data_weights * normalized
        try:
            solved = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(system, right, rcond=None)[0]
        result[start : end + 1] = np.clip(
            solved * crop_width,
            minimum_center,
            maximum_center,
        )

    # The optimization is soft. This projection gives the renderer hard physical
    # limits while retaining the lookahead that is already present in the solved path.
    local_ids = np.zeros(count, dtype=np.int32)
    limiter_profile = Profile(
        name="minimum_jerk_limiter",
        description="",
        dead_zone_fraction=0.0,
        max_speed_crop_widths_per_second=profile.max_speed_crop_widths_per_second,
        max_acceleration_crop_widths_per_second2=profile.max_acceleration_crop_widths_per_second2,
        max_jerk_crop_widths_per_second3=profile.max_jerk_crop_widths_per_second3,
    )
    return _stabilize_centers(
        result.astype(np.float32),
        times,
        local_ids,
        frame_width,
        crop_width,
        limiter_profile,
    ).astype(np.float64)


def _minimum_jerk_centers(
    desired_centers: NDArray[np.float32],
    times: NDArray[np.float64],
    shot_ids: NDArray[np.int32],
    frame_width: int,
    crop_width: int,
    profile: Profile,
) -> NDArray[np.float32]:
    if len(desired_centers) == 0:
        return desired_centers
    result = np.empty_like(desired_centers, dtype=np.float32)
    for shot_id in np.unique(shot_ids):
        indices = np.flatnonzero(shot_ids == shot_id)
        result[indices] = _minimum_jerk_shot(
            desired_centers[indices].astype(np.float64),
            times[indices],
            crop_width,
            frame_width,
            profile,
        ).astype(np.float32)
    return result


def _profile_candidate_coverage(
    profile_values: list[float],
    candidate_start: FloatArray,
    candidate_end: FloatArray,
    frame_span: int,
) -> FloatArray | None:
    if not profile_values:
        return None
    masses = np.asarray(profile_values, dtype=np.float32)
    total = float(masses.sum())
    if total <= 1e-8:
        return None
    masses /= total
    bin_count = len(masses)
    bin_width = frame_span / max(bin_count, 1)
    bin_start = np.arange(bin_count, dtype=np.float32) * bin_width
    bin_end = bin_start + bin_width
    overlap = np.maximum(
        0.0,
        np.minimum(candidate_end[:, None], bin_end[None, :])
        - np.maximum(candidate_start[:, None], bin_start[None, :]),
    )
    fraction = overlap / max(bin_width, 1e-6)
    return np.clip(fraction @ masses, 0.0, 1.0).astype(np.float32)


def _target_observations(
    tracks: list[TargetTrack],
    time_seconds: float,
    sample_fps: float,
) -> list[tuple[TargetTrack, TargetObservation]]:
    tolerance = max(0.08, 0.72 / max(sample_fps, 1e-6))
    visible: list[tuple[TargetTrack, TargetObservation]] = []
    for track in tracks:
        observation = track.nearest(time_seconds, tolerance=tolerance)
        if observation is None or not observation.visible or observation.confidence < 0.28:
            continue
        visible.append((track, observation))
    return visible


def _target_frame_signal(
    bundle: AnalysisBundle,
    frame_index: int,
    profile: Profile,
    tracks: list[TargetTrack],
) -> tuple[FloatArray, FloatArray, float, int]:
    frame = bundle.frames[frame_index]
    axis = _active_axis(bundle)
    frame_span = _frame_span(bundle)
    crop_span = _crop_span(frame)
    candidate_start = frame.x_positions.astype(np.float32)
    candidate_end = candidate_start + float(crop_span)
    candidate_centers = (candidate_start + candidate_end) / 2.0
    visible = _target_observations(tracks, frame.time, bundle.sample_fps)
    if not visible:
        zeros = np.zeros(len(candidate_start), dtype=np.float32)
        return zeros, zeros, 0.0, 0

    score_total = np.zeros(len(candidate_start), dtype=np.float32)
    coverage_total = np.zeros(len(candidate_start), dtype=np.float32)
    total_weight = 0.0
    required_coverages: list[FloatArray] = []
    confidences: list[float] = []

    for track, observation in visible:
        center_norm, extent_norm = _observation_axis_values(observation, axis)
        center = center_norm * frame_span
        if track.target.kind == "point":
            span = max(
                float(crop_span) * profile.target_context_fraction,
                float(extent_norm * frame_span) * 3.0,
                4.0,
            )
        else:
            span = max(
                float(extent_norm * frame_span),
                float(crop_span) * 0.18,
                4.0,
            )
            span += float(crop_span) * 0.08
        span = min(float(frame_span), span)
        target_start = float(np.clip(center - span / 2.0, 0.0, frame_span - span))
        target_end = target_start + span

        overlap = np.maximum(
            0.0,
            np.minimum(candidate_end, target_end) - np.maximum(candidate_start, target_start),
        )
        context_coverage = np.clip(overlap / max(span, 1.0), 0.0, 1.0).astype(np.float32)
        mask_profile = observation.mask_y_profile if axis == "y" else observation.mask_x_profile
        mask_coverage = _profile_candidate_coverage(
            mask_profile,
            candidate_start,
            candidate_end,
            frame_span,
        )
        if mask_coverage is not None and observation.mask_confidence > 0.05:
            mask_weight = float(np.clip(0.48 + 0.42 * observation.mask_confidence, 0.48, 0.90))
            coverage = (
                mask_weight * mask_coverage + (1.0 - mask_weight) * context_coverage
            ).astype(np.float32)
        else:
            coverage = context_coverage
        center_distance = np.abs(candidate_centers - center)
        center_scale = max(float(crop_span) * 0.34, 1.0)
        centering = np.exp(-0.5 * np.square(center_distance / center_scale)).astype(np.float32)
        edge_distance = np.minimum(center - candidate_start, candidate_end - center)
        margin = np.clip(edge_distance / max(float(crop_span) * 0.18, 1.0), 0.0, 1.0)
        score = (0.54 * coverage + 0.28 * centering + 0.18 * margin).astype(np.float32)

        confidence = float(np.clip(observation.confidence, 0.0, 1.0))
        priority = float(np.clip(observation.priority, 0.05, 1.0))
        role_weight = 1.35 if track.target.role == "keep_visible" else 1.0
        weight = max(0.05, confidence * priority * role_weight)
        score_total += score * weight
        coverage_total += coverage * weight
        total_weight += weight
        confidences.append(confidence)
        if track.target.role == "keep_visible":
            required_coverages.append(coverage)

    score_total /= max(total_weight, 1e-6)
    coverage_total /= max(total_weight, 1e-6)
    if required_coverages:
        required_min = np.min(np.stack(required_coverages), axis=0).astype(np.float32)
        score_total = 0.68 * score_total + 0.32 * required_min
        coverage_total = np.minimum(coverage_total, 0.55 + 0.45 * required_min)
        score_total -= np.maximum(0.0, 0.86 - required_min) * 0.82

    reliability = float(np.mean(confidences)) if confidences else 0.0
    return score_total.astype(np.float32), coverage_total.astype(np.float32), reliability, len(visible)

def _profile_frame_scores(
    bundle: AnalysisBundle,
    profile: Profile,
    target_tracks: list[TargetTrack],
) -> tuple[list[FloatArray], list[FloatArray], list[FloatArray], list[float], list[int]]:
    scores: list[FloatArray] = []
    coverages: list[FloatArray] = []
    target_coverages: list[FloatArray] = []
    target_reliabilities: list[float] = []
    target_counts: list[int] = []

    for index, frame in enumerate(bundle.frames):
        base_score, base_coverage = combine_signals(
            x_positions=frame.x_positions,
            crop_width=_crop_span(frame),
            frame_width=_frame_span(bundle),
            weights=profile.signal_weights,
            coverage=frame.signal_coverage,
            centering=frame.signal_centering,
            boundary=frame.signal_boundary,
            reliability=frame.signal_reliability,
        )
        target_score, target_coverage, reliability, count = _target_frame_signal(
            bundle,
            index,
            profile,
            target_tracks,
        )
        if profile.target_weight > 0.0 and count > 0:
            effective_weight = float(np.clip(profile.target_weight * (0.45 + 0.55 * reliability), 0.0, 1.0))
            if profile.target_only:
                combined_score = target_score
                combined_coverage = target_coverage
            else:
                combined_score = (1.0 - effective_weight) * base_score + effective_weight * target_score
                combined_coverage = (
                    (1.0 - effective_weight) * base_coverage
                    + effective_weight * target_coverage
                )
        else:
            combined_score = base_score
            combined_coverage = base_coverage

        scores.append(combined_score.astype(np.float32))
        coverages.append(combined_coverage.astype(np.float32))
        target_coverages.append(target_coverage.astype(np.float32))
        target_reliabilities.append(reliability)
        target_counts.append(count)
    return scores, coverages, target_coverages, target_reliabilities, target_counts


def _true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.append(mask, False)):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def _build_fallback_mask(
    bundle: AnalysisBundle,
    profile: Profile,
    profile_coverages: list[FloatArray],
    shot_ids: NDArray[np.int32],
    target_coverages: list[FloatArray] | None = None,
    target_reliabilities: list[float] | None = None,
    target_counts: list[int] | None = None,
) -> NDArray[np.bool_]:
    """Find sustained periods where no single target-aspect crop preserves enough evidence."""

    raw = np.zeros(len(bundle.frames), dtype=bool)
    for index, frame in enumerate(bundle.frames):
        best_coverage = float(np.max(profile_coverages[index]))
        motion_reliability = float(frame.signal_reliability.get("motion", 0.0))
        severely_distributed = best_coverage < profile.fallback_coverage_threshold
        static_and_distributed = (
            motion_reliability < profile.fallback_motion_reliability_threshold
            and best_coverage < profile.fallback_static_coverage_threshold
        )
        target_infeasible = False
        if (
            target_coverages is not None
            and target_reliabilities is not None
            and target_counts is not None
            and target_counts[index] > 0
            and target_reliabilities[index] >= 0.38
        ):
            target_infeasible = float(np.max(target_coverages[index])) < profile.target_min_coverage

        text_infeasible = False
        if (
            frame.text_regions
            and _text_layout_confidence(frame) >= 0.30
            and "text_complete" in frame.signal_coverage
        ):
            best_text_integrity = float(np.max(frame.signal_coverage["text_complete"]))
            # Only treat sustained, meaningful text as a hard constraint. A single tiny
            # label should not force a layout change, while a title, menu, code block,
            # or UI panel should never be sliced through silently.
            meaningful = sum(
                1
                for region in frame.text_regions
                if region.score >= 0.48 and region.width * region.height >= 0.00055
            )
            text_infeasible = (
                meaningful > 0
                and best_text_integrity < profile.text_integrity_threshold
            )
        raw[index] = (
            severely_distributed
            or static_and_distributed
            or target_infeasible
            or text_infeasible
        )

    result = np.zeros_like(raw)
    sample_fps = max(float(bundle.sample_fps), 1e-6)
    merge_gap = max(0, int(round(profile.fallback_merge_gap_seconds * sample_fps)))
    minimum_length = max(1, int(np.ceil(profile.fallback_min_seconds * sample_fps)))
    padding = max(0, int(round(profile.fallback_padding_seconds * sample_fps)))

    for shot_id in np.unique(shot_ids):
        indices = np.flatnonzero(shot_ids == shot_id)
        if len(indices) == 0:
            continue
        local = raw[indices].copy()

        runs = _true_runs(local)
        for (_, previous_end), (next_start, _) in zip(runs, runs[1:], strict=False):
            if next_start - previous_end - 1 <= merge_gap:
                local[previous_end + 1 : next_start] = True

        for start, end in _true_runs(local):
            run_length = end - start + 1
            dominates_short_shot = run_length >= int(np.ceil(len(local) * 0.60))
            if run_length < minimum_length and not dominates_short_shot:
                local[start : end + 1] = False

        padded = local.copy()
        for start, end in _true_runs(local):
            padded[max(0, start - padding) : min(len(local), end + padding + 1)] = True
        result[indices] = padded

    return result



def _deduplicate_semantic_regions(frame) -> list:
    """Return high-confidence person regions with simple overlap suppression."""

    regions = sorted(
        (region for region in frame.semantic_regions if region.score >= 0.42),
        key=lambda item: item.score,
        reverse=True,
    )
    kept = []
    for region in regions:
        x0, y0 = region.x, region.y
        x1, y1 = region.x + region.width, region.y + region.height
        duplicate = False
        for existing in kept:
            ex0, ey0 = existing.x, existing.y
            ex1, ey1 = existing.x + existing.width, existing.y + existing.height
            overlap_w = max(0.0, min(x1, ex1) - max(x0, ex0))
            overlap_h = max(0.0, min(y1, ey1) - max(y0, ey0))
            intersection = overlap_w * overlap_h
            union = region.width * region.height + existing.width * existing.height - intersection
            if intersection / max(union, 1e-6) >= 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
    return kept


def _cluster_centers(
    centers_and_weights: list[tuple[float, float]],
    crop_fraction: float,
) -> tuple[float, float] | None:
    """Split horizontally distributed targets into two stable visual groups."""

    if len(centers_and_weights) < 2:
        return None
    ordered = sorted(centers_and_weights)
    centers = np.asarray([item[0] for item in ordered], dtype=np.float32)
    weights = np.asarray([max(0.05, item[1]) for item in ordered], dtype=np.float32)
    span = float(centers[-1] - centers[0])
    if span < max(0.22, crop_fraction * 0.62):
        return None

    gaps = np.diff(centers)
    split_index = int(np.argmax(gaps)) + 1
    if float(gaps[split_index - 1]) < max(0.09, crop_fraction * 0.22):
        # Evenly spaced panels still benefit from a two-up composition. Split at
        # the weighted median instead of requiring one unusually large gap.
        split_index = max(1, min(len(centers) - 1, len(centers) // 2))

    left = float(np.average(centers[:split_index], weights=weights[:split_index]))
    right = float(np.average(centers[split_index:], weights=weights[split_index:]))
    if right - left < max(0.18, crop_fraction * 0.48):
        return None
    return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))


def _semantic_split_centers(
    frame,
    crop_fraction: float,
    axis: str = "x",
) -> tuple[float, float] | None:
    regions = _deduplicate_semantic_regions(frame)
    values = []
    for region in regions:
        center, extent = _region_axis_values(region, axis)
        values.append((center, region.score * max(extent, 0.08)))
    return _cluster_centers(values, crop_fraction)

def _text_layout_confidence(frame) -> float:
    """Confidence that detected boxes are meaningful text/UI layout evidence.

    Learned PP-OCR detections can drive hard layout decisions directly. The
    classical fallback must pass a stricter geometry check so foliage, road
    markings, and other repeated edges cannot create a split-screen layout.
    """

    backend = str(getattr(frame, "text_backend", ""))
    reliability = float(frame.signal_reliability.get("text_complete", 0.0))
    if "dnn_ppocr" in backend:
        return reliability
    substantial = [
        region
        for region in frame.text_regions
        if region.score >= 0.55
        and region.width >= 0.04
        and region.height >= 0.018
        and region.width * region.height >= 0.0010
        and region.y > 0.012
        and region.y + region.height < 0.988
    ]
    if len(substantial) < 4:
        return min(reliability, 0.12)
    area = sum(region.width * region.height for region in substantial)
    median_width = float(np.median([region.width for region in substantial]))
    if area < 0.035 or median_width < 0.055:
        return min(reliability, 0.18)
    return reliability


def _text_split_centers(
    frame,
    crop_fraction: float,
    minimum_regions: int,
    integrity_threshold: float,
    axis: str = "x",
) -> tuple[float, float] | None:
    """Find two separated text/UI groups on the active crop axis."""

    if _text_layout_confidence(frame) < 0.30:
        return None
    regions = [
        region
        for region in frame.text_regions
        if region.score >= 0.45 and region.width * region.height >= 0.00045
    ]
    if len(regions) < minimum_regions:
        return None
    if "text_complete" in frame.signal_coverage:
        if float(np.max(frame.signal_coverage["text_complete"])) >= integrity_threshold:
            return None

    weighted = []
    for region in regions:
        center, _ = _region_axis_values(region, axis)
        weighted.append(
            (
                center,
                region.score
                * (0.45 + np.sqrt(max(region.width * region.height, 1e-6)) * 4.0),
            )
        )
    candidate = _cluster_centers(weighted, crop_fraction)
    if candidate is None:
        return None
    first, second = candidate
    midpoint = (first + second) / 2.0
    first_count = 0
    for region in regions:
        center, _ = _region_axis_values(region, axis)
        first_count += int(center <= midpoint)
    second_count = len(regions) - first_count
    if min(first_count, second_count) < 2:
        return None
    return candidate

def _dominant_text_requires_full_context(
    frame,
    crop_fraction: float,
    axis: str = "x",
) -> bool:
    """Return true when one prominent text group cannot fit the active crop span."""

    all_regions = list(frame.text_regions)
    backend = str(getattr(frame, "text_backend", ""))
    if "classical" in backend and len(all_regions) > 5:
        return False

    def start(region) -> float:
        return region.y if axis == "y" else region.x

    def extent(region) -> float:
        return region.height if axis == "y" else region.width

    def orth_start(region) -> float:
        return region.x if axis == "y" else region.y

    def orth_extent(region) -> float:
        return region.width if axis == "y" else region.height

    regions = [
        region
        for region in all_regions
        if region.score >= 0.55 and orth_extent(region) >= 0.045
    ]
    for region in regions:
        active_extent = extent(region)
        perpendicular = orth_extent(region)
        if (
            active_extent >= crop_fraction * 0.94
            and region.width * region.height >= 0.0020
            and perpendicular <= 0.13
            and active_extent / max(perpendicular, 1e-6) >= 2.4
        ):
            return True

    # Merge adjacent boxes that lie on the same line (x crop) or in the same
    # column (y crop), because OCR frequently splits one title into pieces.
    ordered = sorted(regions, key=lambda item: (orth_start(item), start(item)))
    for index, first in enumerate(ordered):
        group = [first]
        first_o0 = orth_start(first)
        first_o1 = first_o0 + orth_extent(first)
        for other in ordered[index + 1 :]:
            other_o0 = orth_start(other)
            other_o1 = other_o0 + orth_extent(other)
            overlap = max(0.0, min(first_o1, other_o1) - max(first_o0, other_o0))
            if overlap / max(min(orth_extent(first), orth_extent(other)), 1e-6) < 0.36:
                continue
            first_center = first_o0 + orth_extent(first) / 2.0
            other_center = other_o0 + orth_extent(other) / 2.0
            if abs(other_center - first_center) > max(orth_extent(first), orth_extent(other)) * 0.75:
                continue
            current_end = max(start(item) + extent(item) for item in group)
            gap = max(0.0, start(other) - current_end)
            maximum_gap = max(0.035, max(orth_extent(first), orth_extent(other)) * 1.8)
            if gap > maximum_gap:
                continue
            group.append(other)
        if len(group) < 2:
            continue
        union_start = min(start(item) for item in group)
        union_end = max(start(item) + extent(item) for item in group)
        union_extent = union_end - union_start
        perpendicular = max(orth_extent(item) for item in group)
        if (
            union_extent >= crop_fraction * 0.94
            and 0.045 <= perpendicular <= 0.13
            and union_extent / max(perpendicular, 1e-6) >= 2.8
        ):
            return True
    return False

def _target_split_centers(
    tracks: list[TargetTrack],
    time_seconds: float,
    sample_fps: float,
    crop_fraction: float,
    axis: str = "x",
) -> tuple[float, float] | None:
    visible = _target_observations(tracks, time_seconds, sample_fps)
    values = []
    for track, observation in visible:
        center, _ = _observation_axis_values(observation, axis)
        values.append(
            (
                center,
                observation.confidence
                * (1.25 if track.target.role == "keep_visible" else 1.0),
            )
        )
    return _cluster_centers(values, crop_fraction)

def _smooth_layout_holes(
    modes: list[str],
    primary: list[float | None],
    secondary: list[float | None],
    primary_fraction: list[float],
    bundle: AnalysisBundle,
    minimum_seconds: float,
) -> None:
    """Fill short layout holes only when the surrounding layout agrees.

    This removes split/crop/split and crop/detail/crop flashes without overriding a
    genuine sustained layout change.
    """

    minimum = max(1, int(round(minimum_seconds * max(bundle.sample_fps, 1e-6))))
    for shot_id in {frame.shot_id for frame in bundle.frames}:
        indices = [index for index, frame in enumerate(bundle.frames) if frame.shot_id == shot_id]
        if len(indices) < 3:
            continue
        local_modes = [modes[index] for index in indices]
        runs: list[tuple[int, int, str]] = []
        start = 0
        for local_index in range(1, len(local_modes) + 1):
            if local_index == len(local_modes) or local_modes[local_index] != local_modes[start]:
                runs.append((start, local_index - 1, local_modes[start]))
                start = local_index
        for run_index in range(1, len(runs) - 1):
            start, end, mode = runs[run_index]
            previous_mode = runs[run_index - 1][2]
            next_mode = runs[run_index + 1][2]
            if previous_mode != next_mode or end - start + 1 > minimum:
                continue
            source_candidates = [
                indices[runs[run_index - 1][1]],
                indices[runs[run_index + 1][0]],
            ]
            for local_index in range(start, end + 1):
                global_index = indices[local_index]
                modes[global_index] = previous_mode
                if previous_mode == "split":
                    nearest = min(
                        source_candidates,
                        key=lambda item: abs(bundle.frames[item].time - bundle.frames[global_index].time),
                    )
                    primary[global_index] = primary[nearest]
                    secondary[global_index] = secondary[nearest]
                    primary_fraction[global_index] = primary_fraction[nearest]



def _speaker_center(frame, axis: str) -> tuple[float, float] | None:
    candidates = [
        region
        for region in frame.semantic_regions
        if region.speaker_score >= 0.08 and region.speaker_confidence >= 0.10
    ]
    if not candidates:
        return None
    winner = max(candidates, key=lambda region: region.speaker_score * region.speaker_confidence)
    center, _ = _region_axis_values(winner, axis)
    confidence = speaker_region_strength(winner)
    return center, confidence


def _shot_framing_groups(bundle: AnalysisBundle) -> tuple[dict[int, int], dict[str, object]]:
    """Cluster visually similar shots so repeated camera angles share framing anchors."""

    shot_indices: dict[int, list[int]] = defaultdict(list)
    for index, frame in enumerate(bundle.frames):
        shot_indices[int(frame.shot_id)].append(index)
    shot_ids = sorted(shot_indices)
    descriptors: dict[int, np.ndarray] = {}
    signatures: dict[int, tuple[int, int]] = {}
    for shot_id, indices in shot_indices.items():
        values = [bundle.frames[index].appearance_descriptor for index in indices]
        values = [value for value in values if value.size]
        if values:
            descriptor = np.mean(np.stack(values), axis=0).astype(np.float32)
            descriptor /= max(float(np.linalg.norm(descriptor)), 1e-6)
            descriptors[shot_id] = descriptor
        people_count = int(round(np.median([len(bundle.frames[index].semantic_regions) for index in indices])))
        text_count = int(round(np.median([len(bundle.frames[index].text_regions) for index in indices])))
        text_bucket = 0 if text_count == 0 else 1 if text_count <= 3 else 2
        signatures[shot_id] = (people_count, text_bucket)

    parent = {shot_id: shot_id for shot_id in shot_ids}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    pair_count = 0
    for offset, first in enumerate(shot_ids):
        first_descriptor = descriptors.get(first)
        if first_descriptor is None:
            continue
        for second in shot_ids[offset + 1 :]:
            second_descriptor = descriptors.get(second)
            if second_descriptor is None:
                continue
            first_people, first_text = signatures[first]
            second_people, second_text = signatures[second]
            if abs(first_people - second_people) > 1 or first_text != second_text:
                continue
            similarity = float(np.dot(first_descriptor, second_descriptor))
            if similarity >= 0.90:
                union(first, second)
                pair_count += 1

    root_to_group: dict[int, int] = {}
    groups: dict[int, int] = {}
    for shot_id in shot_ids:
        root = find(shot_id)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        groups[shot_id] = root_to_group[root]
    group_sizes: dict[int, int] = defaultdict(int)
    for group in groups.values():
        group_sizes[group] += 1
    return groups, {
        "group_count": len(group_sizes),
        "repeated_group_count": sum(size >= 2 for size in group_sizes.values()),
        "matched_pairs": pair_count,
        "group_sizes": dict(group_sizes),
    }


def _apply_consistent_framing(
    bundle: AnalysisBundle,
    desired_centers: NDArray[np.float32],
    *,
    crop_span: int,
    frame_span: int,
    enabled: bool,
) -> tuple[NDArray[np.float32], dict[int, int], dict[str, object]]:
    groups, metadata = _shot_framing_groups(bundle)
    result = desired_centers.astype(np.float32).copy()
    metadata = dict(metadata)
    metadata.update({"enabled": bool(enabled), "locked_shots": 0})
    if not enabled or len(groups) < 2:
        return result, groups, metadata

    shot_indices: dict[int, list[int]] = defaultdict(list)
    for index, frame in enumerate(bundle.frames):
        shot_indices[int(frame.shot_id)].append(index)
    grouped_shots: dict[int, list[int]] = defaultdict(list)
    for shot_id, group in groups.items():
        grouped_shots[group].append(shot_id)

    # Repeated camera angles often drift by half a viewport because a gesture,
    # subtitle, or bright object briefly wins the local score. A user-selected
    # target is handled separately and disables this lock in build_plan.
    maximum_shift = crop_span * 0.85
    locked = 0
    for shot_ids in grouped_shots.values():
        if len(shot_ids) < 2:
            continue
        # The visible inconsistency happens at the cut, so lock the opening
        # framing of matching shots rather than only their whole-shot median.
        opening_samples = max(1, int(round(bundle.sample_fps * 0.8)))
        shot_anchors = {}
        for shot_id in shot_ids:
            indices = np.asarray(shot_indices[shot_id], dtype=np.int32)
            shot_anchors[shot_id] = float(np.median(result[indices[:opening_samples]]))
        common = float(np.median(list(shot_anchors.values())))
        for shot_id in shot_ids:
            delta = common - shot_anchors[shot_id]
            if abs(delta) > maximum_shift:
                continue
            indices = np.asarray(shot_indices[shot_id], dtype=np.int32)
            result[indices] = np.clip(
                result[indices] + 0.90 * delta,
                crop_span / 2.0,
                frame_span - crop_span / 2.0,
            )
            locked += 1
    metadata["locked_shots"] = locked
    return result, groups, metadata


def _correction_frame_indices(
    bundle: AnalysisBundle,
    correction: FramingCorrection,
    shot_groups: dict[int, int],
) -> np.ndarray:
    times = np.asarray([frame.time for frame in bundle.frames], dtype=np.float64)
    nearest = int(np.argmin(np.abs(times - correction.time)))
    shot_id = int(bundle.frames[nearest].shot_id)
    if correction.scope == "all":
        return np.arange(len(bundle.frames), dtype=np.int32)
    if correction.scope == "matching":
        group = shot_groups.get(shot_id)
        return np.asarray(
            [
                index
                for index, frame in enumerate(bundle.frames)
                if shot_groups.get(int(frame.shot_id)) == group
            ],
            dtype=np.int32,
        )
    return np.asarray(
        [index for index, frame in enumerate(bundle.frames) if int(frame.shot_id) == shot_id],
        dtype=np.int32,
    )


def _apply_center_corrections(
    bundle: AnalysisBundle,
    desired_centers: NDArray[np.float32],
    corrections: list[FramingCorrection],
    shot_groups: dict[int, int],
    axis: str,
    frame_span: int,
    crop_span: int,
) -> tuple[NDArray[np.float32], list[dict[str, object]]]:
    result = desired_centers.astype(np.float32).copy()
    applied: list[dict[str, object]] = []
    for correction in corrections:
        indices = _correction_frame_indices(bundle, correction, shot_groups)
        if indices.size == 0:
            continue
        center_norm = correction.center_y if axis == "y" else correction.center_x
        center = float(np.clip(center_norm * frame_span, crop_span / 2.0, frame_span - crop_span / 2.0))
        result[indices] = center
        applied.append({
            "time": correction.time,
            "scope": correction.scope,
            "mode": correction.mode,
            "affected_samples": int(indices.size),
            "center": center_norm,
        })
    return result, applied


def _choose_layouts(
    bundle: AnalysisBundle,
    profile: Profile,
    tracks: list[TargetTrack],
    fallback_mask: NDArray[np.bool_],
    profile_coverages: list[FloatArray],
    target_counts: list[int],
) -> tuple[list[str], list[float | None], list[float | None], list[float]]:
    """Replace unsafe crops with layouts appropriate to the active crop axis.

    When the source is wider than the output, horizontally separated regions are
    stacked top/bottom. When the source is taller than the output, vertically
    separated regions are placed left/right by the renderer.
    """

    modes = ["crop"] * len(bundle.frames)
    primary: list[float | None] = [None] * len(bundle.frames)
    secondary: list[float | None] = [None] * len(bundle.frames)
    primary_fraction: list[float] = [0.5] * len(bundle.frames)
    if not profile.allow_fallback:
        return modes, primary, secondary, primary_fraction

    axis = _active_axis(bundle)
    frame_span = _frame_span(bundle)
    crop_fraction = _crop_span(bundle.frames[0]) / max(frame_span, 1)
    carry_seconds = max(0.45, min(1.0, 2.5 / max(bundle.sample_fps, 1e-6)))
    last_split: tuple[float, float] | None = None
    last_split_time = -1e9
    last_shot = -1

    for index, frame in enumerate(bundle.frames):
        if frame.shot_id != last_shot:
            last_split = None
            last_split_time = -1e9
            last_shot = frame.shot_id

        dominant_text = _dominant_text_requires_full_context(frame, crop_fraction, axis)
        if dominant_text:
            modes[index] = "fallback"
            continue

        split = None
        if profile.semantic_layouts and axis != "none":
            split = _target_split_centers(
                tracks,
                frame.time,
                bundle.sample_fps,
                crop_fraction,
                axis,
            )
            if split is None:
                split = _semantic_split_centers(frame, crop_fraction, axis)
            if split is None:
                split = _text_split_centers(
                    frame,
                    crop_fraction,
                    profile.text_split_min_regions,
                    profile.text_integrity_threshold,
                    axis,
                )
            if split is not None:
                last_split = split
                last_split_time = frame.time
            elif last_split is not None and frame.time - last_split_time <= carry_seconds:
                split = last_split

        if split is not None:
            modes[index] = "split"
            speaker = _speaker_center(frame, axis)
            first, second = split
            if speaker is not None:
                speaker_center, speaker_confidence = speaker
                if abs(speaker_center - second) < abs(speaker_center - first):
                    first, second = second, first
                primary_fraction[index] = 0.5 + 0.14 * speaker_confidence
            primary[index] = first
            secondary[index] = second
            continue

        if not bool(fallback_mask[index]):
            modes[index] = "crop"
            continue

        best_coverage = float(np.max(profile_coverages[index]))
        has_detail = (
            target_counts[index] > 0
            or bool(_deduplicate_semantic_regions(frame))
            or _text_layout_confidence(frame) >= 0.30
            or best_coverage >= 0.20
        )
        modes[index] = "context_detail" if profile.semantic_layouts and has_detail else "fallback"

    # Carry a repeatedly observed two-group layout through the shot. The panel crop
    # doubles the active span because each panel receives roughly half of the output
    # along the orthogonal axis.
    shot_groups: dict[int, list[int]] = defaultdict(list)
    for index, frame in enumerate(bundle.frames):
        shot_groups[frame.shot_id].append(index)
    for indices in shot_groups.values():
        split_indices = [index for index in indices if modes[index] == "split"]
        minimum = max(3, int(np.ceil(len(indices) * 0.15)))
        if len(split_indices) < minimum:
            continue
        primary_values = [primary[index] for index in split_indices if primary[index] is not None]
        secondary_values = [
            secondary[index] for index in split_indices if secondary[index] is not None
        ]
        if not primary_values or not secondary_values:
            continue
        split_axis = choose_layout_split_axis(
            bundle.analysis_width,
            bundle.analysis_height,
            bundle.target_width,
            bundle.target_height,
            panel_fraction=0.5,
        )
        split_panel_fraction = layout_panel_crop_fraction(
            bundle.analysis_width,
            bundle.analysis_height,
            bundle.target_width,
            bundle.target_height,
            active_axis=axis,
            split_axis=split_axis,
            panel_fraction=0.5,
        )
        panel_minimum_center = split_panel_fraction / 2.0
        panel_maximum_center = 1.0 - panel_minimum_center
        stable_primary = float(
            np.clip(
                np.percentile(primary_values, 20.0),
                panel_minimum_center,
                panel_maximum_center,
            )
        )
        stable_secondary = float(
            np.clip(
                np.percentile(secondary_values, 80.0),
                panel_minimum_center,
                panel_maximum_center,
            )
        )
        if stable_secondary <= stable_primary:
            stable_primary = panel_minimum_center
            stable_secondary = panel_maximum_center
        for index in indices:
            if _dominant_text_requires_full_context(
                bundle.frames[index], crop_fraction, axis
            ):
                modes[index] = "fallback"
                primary[index] = None
                secondary[index] = None
                continue
            modes[index] = "split"
            first, second = stable_primary, stable_secondary
            speaker = _speaker_center(bundle.frames[index], axis)
            if speaker is not None:
                speaker_center, speaker_confidence = speaker
                if abs(speaker_center - second) < abs(speaker_center - first):
                    first, second = second, first
                primary_fraction[index] = 0.5 + 0.14 * speaker_confidence
            primary[index] = first
            secondary[index] = second

    _smooth_layout_holes(
        modes,
        primary,
        secondary,
        primary_fraction,
        bundle,
        profile.minimum_layout_seconds,
    )
    return modes, primary, secondary, primary_fraction

def build_plan(
    bundle: AnalysisBundle,
    profile: Profile,
    target_tracks: list[TargetTrack] | None = None,
    framing_corrections: list[FramingCorrection] | None = None,
    *,
    consistent_framing: bool = True,
) -> CropPlan:
    tracks = target_tracks or []
    corrections = framing_corrections or []
    frame_count = len(bundle.frames)
    positions = bundle.frames[0].x_positions
    crop_span = _crop_span(bundle.frames[0])
    frame_span = _frame_span(bundle)
    axis = _active_axis(bundle)
    times = np.array([frame.time for frame in bundle.frames], dtype=np.float64)
    shot_ids = np.array([frame.shot_id for frame in bundle.frames], dtype=np.int32)

    if profile.name == "center":
        fixed_index = int(
            np.argmin(np.abs((positions + crop_span / 2.0) - frame_span / 2.0))
        )
        chosen_indices = np.full(frame_count, fixed_index, dtype=np.int32)
        profile_scores = [frame.evaluation_score for frame in bundle.frames]
        profile_coverages = [frame.evaluation_coverage for frame in bundle.frames]
        target_coverages = [np.zeros_like(frame.evaluation_coverage) for frame in bundle.frames]
        target_reliabilities = [0.0] * frame_count
        target_counts = [0] * frame_count
    else:
        (
            profile_scores,
            profile_coverages,
            target_coverages,
            target_reliabilities,
            target_counts,
        ) = _profile_frame_scores(bundle, profile, tracks)
        chosen_indices = np.zeros(frame_count, dtype=np.int32)
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index, shot_id in enumerate(shot_ids):
            grouped_indices[int(shot_id)].append(index)

        for indices in grouped_indices.values():
            matrix = np.stack([profile_scores[index] for index in indices]).astype(np.float32)
            if profile.temporal:
                local_path = _dynamic_programming_path(
                    matrix,
                    positions,
                    crop_span,
                    movement_penalty=profile.movement_penalty,
                    maximum_transition_fraction=profile.max_transition_crop_fraction,
                )
            else:
                local_path = np.argmax(matrix, axis=1).astype(np.int32)
            chosen_indices[np.array(indices, dtype=np.int32)] = local_path

    desired_centers = positions[chosen_indices].astype(np.float32) + crop_span / 2.0
    framing_lock_metadata: dict[str, object] = {"enabled": False}
    shot_groups: dict[int, int] = {int(frame.shot_id): int(frame.shot_id) for frame in bundle.frames}
    applied_corrections: list[dict[str, object]] = []
    if profile.name != "center":
        desired_centers, shot_groups, framing_lock_metadata = _apply_consistent_framing(
            bundle,
            desired_centers,
            crop_span=crop_span,
            frame_span=frame_span,
            enabled=consistent_framing and not tracks,
        )
        if tracks and consistent_framing:
            framing_lock_metadata["disabled_reason"] = "selected_target_takes_priority"
        desired_centers, applied_corrections = _apply_center_corrections(
            bundle,
            desired_centers,
            corrections,
            shot_groups,
            axis,
            frame_span,
            crop_span,
        )
    if profile.stabilize:
        if profile.camera_solver == "minimum_jerk":
            centers = _minimum_jerk_centers(
                desired_centers,
                times,
                shot_ids,
                frame_span,
                crop_span,
                profile,
            )
        else:
            centers = _stabilize_centers(
                desired_centers,
                times,
                shot_ids,
                frame_span,
                crop_span,
                profile,
            )
    else:
        centers = desired_centers

    fallback_mask = np.zeros(frame_count, dtype=bool)
    if profile.allow_fallback:
        fallback_mask = _build_fallback_mask(
            bundle,
            profile,
            profile_coverages,
            shot_ids,
            target_coverages,
            target_reliabilities,
            target_counts,
        )

    layout_modes, layout_primary, layout_secondary, split_primary_fractions = _choose_layouts(
        bundle,
        profile,
        tracks,
        fallback_mask,
        profile_coverages,
        target_counts,
    )

    if profile.name != "center":
        for correction in corrections:
            if correction.mode == "auto":
                continue
            indices = _correction_frame_indices(bundle, correction, shot_groups)
            for index in indices:
                layout_modes[index] = correction.mode
                if correction.mode == "split":
                    if correction.secondary_center_x is not None or correction.secondary_center_y is not None:
                        secondary_norm = (
                            correction.secondary_center_y if axis == "y" else correction.secondary_center_x
                        )
                        layout_secondary[index] = secondary_norm
                    elif layout_secondary[index] is None:
                        active_center = correction.center_y if axis == "y" else correction.center_x
                        layout_secondary[index] = float(np.clip(1.0 - active_center, 0.0, 1.0))
                    layout_primary[index] = correction.center_y if axis == "y" else correction.center_x
                    split_primary_fractions[index] = 0.5
                elif correction.mode in {"crop", "context_detail", "fallback"}:
                    layout_primary[index] = None
                    layout_secondary[index] = None

    split_primary_pixels = centers.astype(np.float32).copy()
    split_secondary_pixels = centers.astype(np.float32).copy()
    for index, mode in enumerate(layout_modes):
        if mode == "split":
            if layout_primary[index] is not None:
                split_primary_pixels[index] = float(layout_primary[index]) * frame_span
            if layout_secondary[index] is not None:
                split_secondary_pixels[index] = float(layout_secondary[index]) * frame_span
    if any(mode == "split" for mode in layout_modes):
        split_profile = Profile(
            name="split_layout",
            description="",
            dead_zone_fraction=0.16,
            max_speed_crop_widths_per_second=1.2,
            max_acceleration_crop_widths_per_second2=3.5,
        )
        split_primary_pixels = _minimum_jerk_centers(
            split_primary_pixels, times, shot_ids, frame_span, crop_span, split_profile
        )
        split_secondary_pixels = _minimum_jerk_centers(
            split_secondary_pixels, times, shot_ids, frame_span, crop_span, split_profile
        )

    points: list[PlanPoint] = []
    for index, frame in enumerate(bundle.frames):
        chosen_index = int(
            np.argmin(np.abs((frame.x_positions + crop_span / 2.0) - centers[index]))
        )
        layout_mode = layout_modes[index]
        secondary_active: float | None = None
        point_center = float(centers[index])
        if layout_mode == "split":
            mode = "split"
            point_center = float(split_primary_pixels[index])
            secondary_active = float(
                np.clip(split_secondary_pixels[index] / frame_span, 0.0, 1.0)
            )
            coverage = max(float(profile_coverages[index][chosen_index]), 0.88)
        elif layout_mode == "context_detail":
            mode = "context_detail"
            coverage = max(float(profile_coverages[index][chosen_index]), 0.74)
        elif layout_mode == "fallback":
            mode = "fallback"
            coverage = 1.0
        else:
            coverage = float(profile_coverages[index][chosen_index])
            if index == 0 or frame.shot_id != bundle.frames[index - 1].shot_id:
                mode = "hold"
            else:
                movement = abs(float(centers[index] - centers[index - 1])) / max(crop_span, 1)
                mode = "hold" if movement < 0.015 else "pan"

        confidence = _local_confidence(profile_scores[index], chosen_index)
        if target_counts[index] > 0:
            confidence = float(
                np.clip(
                    0.56 * confidence + 0.44 * target_reliabilities[index],
                    0.0,
                    1.0,
                )
            )
        active_center_norm = float(np.clip(point_center / frame_span, 0.0, 1.0))
        center_x_norm, center_y_norm = _axis_center_xy(axis, active_center_norm)
        secondary_x: float | None = None
        secondary_y: float | None = None
        if secondary_active is not None:
            secondary_x, secondary_y = _axis_center_xy(axis, secondary_active)
        points.append(
            PlanPoint(
                frame_index=frame.frame_index,
                time=frame.time,
                shot_id=frame.shot_id,
                center_x_norm=center_x_norm,
                mode=mode,
                content_coverage=coverage,
                confidence=confidence,
                secondary_center_x_norm=secondary_x,
                layout_detail_fraction=profile.context_detail_fraction,
                center_y_norm=center_y_norm,
                secondary_center_y_norm=secondary_y,
                split_primary_fraction=float(split_primary_fractions[index]),
            )
        )

    plan = CropPlan(
        profile=profile.name,
        description=profile.description,
        source=bundle.video.path,
        source_width=bundle.video.width,
        source_height=bundle.video.height,
        source_fps=bundle.video.fps,
        target_width=bundle.target_width,
        target_height=bundle.target_height,
        sample_fps=bundle.sample_fps,
        points=points,
        crop_axis=axis,
        implementation={
            **opencv_runtime_info(),
            "opencv_runtime_major": int(cv.__version__.split(".")[0]),
            "analysis_width": bundle.analysis_width,
            "analysis_height": bundle.analysis_height,
            "target_aspect_ratio": bundle.target_width / max(bundle.target_height, 1),
            "crop_axis": axis,
            "analysis_crop_width": bundle.frames[0].crop_width,
            "analysis_crop_height": bundle.frames[0].crop_height or bundle.analysis_height,
            "candidate_count": len(positions),
            "signals": sorted(profile.signal_weights),
            "temporal_optimisation": profile.temporal,
            "camera_stabilisation": profile.stabilize,
            "camera_solver": profile.camera_solver if profile.stabilize else "none",
            "minimum_jerk_weights": {
                "position": profile.minimum_jerk_position_weight,
                "velocity": profile.minimum_jerk_velocity_weight,
                "acceleration": profile.minimum_jerk_acceleration_weight,
                "jerk": profile.minimum_jerk_jerk_weight,
            },
            "safe_fallback": profile.allow_fallback,
            "movement_penalty": profile.movement_penalty,
            "max_speed_viewports_per_second": profile.max_speed_crop_widths_per_second,
            "max_acceleration_viewports_per_second2": profile.max_acceleration_crop_widths_per_second2,
            "max_jerk_viewports_per_second3": profile.max_jerk_crop_widths_per_second3,
            "layout_transition_seconds": profile.layout_transition_seconds,
            "minimum_layout_seconds": profile.minimum_layout_seconds,
            "fallback_coverage_threshold": profile.fallback_coverage_threshold,
            "fallback_static_coverage_threshold": profile.fallback_static_coverage_threshold,
            "fallback_motion_reliability_threshold": profile.fallback_motion_reliability_threshold,
            "fallback_min_seconds": profile.fallback_min_seconds,
            "fallback_merge_gap_seconds": profile.fallback_merge_gap_seconds,
            "fallback_padding_seconds": profile.fallback_padding_seconds,
            "target_count": len(tracks),
            "target_weight": profile.target_weight,
            "target_context_fraction": profile.target_context_fraction,
            "target_min_coverage": profile.target_min_coverage,
            "semantic_layouts": profile.semantic_layouts,
            "consistent_framing": framing_lock_metadata,
            "framing_corrections": applied_corrections,
            "semantic_backends": sorted({frame.semantic_backend for frame in bundle.frames}),
            "context_detail_fraction": profile.context_detail_fraction,
            "text_integrity_threshold": profile.text_integrity_threshold,
            "text_backends": sorted({frame.text_backend for frame in bundle.frames}),
            "motion_backends": sorted({frame.motion_backend for frame in bundle.frames}),
            "depth_backends": sorted({frame.depth_backend for frame in bundle.frames}),
            "mask_backends": sorted({item.mask_backend for track in tracks for item in track.observations if item.mask_backend != "none"}),
            "opencv_components": {
                "video_io": "cv.VideoCapture and cv.VideoWriter",
                "shot_detection": "histogram/structure change metrics",
                "motion": "Shi-Tomasi, pyramidal Lucas-Kanade and RANSAC camera compensation, with adaptive Farneback only when sparse evidence is weak",
                "depth": "disabled in the production pipeline; Depth Anything remains an explicit experiment",
                "segmentation": "box guidance in production; EfficientSAM and GrabCut masks remain explicit experiments",
                "camera_path": "shot-level quadratic minimum-jerk solve followed by hard speed, acceleration, and jerk limits",
                "saliency": "spectral residual saliency",
                "structure": "Sobel gradients and morphology",
                "text": "OpenCV DNN PP-OCR with classical morphology fallback",
                "people": "OpenCV DNN person detector and YuNet face detector",
                "active_speaker": "YuNet face regions, Farneback mouth motion, and lightweight audio activity gating",
                "framing_consistency": "shot appearance descriptors and shared opening anchors",
                "manual_corrections": "cached-analysis crop and layout constraints",
                "tracking": "template matching, optical flow, ORB/RANSAC, CSRT/KCF and Kalman filtering; ViTTrack remains an explicit experiment",
                "geometry": "integral-image candidate scoring and axis-generic crop planning",
            },
        },
    )
    plan.metrics = calculate_plan_metrics(bundle, plan, tracks, profile)
    return plan

def _crop_interval(center: float, width: float) -> tuple[float, float]:
    width = float(np.clip(width, 0.0, 1.0))
    x0 = float(np.clip(center - width / 2.0, 0.0, 1.0 - width))
    return x0, x0 + width


def _detail_intervals(point: PlanPoint, plan: CropPlan) -> list[tuple[float, float]]:
    geometry = crop_geometry(
        plan.source_width,
        plan.source_height,
        plan.target_width,
        plan.target_height,
    )
    axis = plan.crop_axis if plan.crop_axis in {"x", "y", "none"} else geometry.axis
    if point.mode == "fallback":
        return []

    center = _point_axis_center(point, axis)
    if point.mode == "context_detail":
        split_axis = choose_layout_split_axis(
            plan.source_width,
            plan.source_height,
            plan.target_width,
            plan.target_height,
            panel_fraction=point.layout_detail_fraction,
        )
        detail_span = layout_panel_crop_fraction(
            plan.source_width,
            plan.source_height,
            plan.target_width,
            plan.target_height,
            active_axis=axis,
            split_axis=split_axis,
            panel_fraction=point.layout_detail_fraction,
        )
        return [_crop_interval(center, detail_span)]

    if point.mode == "split":
        split_axis = choose_layout_split_axis(
            plan.source_width,
            plan.source_height,
            plan.target_width,
            plan.target_height,
            panel_fraction=0.5,
        )
        panel_span = layout_panel_crop_fraction(
            plan.source_width,
            plan.source_height,
            plan.target_width,
            plan.target_height,
            active_axis=axis,
            split_axis=split_axis,
            panel_fraction=0.5,
        )
        intervals = [_crop_interval(center, panel_span)]
        secondary = _point_secondary_axis_center(point, axis)
        if secondary is not None:
            intervals.append(_crop_interval(secondary, panel_span))
        return intervals

    return [_crop_interval(center, geometry.crop_fraction)]


def _normalised_profile_coverage(
    profile_values: list[float],
    intervals: list[tuple[float, float]],
) -> float | None:
    if not profile_values:
        return None
    masses = np.asarray(profile_values, dtype=np.float64)
    total = float(masses.sum())
    if total <= 1e-9:
        return None
    masses /= total
    count = len(masses)
    bin_start = np.arange(count, dtype=np.float64) / count
    bin_end = (np.arange(count, dtype=np.float64) + 1.0) / count
    covered = np.zeros(count, dtype=np.float64)
    for crop_start, crop_end in intervals:
        overlap = np.maximum(
            0.0,
            np.minimum(bin_end, crop_end) - np.maximum(bin_start, crop_start),
        )
        covered = np.maximum(covered, overlap * count)
    return float(np.clip(np.dot(masses, covered), 0.0, 1.0))


def _target_metrics(
    bundle: AnalysisBundle,
    plan: CropPlan,
    tracks: list[TargetTrack],
    profile: Profile,
) -> dict[str, float]:
    if not tracks:
        return {}
    axis = _active_axis(bundle)
    crop_fraction = _crop_span(bundle.frames[0]) / max(_frame_span(bundle), 1)
    target_visible = 0
    target_inside_detail = 0
    target_visible_anywhere = 0
    target_coverage_values: list[float] = []
    target_confidences: list[float] = []
    target_mask_coverages: list[float] = []
    target_mask_confidences: list[float] = []

    for point in plan.points:
        intervals = _detail_intervals(point, plan)
        visible = _target_observations(tracks, point.time, bundle.sample_fps)
        for track, observation in visible:
            target_visible += 1
            target_confidences.append(observation.confidence)
            center, extent = _observation_axis_values(observation, axis)
            if track.target.kind == "point":
                span = max(crop_fraction * profile.target_context_fraction, extent * 3.0)
            else:
                span = max(extent + crop_fraction * 0.08, crop_fraction * 0.18)
            span = min(1.0, span)
            target_start = float(np.clip(center - span / 2.0, 0.0, 1.0 - span))
            target_end = target_start + span

            detail_coverage = 0.0
            for crop_start, crop_end in intervals:
                overlap = max(0.0, min(crop_end, target_end) - max(crop_start, target_start))
                detail_coverage = max(detail_coverage, min(1.0, overlap / max(span, 1e-6)))
            target_coverage_values.append(detail_coverage)
            mask_profile = observation.mask_y_profile if axis == "y" else observation.mask_x_profile
            mask_coverage = _normalised_profile_coverage(mask_profile, intervals)
            if mask_coverage is not None:
                target_mask_coverages.append(mask_coverage)
                target_mask_confidences.append(observation.mask_confidence)
            if any(start <= center <= end for start, end in intervals):
                target_inside_detail += 1
            if point.mode in {"fallback", "context_detail"} or detail_coverage > 0.0:
                target_visible_anywhere += 1

    return {
        "target_visible_observations": float(target_visible),
        "target_in_crop_fraction": float(target_inside_detail / max(target_visible, 1)),
        "target_visible_anywhere_fraction": float(
            target_visible_anywhere / max(target_visible, 1)
        ),
        "mean_target_context_coverage": (
            float(np.mean(target_coverage_values)) if target_coverage_values else 0.0
        ),
        "p10_target_context_coverage": (
            float(np.percentile(target_coverage_values, 10.0)) if target_coverage_values else 0.0
        ),
        "mean_target_tracking_confidence": (
            float(np.mean(target_confidences)) if target_confidences else 0.0
        ),
        "target_mask_observations": float(len(target_mask_coverages)),
        "mean_target_mask_coverage": (
            float(np.mean(target_mask_coverages)) if target_mask_coverages else 0.0
        ),
        "p10_target_mask_coverage": (
            float(np.percentile(target_mask_coverages, 10.0)) if target_mask_coverages else 0.0
        ),
        "target_mask_clipped_fraction": (
            float(np.mean(np.asarray(target_mask_coverages) < 0.98))
            if target_mask_coverages
            else 0.0
        ),
        "mean_target_mask_confidence": (
            float(np.mean(target_mask_confidences)) if target_mask_confidences else 0.0
        ),
    }

def _text_metrics(bundle: AnalysisBundle, plan: CropPlan) -> dict[str, float]:
    axis = _active_axis(bundle)
    weighted_integrity: list[float] = []
    cut_weights = 0.0
    total_weights = 0.0
    visible_anywhere_weights = 0.0
    trusted_frames = 0

    for frame, point in zip(bundle.frames, plan.points, strict=True):
        text_confidence = _text_layout_confidence(frame)
        if not frame.text_regions or text_confidence < 0.22:
            continue
        trusted_frames += 1
        intervals = _detail_intervals(point, plan)
        for region in frame.text_regions:
            weight = text_confidence * float(np.clip(region.score, 0.1, 1.0)) * (
                0.35 + np.sqrt(max(region.width * region.height, 1e-6)) * 4.5
            )
            center, extent = _region_axis_values(region, axis)
            start = center - extent / 2.0
            end = center + extent / 2.0
            best = 0.0
            for crop_start, crop_end in intervals:
                overlap = max(0.0, min(crop_end, end) - max(crop_start, start))
                best = max(best, min(1.0, overlap / max(extent, 1e-6)))
            weighted_integrity.append(best * weight)
            total_weights += weight
            if 1e-4 < best < 0.999:
                cut_weights += weight
            if point.mode in {"fallback", "context_detail"} or best > 1e-4:
                visible_anywhere_weights += weight

    if total_weights <= 1e-8:
        return {
            "mean_text_integrity_in_detail": 1.0,
            "text_region_cut_fraction": 0.0,
            "text_visible_anywhere_fraction": 1.0,
            "trusted_text_frame_fraction": 0.0,
        }
    return {
        "mean_text_integrity_in_detail": float(sum(weighted_integrity) / total_weights),
        "text_region_cut_fraction": float(cut_weights / total_weights),
        "text_visible_anywhere_fraction": float(visible_anywhere_weights / total_weights),
        "trusted_text_frame_fraction": float(trusted_frames / max(len(bundle.frames), 1)),
    }

def calculate_plan_metrics(
    bundle: AnalysisBundle,
    plan: CropPlan,
    target_tracks: list[TargetTrack] | None = None,
    profile: Profile | None = None,
) -> dict[str, float]:
    axis = _active_axis(bundle)
    crop_span = _crop_span(bundle.frames[0])
    frame_span = _frame_span(bundle)
    centers = np.array(
        [_point_axis_center(point, axis) * frame_span for point in plan.points],
        dtype=np.float64,
    )
    times = np.array([point.time for point in plan.points], dtype=np.float64)
    fallback = np.array([point.mode == "fallback" for point in plan.points], dtype=bool)
    context_detail = np.array(
        [point.mode == "context_detail" for point in plan.points], dtype=bool
    )
    split_layout = np.array([point.mode == "split" for point in plan.points], dtype=bool)
    alternate_layout = fallback | context_detail | split_layout

    layout_families = np.array(
        [
            "crop" if point.mode in {"hold", "pan", "crop"} else point.mode
            for point in plan.points
        ],
        dtype=object,
    )
    layout_switch_count = 0
    for shot_id in np.unique(np.array([point.shot_id for point in plan.points], dtype=np.int32)):
        indices = np.flatnonzero(
            np.array([point.shot_id for point in plan.points], dtype=np.int32) == shot_id
        )
        if len(indices) > 1:
            layout_switch_count += int(
                np.count_nonzero(layout_families[indices][1:] != layout_families[indices][:-1])
            )
    duration_minutes = max(bundle.video.duration / 60.0, 1.0 / 60.0)
    layout_switches_per_minute = float(layout_switch_count / duration_minutes)

    coverage_values: list[float] = []
    evaluation_values: list[float] = []
    for index, (frame, point) in enumerate(zip(bundle.frames, plan.points, strict=True)):
        if fallback[index]:
            coverage_values.append(1.0)
            evaluation_values.append(0.74)
            continue
        candidate_centers = frame.x_positions + _crop_span(frame) / 2.0
        chosen_index = int(np.argmin(np.abs(candidate_centers - centers[index])))
        if alternate_layout[index]:
            coverage_values.append(float(point.content_coverage))
            evaluation_values.append(max(float(frame.evaluation_score[chosen_index]), 0.70))
        else:
            coverage_values.append(float(frame.evaluation_coverage[chosen_index]))
            evaluation_values.append(float(frame.evaluation_score[chosen_index]))

    shot_ids = np.array([point.shot_id for point in plan.points], dtype=np.int32)
    velocity_parts: list[np.ndarray] = []
    acceleration_parts: list[np.ndarray] = []
    jerk_parts: list[np.ndarray] = []
    path_length = 0.0
    for shot_id in np.unique(shot_ids):
        indices = np.flatnonzero(shot_ids == shot_id)
        shot_centers = centers[indices]
        shot_times = times[indices]
        if len(shot_centers) <= 1:
            continue
        signed_deltas = np.diff(shot_centers) / max(crop_span, 1)
        deltas = np.abs(signed_deltas)
        dt = np.maximum(np.diff(shot_times), 1e-3)
        signed_velocity = signed_deltas / dt
        shot_velocity = np.abs(signed_velocity)
        velocity_parts.append(shot_velocity)
        path_length += float(np.sum(deltas))
        if len(signed_velocity) > 1:
            acceleration_dt = np.maximum(np.diff(shot_times[1:]), 1e-3)
            signed_acceleration = np.diff(signed_velocity) / acceleration_dt
            acceleration_parts.append(np.abs(signed_acceleration))
            if len(signed_acceleration) > 1:
                jerk_dt = np.maximum(np.diff(shot_times[2:]), 1e-3)
                jerk_parts.append(np.abs(np.diff(signed_acceleration)) / jerk_dt)

    velocity = np.concatenate(velocity_parts) if velocity_parts else np.zeros(1, dtype=np.float64)
    acceleration = (
        np.concatenate(acceleration_parts)
        if acceleration_parts
        else np.zeros(1, dtype=np.float64)
    )
    jerk = np.concatenate(jerk_parts) if jerk_parts else np.zeros(1, dtype=np.float64)
    mean_velocity = float(np.mean(velocity))
    mean_acceleration = float(np.mean(acceleration))
    mean_jerk = float(np.mean(jerk))
    stability = float(np.exp(-0.65 * mean_velocity - 0.16 * mean_acceleration - 0.025 * mean_jerk))
    full_context_fallback_fraction = float(np.mean(fallback))
    context_detail_fraction = float(np.mean(context_detail))
    split_fraction = float(np.mean(split_layout))
    alternate_layout_fraction = float(np.mean(alternate_layout))
    fallback_fraction = alternate_layout_fraction
    source_aspect = bundle.video.width / max(bundle.video.height, 1)
    target_aspect = bundle.target_width / max(bundle.target_height, 1)
    fill_ratio = min(
        1.0,
        source_aspect / max(target_aspect, 1e-6),
        target_aspect / max(source_aspect, 1e-6),
    )
    proxy_score = 100.0 * (
        0.74 * float(np.mean(coverage_values))
        + 0.26 * stability
        - 0.08 * full_context_fallback_fraction * (1.0 - fill_ratio)
        - 0.025 * context_detail_fraction
        - 0.015 * split_fraction
    )

    # Repeated camera angles should retain a stable visual anchor. Measure the
    # spread between shot-level median centres inside each appearance group.
    shot_groups, _ = _shot_framing_groups(bundle)
    grouped_shots: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for point, center in zip(plan.points, centers, strict=True):
        grouped_shots[shot_groups.get(int(point.shot_id), int(point.shot_id))][int(point.shot_id)].append(float(center))
    repeated_spreads: list[float] = []
    opening_samples = max(1, int(round(bundle.sample_fps * 0.8)))
    for shots in grouped_shots.values():
        if len(shots) < 2:
            continue
        anchors = [float(np.median(values[:opening_samples])) for values in shots.values()]
        repeated_spreads.append(float(np.std(anchors) / max(crop_span, 1)))

    speaker_frames = 0
    speaker_panel_fractions: list[float] = []
    for frame, point in zip(bundle.frames, plan.points, strict=True):
        if any(speaker_region_strength(region) > 0.0 for region in frame.semantic_regions):
            speaker_frames += 1
            if point.mode == "split":
                speaker_panel_fractions.append(float(point.split_primary_fraction))

    metrics = {
        "mean_shared_coverage": float(np.mean(coverage_values)),
        "p10_shared_coverage": float(np.percentile(coverage_values, 10.0)),
        "mean_shared_frame_score": float(np.mean(evaluation_values)),
        "path_length_viewports": path_length,
        "mean_velocity_viewports_per_second": mean_velocity,
        "p95_velocity_viewports_per_second": float(np.percentile(velocity, 95.0)),
        "mean_acceleration_viewports_per_second2": mean_acceleration,
        "mean_jerk_viewports_per_second3": mean_jerk,
        "p95_jerk_viewports_per_second3": float(np.percentile(jerk, 95.0)),
        # Backward-compatible metric names from the portrait-only releases.
        "path_length_crop_widths": path_length,
        "mean_velocity_crop_widths_per_second": mean_velocity,
        "p95_velocity_crop_widths_per_second": float(np.percentile(velocity, 95.0)),
        "mean_acceleration_crop_widths_per_second2": mean_acceleration,
        "mean_jerk_crop_widths_per_second3": mean_jerk,
        "p95_jerk_crop_widths_per_second3": float(np.percentile(jerk, 95.0)),
        "stability_score": stability,
        "fallback_fraction": fallback_fraction,
        "full_context_fallback_fraction": full_context_fallback_fraction,
        "context_detail_fraction": context_detail_fraction,
        "split_layout_fraction": split_fraction,
        "alternate_layout_fraction": alternate_layout_fraction,
        "layout_switch_count": float(layout_switch_count),
        "layout_switches_per_minute": layout_switches_per_minute,
        "fallback_foreground_fill_ratio": float(fill_ratio),
        "mean_confidence": float(np.mean([point.confidence for point in plan.points])),
        "shot_count": float(1 + max(point.shot_id for point in plan.points)),
        "proxy_quality_score_0_100": float(np.clip(proxy_score, 0.0, 100.0)),
        "crop_axis_x": 1.0 if axis == "x" else 0.0,
        "crop_axis_y": 1.0 if axis == "y" else 0.0,
        "repeated_shot_center_spread_viewports": (
            float(np.mean(repeated_spreads)) if repeated_spreads else 0.0
        ),
        "active_speaker_evidence_fraction": float(speaker_frames / max(len(plan.points), 1)),
        "mean_active_speaker_panel_fraction": (
            float(np.mean(speaker_panel_fractions)) if speaker_panel_fractions else 0.5
        ),
    }
    metrics.update(_text_metrics(bundle, plan))
    tracks = target_tracks or []
    if tracks:
        metrics.update(_target_metrics(bundle, plan, tracks, profile or Profile("unknown", "")))
    return metrics

