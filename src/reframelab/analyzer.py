from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from pathlib import Path

import cv2 as cv
import numpy as np

from .active_speaker import add_active_speaker_evidence
from .advanced_cv import DenseFlowEstimator, DepthEstimator
from .detector_pool import get_detector_bundle
from .geometry import crop_geometry, scale_crop_geometry
from .models import AnalysisBundle, FrameEvidence, VideoInfo
from .profiles import EVALUATION_WEIGHTS
from .scoring import (
    candidate_positions,
    candidate_statistics,
    combine_signals,
    text_region_statistics,
)
from .semantic import PersonDetector, people_importance_map
from .signals import SignalResult, compute_signals, shot_change_score
from .text import TextDetector
from .video import even, iter_analysis_frames, prefetch_iterator, probe_video

ProgressCallback = Callable[[str], None]


def _analysis_dimensions(video: VideoInfo, analysis_height: int) -> tuple[int, int]:
    scale = min(1.0, analysis_height / video.height)
    height = even(int(round(video.height * scale)))
    width = even(int(round(video.width * scale)))
    return width, height


def _resize_map(signal_map: np.ndarray, width: int, height: int) -> np.ndarray:
    if signal_map.shape[:2] == (height, width):
        return signal_map.astype(np.float32, copy=False)
    interpolation = cv.INTER_AREA if signal_map.shape[0] >= height else cv.INTER_LINEAR
    return cv.resize(signal_map, (width, height), interpolation=interpolation).astype(np.float32)


def _score_difficulty(frame: FrameEvidence, shot_threshold: float) -> float:
    coverage = float(np.max(frame.evaluation_coverage)) if frame.evaluation_coverage.size else 0.0
    scores = np.sort(frame.evaluation_score.astype(np.float64))
    if scores.size >= 2:
        spread = max(float(scores[-1] - scores[0]), 1e-6)
        margin = float((scores[-1] - scores[-2]) / spread)
    else:
        margin = 1.0
    motion = float(frame.signal_reliability.get("motion", 0.0))
    text_complete = float(frame.signal_reliability.get("text_complete", 0.0))
    people = float(frame.signal_reliability.get("people", 0.0))
    text_risk = 0.0
    if text_complete > 0.12:
        text_coverage = frame.signal_coverage.get("text_complete")
        best_text = float(np.max(text_coverage)) if text_coverage is not None else 0.0
        text_risk = max(0.0, 0.94 - best_text)
    multi_person = min(1.0, max(0, len(frame.semantic_regions) - 1) / 2.0)
    cut_risk = float(np.clip(frame.shot_score / max(shot_threshold, 1e-6), 0.0, 1.0))
    return float(
        np.clip(
            0.30 * max(0.0, 0.80 - coverage) / 0.80
            + 0.18 * max(0.0, 0.13 - margin) / 0.13
            + 0.17 * motion
            + 0.17 * text_risk
            + 0.10 * multi_person
            + 0.08 * cut_risk
            + 0.04 * people,
            0.0,
            1.0,
        )
    )


def _merge_intervals(
    intervals: list[tuple[float, float]],
    *,
    duration: float,
    merge_gap: float = 0.75,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((max(0.0, a), min(duration, b)) for a, b in intervals if b > a)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + merge_gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _difficult_intervals(
    frames: list[FrameEvidence],
    *,
    duration: float,
    sample_fps: float,
    shot_threshold: float,
    maximum_fraction: float = 0.58,
) -> tuple[list[tuple[float, float]], list[float]]:
    if not frames:
        return [], []
    difficulties = [_score_difficulty(frame, shot_threshold) for frame in frames]
    threshold = 0.30
    selected = [index for index, score in enumerate(difficulties) if score >= threshold]
    # Bound the *time* refined rather than merely the number of coarse samples.
    # Padding each selected sample can otherwise merge into almost the full video.
    padding = max(0.22, 0.65 / max(sample_fps, 1e-6))
    approximate_window = max(2.0 * padding, 1.0 / max(sample_fps, 1e-6))
    maximum_selected = max(1, int((duration * maximum_fraction) / approximate_window))
    if len(selected) > maximum_selected:
        selected = sorted(
            range(len(frames)),
            key=lambda index: difficulties[index],
            reverse=True,
        )[:maximum_selected]
        selected.sort()
    intervals = [
        (frames[index].time - padding, frames[index].time + padding)
        for index in selected
    ]
    merged = _merge_intervals(intervals, duration=duration, merge_gap=0.18)
    # If neighbouring windows still merge beyond the budget, trim the lowest-value
    # interval edges while retaining the highest-difficulty centres.
    budget = duration * maximum_fraction
    total = sum(end - start for start, end in merged)
    if total > budget and budget > 0.0:
        ranked = sorted(
            merged,
            key=lambda interval: max(
                (
                    difficulties[index]
                    for index, frame in enumerate(frames)
                    if interval[0] <= frame.time <= interval[1]
                ),
                default=0.0,
            ),
            reverse=True,
        )
        kept: list[tuple[float, float]] = []
        remaining = budget
        for start, end in ranked:
            if remaining <= 1e-6:
                break
            length = end - start
            if length <= remaining:
                kept.append((start, end))
                remaining -= length
            else:
                centre = (start + end) / 2.0
                kept.append((max(0.0, centre - remaining / 2.0), min(duration, centre + remaining / 2.0)))
                remaining = 0.0
        merged = sorted(kept)
    return merged, difficulties


def _shot_lookup(frames: list[FrameEvidence]) -> Callable[[float], int]:
    times = [frame.time for frame in frames]
    shot_ids = [frame.shot_id for frame in frames]

    def lookup(time_seconds: float) -> int:
        index = bisect.bisect_right(times, time_seconds) - 1
        return shot_ids[max(0, min(index, len(shot_ids) - 1))]

    return lookup


def _static_scene_metrics(reference_bgr: np.ndarray, current_bgr: np.ndarray) -> dict[str, float]:
    """Measure whether two sampled frames carry effectively identical evidence."""

    preview_size = (160, 90)
    reference = cv.resize(reference_bgr, preview_size, interpolation=cv.INTER_AREA)
    current = cv.resize(current_bgr, preview_size, interpolation=cv.INTER_AREA)
    reference_gray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current, cv.COLOR_BGR2GRAY)
    pixel_delta = float(np.mean(cv.absdiff(reference_gray, current_gray)) / 255.0)
    reference_hist = cv.calcHist([reference_gray], [0], None, [32], [0, 256])
    current_hist = cv.calcHist([current_gray], [0], None, [32], [0, 256])
    cv.normalize(reference_hist, reference_hist)
    cv.normalize(current_hist, current_hist)
    histogram_delta = float(cv.compareHist(reference_hist, current_hist, cv.HISTCMP_BHATTACHARYYA))
    reference_edges = cv.Canny(reference_gray, 64, 160)
    current_edges = cv.Canny(current_gray, 64, 160)
    edge_delta = float(np.mean(cv.absdiff(reference_edges, current_edges) > 0))
    return {
        "pixel_delta": pixel_delta,
        "histogram_delta": histogram_delta,
        "edge_delta": edge_delta,
    }


def _appearance_descriptor(frame_bgr: np.ndarray) -> np.ndarray:
    """Compact shot fingerprint used to keep repeated camera angles consistent."""

    preview = cv.resize(frame_bgr, (96, 54), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(preview, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).reshape(-1)
    histogram = histogram.astype(np.float32)
    histogram /= max(float(np.linalg.norm(histogram)), 1e-6)
    gray = cv.cvtColor(preview, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 64, 160).astype(np.float32) / 255.0
    column_profile = cv.resize(
        np.mean(edges, axis=0, keepdims=True), (16, 1), interpolation=cv.INTER_AREA
    ).reshape(-1)
    row_profile = cv.resize(
        np.mean(edges, axis=1, keepdims=True).T, (12, 1), interpolation=cv.INTER_AREA
    ).reshape(-1)
    descriptor = np.concatenate([histogram, column_profile, row_profile]).astype(np.float32)
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-6)
    return descriptor


def _is_static_scene(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["pixel_delta"] <= 0.012
        and metrics["histogram_delta"] <= 0.055
        and metrics["edge_delta"] <= 0.032
    )


def _analyze_pass(
    input_path: str | Path,
    *,
    video: VideoInfo,
    target_width: int,
    target_height: int,
    processing_height: int,
    reference_height: int,
    sample_fps: float,
    candidate_step: int,
    shot_threshold: float,
    minimum_shot_seconds: float,
    person_detector: PersonDetector | None,
    text_detector: TextDetector | None,
    dense_flow: DenseFlowEstimator | None,
    depth_estimator: DepthEstimator | None,
    adaptive_dense_flow: bool,
    advanced_vision: bool,
    use_analysis_proxy: bool,
    detector_cadence: bool,
    text_dnn_interval_seconds: float,
    people_interval_seconds: float,
    static_scene_reuse: bool,
    dnn_optimization: bool,
    dnn_batch_size: int,
    pipeline_queues: bool,
    start_time: float = 0.0,
    end_time: float | None = None,
    source_shot_lookup: Callable[[float], int] | None = None,
    active_intervals: list[tuple[float, float]] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[FrameEvidence], dict[str, object]]:
    processing_width, processing_height = _analysis_dimensions(video, processing_height)
    reference_width, reference_height = _analysis_dimensions(video, reference_height)
    source_geometry = crop_geometry(video.width, video.height, target_width, target_height)
    geometry = scale_crop_geometry(
        source_geometry,
        reference_width,
        reference_height,
        even_dimensions=True,
    )
    frame_span = geometry.frame_span
    crop_span = geometry.crop_span
    positions = candidate_positions(frame_span, crop_span, candidate_step)

    frames: list[FrameEvidence] = []
    previous_sample: np.ndarray | None = None
    previous_shot_id: int | None = None
    shot_id = 0
    last_cut_time = start_time - minimum_shot_seconds
    last_text_dnn_time = -1e9
    last_people_time = -1e9
    cached_people: list = []
    duration = max(0.0, (end_time if end_time is not None else video.duration) - start_time)
    active_duration = (
        sum(max(0.0, end - start) for start, end in active_intervals)
        if active_intervals
        else duration
    )
    expected_samples = max(1, int(np.ceil(active_duration * sample_fps)))

    timing = {
        "frame_stream_seconds": 0.0,
        "text_seconds": 0.0,
        "signals_seconds": 0.0,
        "people_seconds": 0.0,
        "dense_flow_seconds": 0.0,
        "scoring_seconds": 0.0,
        "static_check_seconds": 0.0,
        "batch_wait_seconds": 0.0,
    }
    counts: dict[str, int] = {
        "samples": 0,
        "text_dnn_calls": 0,
        "text_dnn_batches": 0,
        "text_classical_calls": 0,
        "people_calls": 0,
        "people_batches": 0,
        "dense_flow_calls": 0,
        "static_reuse_frames": 0,
        "full_analysis_frames": 0,
        "prefetched_frames": 0,
        "text_detector_errors": 0,
        "people_detector_errors": 0,
    }
    detector_errors: list[str] = []
    detector_warnings: list[str] = []
    static_metrics_total = {"pixel_delta": 0.0, "histogram_delta": 0.0, "edge_delta": 0.0}
    static_metrics_count = 0

    last_full_frame: np.ndarray | None = None
    last_full_time = -1e9
    last_full_maps: dict[str, np.ndarray] | None = None
    last_full_reliability: dict[str, float] | None = None
    last_full_text_result = None
    last_full_semantic: list = []
    last_full_motion_backend = "sparse_lk"
    last_full_depth_backend = "disabled"
    static_refresh_seconds = 3.0

    def analysis_stream():
        if active_intervals is None:
            yield from iter_analysis_frames(
                input_path,
                width=processing_width,
                height=processing_height,
                sample_fps=sample_fps,
                start_time=start_time,
                end_time=end_time,
                use_ffmpeg_proxy=use_analysis_proxy,
            )
            return
        upper_bound = end_time if end_time is not None else video.duration
        for interval_start, interval_end in active_intervals:
            clipped_start = max(start_time, interval_start)
            clipped_end = min(upper_bound, interval_end)
            if clipped_end <= clipped_start:
                continue
            yield from iter_analysis_frames(
                input_path,
                width=processing_width,
                height=processing_height,
                sample_fps=sample_fps,
                start_time=clipped_start,
                end_time=clipped_end,
                use_ffmpeg_proxy=use_analysis_proxy,
            )

    raw_stream = analysis_stream()
    stream = prefetch_iterator(raw_stream, maxsize=max(4, dnn_batch_size * 2)) if pipeline_queues else raw_stream
    batch_size = max(1, int(dnn_batch_size if dnn_optimization else 1))

    wall_start = time.perf_counter()
    previous_stream_time: float | None = None

    def process_chunk(chunk: list[tuple[int, float, np.ndarray]]) -> None:
        nonlocal previous_sample, previous_shot_id, shot_id, last_cut_time
        nonlocal last_text_dnn_time, last_people_time, cached_people
        nonlocal last_full_frame, last_full_time, last_full_maps, last_full_reliability
        nonlocal last_full_text_result, last_full_semantic
        nonlocal last_full_motion_backend, last_full_depth_backend
        nonlocal static_metrics_count, previous_stream_time

        tasks: list[dict[str, object]] = []
        for _sample_index, time_seconds, resized in chunk:
            if (
                previous_stream_time is not None
                and time_seconds - previous_stream_time > 1.75 / max(sample_fps, 1e-6)
            ):
                previous_sample = None
                previous_shot_id = None
                last_full_frame = None
                last_full_maps = None
                last_full_reliability = None
                last_full_text_result = None
                last_full_semantic = []
            previous_stream_time = time_seconds
            cut_score = shot_change_score(previous_sample, resized)
            detected_cut = (
                previous_sample is not None
                and cut_score >= shot_threshold
                and time_seconds - last_cut_time >= minimum_shot_seconds
            )
            if source_shot_lookup is None:
                if detected_cut:
                    shot_id += 1
                    last_cut_time = time_seconds
                current_shot_id = shot_id
            else:
                current_shot_id = source_shot_lookup(time_seconds)
            is_cut = previous_shot_id is not None and current_shot_id != previous_shot_id
            is_cut = bool(is_cut or detected_cut)
            signal_previous = None if is_cut else previous_sample

            static_hit = False
            metrics: dict[str, float] | None = None
            if (
                static_scene_reuse
                and not is_cut
                and last_full_frame is not None
                and last_full_maps is not None
                and time_seconds - last_full_time <= static_refresh_seconds
            ):
                static_started = time.perf_counter()
                metrics = _static_scene_metrics(last_full_frame, resized)
                timing["static_check_seconds"] += time.perf_counter() - static_started
                static_hit = _is_static_scene(metrics)
                for name, value in metrics.items():
                    static_metrics_total[name] += value
                static_metrics_count += 1

            use_text_dnn = bool(
                not static_hit
                and text_detector is not None
                and (
                    not detector_cadence
                    or is_cut
                    or not frames
                    or time_seconds - last_text_dnn_time >= text_dnn_interval_seconds
                )
            )
            if use_text_dnn:
                last_text_dnn_time = time_seconds
            run_people = bool(
                not static_hit
                and person_detector is not None
                and (
                    not detector_cadence
                    or is_cut
                    or not frames
                    or time_seconds - last_people_time >= people_interval_seconds
                )
            )
            if run_people:
                last_people_time = time_seconds

            tasks.append(
                {
                    "time": time_seconds,
                    "frame": resized,
                    "cut_score": cut_score,
                    "shot_id": current_shot_id,
                    "is_cut": is_cut,
                    "previous": signal_previous,
                    "static": static_hit,
                    "static_metrics": metrics,
                    "use_text_dnn": use_text_dnn,
                    "run_people": run_people,
                }
            )
            previous_sample = resized
            previous_shot_id = current_shot_id

        # Run scheduled DNN work in groups. Classical text processing is also grouped
        # so the model path and its fallback share one API.
        nonstatic = [task for task in tasks if not task["static"]]
        if text_detector is not None and nonstatic:
            text_started = time.perf_counter()
            flags = [bool(task["use_text_dnn"]) for task in nonstatic]
            try:
                text_results = text_detector.detect_batch(
                    [task["frame"] for task in nonstatic], use_dnn=flags
                )
            except Exception as exc:  # Optional semantic detector: degrade, do not lose the job.
                counts["text_detector_errors"] += 1
                message = f"text detector: {type(exc).__name__}: {exc}"
                detector_errors.append(message)
                if progress:
                    progress(
                        "Text detection failed for one analysis batch; continuing with "
                        "classical text geometry."
                    )
                try:
                    text_results = text_detector.detect_batch(
                        [task["frame"] for task in nonstatic],
                        use_dnn=[False] * len(nonstatic),
                    )
                except Exception as fallback_exc:
                    detector_errors.append(
                        f"classical text fallback: {type(fallback_exc).__name__}: {fallback_exc}"
                    )
                    text_results = [None for _ in nonstatic]
            for task, result in zip(nonstatic, text_results, strict=False):
                task["text_result"] = result
            dnn_frames = sum(flags)
            counts["text_dnn_calls"] += dnn_frames
            counts["text_classical_calls"] += len(flags) - dnn_frames
            if dnn_frames:
                counts["text_dnn_batches"] += 1
            timing["text_seconds"] += time.perf_counter() - text_started
        else:
            for task in nonstatic:
                task["text_result"] = None

        people_tasks = [task for task in tasks if bool(task["run_people"])]
        if person_detector is not None and people_tasks:
            people_started = time.perf_counter()
            try:
                detected = person_detector.detect_batch(
                    [task["frame"] for task in people_tasks]
                )
                if len(detected) != len(people_tasks):
                    raise RuntimeError(
                        f"Person detector returned {len(detected)} results for "
                        f"{len(people_tasks)} frames"
                    )
            except Exception as exc:  # Person semantics are optional crop evidence.
                counts["people_detector_errors"] += 1
                detector_errors.append(f"people detector: {type(exc).__name__}: {exc}")
                if progress:
                    progress(
                        "Person and face detection failed for one analysis batch; "
                        "continuing with motion, text, structure, and saliency."
                    )
                detected = [[] for _ in people_tasks]
            drain_warnings = getattr(person_detector, "drain_runtime_warnings", None)
            if callable(drain_warnings):
                for warning in drain_warnings():
                    detector_warnings.append(str(warning))
                    if progress:
                        progress(str(warning))
            for task, result in zip(people_tasks, detected, strict=False):
                task["people_result"] = result
            counts["people_calls"] += len(people_tasks)
            counts["people_batches"] += 1
            timing["people_seconds"] += time.perf_counter() - people_started

        for task in tasks:
            frame_start = time.perf_counter()
            time_seconds = float(task["time"])
            resized = task["frame"]
            is_static = bool(task["static"])
            if is_static and last_full_maps is not None and last_full_reliability is not None:
                signal_maps = dict(last_full_maps)
                signal_maps["motion"] = np.zeros(resized.shape[:2], dtype=np.float32)
                signal_reliability = dict(last_full_reliability)
                signal_reliability["motion"] = 0.0
                signals = SignalResult(signal_maps, signal_reliability)
                text_result = last_full_text_result
                semantic_regions = list(last_full_semantic)
                motion_backend = "static_scene_reuse"
                depth_backend = last_full_depth_backend
                counts["static_reuse_frames"] += 1
            else:
                text_result = task.get("text_result")
                signals_start = time.perf_counter()
                signals = compute_signals(
                    resized,
                    task["previous"],
                    text_map=(text_result.importance_map if text_result else None),
                    text_reliability=(text_result.reliability if text_result else None),
                )
                timing["signals_seconds"] += time.perf_counter() - signals_start

                motion_backend = "sparse_lk"
                sparse_reliability = float(signals.reliability.get("motion", 0.0))
                use_dense_flow = (
                    dense_flow is not None
                    and task["previous"] is not None
                    and (advanced_vision or (adaptive_dense_flow and sparse_reliability < 0.16))
                )
                if use_dense_flow:
                    dense_started = time.perf_counter()
                    dense_result = dense_flow.residual_motion(task["previous"], resized)
                    sparse_motion = signals.maps.get("motion")
                    maximum_dense_weight = 0.62 if dense_result.backend == "opencv_dnn_raft" else 0.28
                    dense_weight = float(
                        np.clip(dense_result.reliability * maximum_dense_weight, 0.0, maximum_dense_weight)
                    )
                    if sparse_motion is None:
                        fused_motion = dense_result.residual_map
                    else:
                        agreement = np.sqrt(
                            np.clip(sparse_motion, 0.0, 1.0)
                            * np.clip(dense_result.residual_map, 0.0, 1.0)
                        )
                        fused_motion = (
                            (1.0 - dense_weight) * sparse_motion
                            + dense_weight * (0.68 * dense_result.residual_map + 0.32 * agreement)
                        ).astype(np.float32)
                    signals.maps["motion"] = fused_motion
                    signals.reliability["motion"] = float(
                        np.clip(
                            sparse_reliability * (1.0 - 0.35 * dense_weight)
                            + dense_result.reliability * dense_weight,
                            0.0,
                            1.0,
                        )
                    )
                    motion_backend = dense_result.backend
                    counts["dense_flow_calls"] += 1
                    timing["dense_flow_seconds"] += time.perf_counter() - dense_started

                depth_backend = "disabled"
                if depth_estimator is not None:
                    depth_result = depth_estimator.infer(resized)
                    text_gate = 1.0 - 0.78 * float(
                        np.clip(signals.reliability.get("text", 0.0), 0.0, 1.0)
                    )
                    signals.maps["depth"] = depth_result.importance_map
                    signals.reliability["depth"] = float(
                        np.clip(depth_result.reliability * text_gate, 0.0, 1.0)
                    )
                    depth_backend = depth_result.backend
                else:
                    signals.maps["depth"] = np.zeros(resized.shape[:2], dtype=np.float32)
                    signals.reliability["depth"] = 0.0

                if bool(task["run_people"]):
                    cached_people = list(task.get("people_result", []))
                semantic_regions = list(cached_people)
                if person_detector is not None:
                    people_map, people_reliability = people_importance_map(
                        resized.shape[:2], semantic_regions
                    )
                    signals.maps["people"] = people_map
                    signals.reliability["people"] = people_reliability

                counts["full_analysis_frames"] += 1
                last_full_frame = resized
                last_full_time = time_seconds
                last_full_maps = dict(signals.maps)
                last_full_reliability = dict(signals.reliability)
                last_full_text_result = text_result
                last_full_semantic = list(semantic_regions)
                last_full_motion_backend = motion_backend
                last_full_depth_backend = depth_backend

            scoring_started = time.perf_counter()
            signal_coverage: dict[str, np.ndarray] = {}
            signal_centering: dict[str, np.ndarray] = {}
            signal_boundary: dict[str, np.ndarray] = {}
            for name, signal_map in signals.maps.items():
                reference_map = _resize_map(signal_map, reference_width, reference_height)
                coverage, centering, boundary = candidate_statistics(
                    reference_map, positions, crop_span, axis=geometry.axis
                )
                signal_coverage[name] = coverage
                signal_centering[name] = centering
                signal_boundary[name] = boundary

            text_regions = text_result.regions if text_result else []
            text_backend = text_result.backend if text_result else "disabled"
            if is_static and text_backend != "disabled":
                text_backend += "+static_reuse"
            if text_regions:
                text_coverage, text_centering, text_boundary = text_region_statistics(
                    text_regions, positions, crop_span, frame_span, axis=geometry.axis
                )
                signal_coverage["text_complete"] = text_coverage
                signal_centering["text_complete"] = text_centering
                signal_boundary["text_complete"] = text_boundary
                signals.reliability["text_complete"] = float(
                    max(0.18, text_result.reliability if text_result else 0.0)
                )

            evaluation_score, evaluation_coverage = combine_signals(
                x_positions=positions,
                crop_width=crop_span,
                frame_width=frame_span,
                weights=EVALUATION_WEIGHTS,
                coverage=signal_coverage,
                centering=signal_centering,
                boundary=signal_boundary,
                reliability=signals.reliability,
            )
            timing["scoring_seconds"] += time.perf_counter() - scoring_started

            source_frame_index = max(0, int(round(time_seconds * video.fps)))
            frames.append(
                FrameEvidence(
                    frame_index=source_frame_index,
                    time=time_seconds,
                    shot_id=int(task["shot_id"]),
                    shot_score=float(task["cut_score"]),
                    x_positions=positions.copy(),
                    crop_width=geometry.crop_width,
                    signal_coverage=signal_coverage,
                    signal_centering=signal_centering,
                    signal_boundary=signal_boundary,
                    signal_reliability=signals.reliability,
                    evaluation_score=evaluation_score,
                    evaluation_coverage=evaluation_coverage,
                    crop_axis=geometry.axis,
                    crop_height=geometry.crop_height,
                    semantic_regions=semantic_regions,
                    semantic_backend=(person_detector.backend if person_detector else "disabled"),
                    text_regions=text_regions,
                    text_backend=text_backend,
                    motion_backend=motion_backend,
                    depth_backend=depth_backend,
                    appearance_descriptor=_appearance_descriptor(resized),
                )
            )
            counts["samples"] += 1
            timing["frame_stream_seconds"] += time.perf_counter() - frame_start
            if progress and (
                len(frames) == 1 or len(frames) % max(1, int(sample_fps * 4)) == 0
            ):
                progress(f"Analysed {len(frames)}/{expected_samples} sampled frames")

    chunk: list[tuple[int, float, np.ndarray]] = []
    for item in stream:
        counts["prefetched_frames"] += int(pipeline_queues)
        chunk.append(item)
        if len(chunk) >= batch_size:
            process_chunk(chunk)
            chunk = []
    if chunk:
        process_chunk(chunk)

    timing["wall_seconds"] = time.perf_counter() - wall_start
    static_average = {
        name: value / max(static_metrics_count, 1)
        for name, value in static_metrics_total.items()
    }
    return frames, {
        "processing_resolution": [processing_width, processing_height],
        "reference_resolution": [reference_width, reference_height],
        "sample_fps": sample_fps,
        "start_time": start_time,
        "end_time": end_time,
        "active_intervals": (
            [[start, end] for start, end in active_intervals]
            if active_intervals is not None
            else None
        ),
        "analysis_proxy": use_analysis_proxy,
        "detector_cadence": detector_cadence,
        "static_scene_reuse": static_scene_reuse,
        "dnn_optimization": dnn_optimization,
        "dnn_batch_size": batch_size,
        "pipeline_queues": pipeline_queues,
        "static_metrics_mean": static_average,
        "timings": timing,
        "counts": counts,
        "detector_errors": detector_errors,
        "detector_warnings": detector_warnings,
    }

def analyze_video(
    input_path: str | Path,
    *,
    target_width: int = 720,
    target_height: int = 1280,
    analysis_height: int = 240,
    sample_fps: float = 5.0,
    candidate_step: int = 8,
    shot_threshold: float = 0.43,
    minimum_shot_seconds: float = 0.55,
    semantic_people: bool = True,
    semantic_text: bool = True,
    active_speaker: bool = True,
    adaptive_dense_flow: bool = True,
    advanced_vision: bool = False,
    enable_dense_flow: bool = False,
    enable_depth: bool = False,
    model_dir: str | Path | None = None,
    allow_model_download: bool = True,
    use_analysis_proxy: bool = True,
    coarse_to_fine: bool = True,
    detector_cadence: bool = True,
    static_scene_reuse: bool = True,
    persistent_models: bool = True,
    dnn_optimization: bool = True,
    dnn_batch_size: int = 4,
    pipeline_queues: bool = True,
    text_dnn_interval_seconds: float = 1.0,
    people_interval_seconds: float = 0.45,
    refine_max_fraction: float = 0.40,
    progress: ProgressCallback | None = None,
) -> AnalysisBundle:
    video = probe_video(input_path)
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if analysis_height < 32:
        raise ValueError("analysis_height must be at least 32")

    reference_width, reference_height = _analysis_dimensions(video, analysis_height)
    detector_bundle = get_detector_bundle(
        semantic_people=semantic_people,
        semantic_text=semantic_text,
        model_dir=model_dir,
        allow_download=allow_model_download,
        persistent=persistent_models,
        dnn_optimization=dnn_optimization,
        dnn_batch_size=dnn_batch_size,
        progress=progress,
    )
    person_detector = detector_bundle.person
    text_detector = detector_bundle.text
    reset_person_runtime = getattr(person_detector, "reset_runtime_state", None)
    if callable(reset_person_runtime):
        reset_person_runtime()
    dense_flow = (
        DenseFlowEstimator(
            model_dir=model_dir,
            allow_download=allow_model_download,
            progress=progress,
            enable_learned=advanced_vision and enable_dense_flow,
        )
        if adaptive_dense_flow or (advanced_vision and enable_dense_flow)
        else None
    )
    depth_estimator = (
        DepthEstimator(
            model_dir=model_dir,
            allow_download=allow_model_download,
            progress=progress,
            enable_learned=True,
        )
        if advanced_vision and enable_depth
        else None
    )

    total_start = time.perf_counter()
    passes: list[dict[str, object]] = []
    refined_intervals: list[tuple[float, float]] = []
    use_coarse = bool(
        coarse_to_fine
        and video.duration >= 24.0
        and (sample_fps > 3.05 or analysis_height > 180)
    )
    if use_coarse:
        coarse_fps = min(2.5, sample_fps)
        coarse_height = min(160, analysis_height)
        if progress:
            progress(
                f"Running coarse analysis at {coarse_fps:g} fps and {coarse_height}px"
            )
        coarse_frames, coarse_stats = _analyze_pass(
            input_path,
            video=video,
            target_width=target_width,
            target_height=target_height,
            processing_height=coarse_height,
            reference_height=analysis_height,
            sample_fps=coarse_fps,
            candidate_step=candidate_step,
            shot_threshold=shot_threshold,
            minimum_shot_seconds=minimum_shot_seconds,
            person_detector=person_detector,
            text_detector=text_detector,
            dense_flow=dense_flow,
            depth_estimator=depth_estimator,
            adaptive_dense_flow=adaptive_dense_flow,
            advanced_vision=advanced_vision,
            use_analysis_proxy=use_analysis_proxy,
            detector_cadence=detector_cadence,
            text_dnn_interval_seconds=max(text_dnn_interval_seconds, 2.0),
            people_interval_seconds=max(people_interval_seconds, 1.5),
            static_scene_reuse=static_scene_reuse,
            dnn_optimization=dnn_optimization,
            dnn_batch_size=dnn_batch_size,
            pipeline_queues=pipeline_queues,
            progress=progress,
        )
        passes.append({"kind": "coarse", **coarse_stats})
        refined_intervals, difficulties = _difficult_intervals(
            coarse_frames,
            duration=video.duration,
            sample_fps=coarse_fps,
            shot_threshold=shot_threshold,
            maximum_fraction=float(np.clip(refine_max_fraction, 0.05, 1.0)),
        )
        if refined_intervals:
            if progress:
                total_refine = sum(end - start for start, end in refined_intervals)
                progress(
                    f"Refining {len(refined_intervals)} difficult interval(s), "
                    f"{total_refine:.1f}s total"
                )
            lookup = _shot_lookup(coarse_frames)
            refined_frames, refined_stats = _analyze_pass(
                input_path,
                video=video,
                target_width=target_width,
                target_height=target_height,
                processing_height=analysis_height,
                reference_height=analysis_height,
                sample_fps=sample_fps,
                candidate_step=candidate_step,
                shot_threshold=shot_threshold,
                minimum_shot_seconds=minimum_shot_seconds,
                person_detector=person_detector,
                text_detector=text_detector,
                dense_flow=dense_flow,
                depth_estimator=depth_estimator,
                adaptive_dense_flow=adaptive_dense_flow,
                advanced_vision=advanced_vision,
                use_analysis_proxy=use_analysis_proxy,
                detector_cadence=detector_cadence,
                text_dnn_interval_seconds=text_dnn_interval_seconds,
                people_interval_seconds=people_interval_seconds,
                static_scene_reuse=static_scene_reuse,
                dnn_optimization=dnn_optimization,
                dnn_batch_size=dnn_batch_size,
                pipeline_queues=pipeline_queues,
                source_shot_lookup=lookup,
                active_intervals=refined_intervals,
                progress=progress,
            )
            passes.append({"kind": "refined", **refined_stats})
            retained_coarse = [
                frame
                for frame in coarse_frames
                if not any(start <= frame.time <= end for start, end in refined_intervals)
            ]
            frames = sorted(retained_coarse + refined_frames, key=lambda frame: frame.time)
            # Remove near-duplicate samples at interval boundaries, preferring the
            # later item in the list (the refined frame after sorting by source order).
            deduplicated: list[FrameEvidence] = []
            minimum_gap = 0.35 / max(sample_fps, 1e-6)
            for frame in frames:
                if deduplicated and frame.time - deduplicated[-1].time < minimum_gap:
                    if frame.time >= deduplicated[-1].time:
                        deduplicated[-1] = frame
                else:
                    deduplicated.append(frame)
            frames = deduplicated
        else:
            frames = coarse_frames
            difficulties = [_score_difficulty(frame, shot_threshold) for frame in frames]
    else:
        frames, single_stats = _analyze_pass(
            input_path,
            video=video,
            target_width=target_width,
            target_height=target_height,
            processing_height=analysis_height,
            reference_height=analysis_height,
            sample_fps=sample_fps,
            candidate_step=candidate_step,
            shot_threshold=shot_threshold,
            minimum_shot_seconds=minimum_shot_seconds,
            person_detector=person_detector,
            text_detector=text_detector,
            dense_flow=dense_flow,
            depth_estimator=depth_estimator,
            adaptive_dense_flow=adaptive_dense_flow,
            advanced_vision=advanced_vision,
            use_analysis_proxy=use_analysis_proxy,
            detector_cadence=detector_cadence,
            text_dnn_interval_seconds=text_dnn_interval_seconds,
            people_interval_seconds=people_interval_seconds,
            static_scene_reuse=static_scene_reuse,
            dnn_optimization=dnn_optimization,
            dnn_batch_size=dnn_batch_size,
            pipeline_queues=pipeline_queues,
            progress=progress,
        )
        passes.append({"kind": "single", **single_stats})
        difficulties = [_score_difficulty(frame, shot_threshold) for frame in frames]

    if not frames:
        raise RuntimeError(f"No frames could be decoded from {input_path}")

    try:
        active_speaker_stats = add_active_speaker_evidence(
            input_path,
            AnalysisBundle(
                video=video,
                analysis_width=reference_width,
                analysis_height=reference_height,
                target_width=target_width,
                target_height=target_height,
                sample_fps=sample_fps,
                frames=frames,
                crop_axis=frames[0].crop_axis,
            ),
            enabled=active_speaker,
            progress=progress,
        )
    except Exception as exc:  # Optional evidence must not cancel the user's run.
        active_speaker_stats = {
            "enabled": bool(active_speaker),
            "backend": "failed_optional",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": 0.0,
        }
        if progress:
            progress(f"Active-speaker analysis failed; continuing without it: {exc}")

    detector_runtime = dict(detector_bundle.metadata)
    detector_runtime["person"] = (
        person_detector.runtime_metadata() if person_detector is not None else None
    )
    detector_runtime["text"] = (
        text_detector.runtime_metadata() if text_detector is not None else None
    )
    performance = {
        "total_seconds": time.perf_counter() - total_start,
        "analysis_proxy": use_analysis_proxy,
        "coarse_to_fine": use_coarse,
        "detector_cadence": detector_cadence,
        "static_scene_reuse": static_scene_reuse,
        "persistent_models": persistent_models,
        "dnn_optimization": dnn_optimization,
        "dnn_batch_size": int(max(1, dnn_batch_size)),
        "pipeline_queues": pipeline_queues,
        "detector_runtime": detector_runtime,
        "active_speaker": active_speaker_stats,
        "text_dnn_interval_seconds": text_dnn_interval_seconds,
        "people_interval_seconds": people_interval_seconds,
        "refined_intervals": [[start, end] for start, end in refined_intervals],
        "refined_fraction": (
            sum(end - start for start, end in refined_intervals) / max(video.duration, 1e-6)
        ),
        "difficulty_mean": float(np.mean(difficulties)) if difficulties else 0.0,
        "difficulty_p90": float(np.percentile(difficulties, 90.0)) if difficulties else 0.0,
        "passes": passes,
        "final_sample_count": len(frames),
    }
    return AnalysisBundle(
        video=video,
        analysis_width=reference_width,
        analysis_height=reference_height,
        target_width=target_width,
        target_height=target_height,
        sample_fps=sample_fps,
        frames=frames,
        crop_axis=frames[0].crop_axis,
        performance=performance,
    )
