from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2 as cv
import numpy as np

from .models import AnalysisBundle, SemanticRegion
from .profiles import EVALUATION_WEIGHTS
from .scoring import candidate_statistics, combine_signals
from .semantic import speaker_importance_map
from .video import iter_analysis_frames, require_ffmpeg

ProgressCallback = Callable[[str], None]


def _audio_activity(
    input_path: str | Path,
    duration: float,
    *,
    sample_rate: int = 16000,
    hop_seconds: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a lightweight speech-activity curve from mono PCM audio.

    This is intentionally a small signal-processing baseline, not a speech model.
    Log energy is compared with the clip's estimated noise floor, then smoothed.
    """

    ffmpeg = require_ffmpeg()
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or len(result.stdout) < sample_rate // 5:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)

    hop = max(1, int(round(sample_rate * hop_seconds)))
    window = max(hop, int(round(sample_rate * 0.032)))
    frame_count = max(1, 1 + max(0, samples.size - window) // hop)
    energy = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop
        segment = samples[start : start + window]
        if segment.size < window:
            segment = np.pad(segment, (0, window - segment.size))
        segment = segment - float(np.mean(segment))
        energy[index] = 10.0 * np.log10(float(np.mean(segment * segment)) + 1e-10)

    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)
    noise_floor = float(np.percentile(finite, 20.0))
    speech_level = float(np.percentile(finite, 85.0))
    dynamic_range = speech_level - noise_floor

    # A flat, near-silent track should not be interpreted as continuous speech.
    # Log-energy normalization alone would otherwise assign it a moderate score
    # simply because every sample is equally quiet.
    if speech_level < -52.0 and dynamic_range < 3.0:
        probability = np.zeros_like(energy, dtype=np.float64)
    elif dynamic_range < 1.5:
        # Constant audible material (for example a test tone) contains activity,
        # but it provides little timing information. Give it a conservative score
        # instead of pretending that it is confidently speech.
        probability = np.full_like(energy, 0.55, dtype=np.float64)
    else:
        scale = max(3.0, dynamic_range)
        midpoint = noise_floor + min(8.0, scale * 0.45)
        probability = 1.0 / (1.0 + np.exp(-(energy - midpoint) / max(1.5, scale * 0.18)))
    # A small symmetric smoother removes one-frame audio spikes without adding a
    # meaningful lag to the framing decision.
    kernel = np.asarray([0.08, 0.16, 0.22, 0.28, 0.18, 0.08], dtype=np.float64)
    probability = np.convolve(probability, kernel / kernel.sum(), mode="same")
    if duration > 0.0:
        maximum_count = int(np.ceil(duration / hop_seconds)) + 2
        probability = probability[:maximum_count]
    times = np.arange(probability.size, dtype=np.float64) * hop_seconds
    return times, np.clip(probability, 0.0, 1.0).astype(np.float32)


def _audio_at(times: np.ndarray, values: np.ndarray, time_seconds: float) -> float:
    if times.size == 0 or values.size == 0:
        return 0.0
    index = int(np.clip(np.searchsorted(times, time_seconds), 0, len(times) - 1))
    if index > 0 and abs(times[index - 1] - time_seconds) < abs(times[index] - time_seconds):
        index -= 1
    return float(values[index])


def _face_box(region: SemanticRegion, width: int, height: int) -> tuple[int, int, int, int] | None:
    if region.face_width <= 1e-5 or region.face_height <= 1e-5:
        return None
    x0 = int(np.clip(round(region.face_x * width), 0, max(0, width - 2)))
    y0 = int(np.clip(round(region.face_y * height), 0, max(0, height - 2)))
    x1 = int(np.clip(round((region.face_x + region.face_width) * width), x0 + 2, width))
    y1 = int(np.clip(round((region.face_y + region.face_height) * height), y0 + 2, height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _face_signature(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    face = gray[y0:y1, x0:x1]
    if face.size == 0:
        return np.zeros((48, 48), dtype=np.uint8)
    return cv.resize(face, (48, 48), interpolation=cv.INTER_AREA)


def _mouth_motion(previous: np.ndarray, current: np.ndarray) -> float:
    """Estimate lip motion while discounting rigid head motion.

    Dense flow on a tiny normalized face patch gives every pixel a motion vector.
    The lower-face motion is compared with the upper face, which largely represents
    camera and head movement shared by the whole face.
    """

    flow = cv.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        0.5,
        2,
        11,
        2,
        5,
        1.1,
        0,
    )
    magnitude = cv.magnitude(flow[..., 0], flow[..., 1])
    upper = magnitude[6:25, 7:41]
    mouth = magnitude[27:44, 7:41]
    upper_level = float(np.median(upper)) if upper.size else 0.0
    mouth_level = float(np.mean(mouth)) if mouth.size else 0.0
    difference = max(0.0, mouth_level - 0.65 * upper_level)
    return float(np.clip(difference / 1.35, 0.0, 1.0))


def _region_iou(first: SemanticRegion, second: SemanticRegion) -> float:
    ax0, ay0 = first.face_x, first.face_y
    ax1, ay1 = first.face_x + first.face_width, first.face_y + first.face_height
    bx0, by0 = second.face_x, second.face_y
    bx1, by1 = second.face_x + second.face_width, second.face_y + second.face_height
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = first.face_width * first.face_height + second.face_width * second.face_height - intersection
    return intersection / max(union, 1e-8)


def _assign_tracks(
    regions: list[SemanticRegion],
    previous: dict[int, SemanticRegion],
    next_track_id: int,
) -> tuple[list[int], int]:
    assignments: list[int] = []
    used: set[int] = set()
    for region in regions:
        best_id: int | None = None
        best_score = 0.0
        for track_id, old in previous.items():
            if track_id in used:
                continue
            iou = _region_iou(region, old)
            dx = region.face_x + region.face_width / 2.0 - (old.face_x + old.face_width / 2.0)
            dy = region.face_y + region.face_height / 2.0 - (old.face_y + old.face_height / 2.0)
            distance = float(np.hypot(dx, dy))
            score = iou + max(0.0, 0.24 - distance) * 1.8
            if score > best_score and (iou >= 0.08 or distance <= 0.16):
                best_id, best_score = track_id, score
        if best_id is None:
            best_id = next_track_id
            next_track_id += 1
        assignments.append(best_id)
        used.add(best_id)
    return assignments, next_track_id


def add_active_speaker_evidence(
    input_path: str | Path,
    bundle: AnalysisBundle,
    *,
    enabled: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Attach audio-visual speaking scores to semantic regions and crop evidence."""

    started = time.perf_counter()
    if not enabled or not bundle.frames:
        return {"enabled": bool(enabled), "backend": "disabled", "seconds": 0.0}
    multi_person_frames = sum(len(frame.semantic_regions) >= 2 for frame in bundle.frames)
    face_frames = sum(
        any(region.face_width > 0.0 and region.face_height > 0.0 for region in frame.semantic_regions)
        for frame in bundle.frames
    )
    if face_frames < 2 or multi_person_frames < max(2, int(len(bundle.frames) * 0.03)):
        return {
            "enabled": True,
            "backend": "not_needed",
            "seconds": time.perf_counter() - started,
            "face_frames": face_frames,
            "multi_person_frames": multi_person_frames,
        }

    audio_times, audio_values = _audio_activity(input_path, bundle.video.duration)
    if audio_values.size == 0:
        return {
            "enabled": True,
            "backend": "no_audio",
            "seconds": time.perf_counter() - started,
            "face_frames": face_frames,
            "multi_person_frames": multi_person_frames,
        }
    if progress:
        progress("Estimating the active speaker from audio and mouth motion")

    evidence_times = np.asarray([frame.time for frame in bundle.frames], dtype=np.float64)
    previous_regions: dict[int, SemanticRegion] = {}
    previous_faces: dict[int, np.ndarray] = {}
    smoothed: dict[int, float] = {}
    next_track_id = 1
    last_shot_id: int | None = None
    processed = 0
    speaking_frames = 0
    switches = 0
    previous_winner: int | None = None

    frame_iterator = iter_analysis_frames(
        input_path,
        width=bundle.analysis_width,
        height=bundle.analysis_height,
        sample_fps=bundle.sample_fps,
        use_ffmpeg_proxy=True,
    )
    for _, time_seconds, frame_bgr in frame_iterator:
        nearest_index = int(np.argmin(np.abs(evidence_times - time_seconds)))
        evidence = bundle.frames[nearest_index]
        if abs(evidence.time - time_seconds) > 0.65 / max(bundle.sample_fps, 1e-6):
            continue
        regions = list(evidence.semantic_regions)
        if evidence.shot_id != last_shot_id:
            previous_regions.clear()
            previous_faces.clear()
            smoothed.clear()
            previous_winner = None
            last_shot_id = evidence.shot_id
        track_ids, next_track_id = _assign_tracks(regions, previous_regions, next_track_id)
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        audio_probability = _audio_at(audio_times, audio_values, time_seconds)
        scores: list[float] = []
        current_faces: dict[int, np.ndarray] = {}
        current_regions: dict[int, SemanticRegion] = {}

        for region, track_id in zip(regions, track_ids, strict=True):
            box = _face_box(region, frame_bgr.shape[1], frame_bgr.shape[0])
            visual = 0.0
            if box is not None:
                face = _face_signature(gray, box)
                current_faces[track_id] = face
                old_face = previous_faces.get(track_id)
                if old_face is not None:
                    visual = _mouth_motion(old_face, face)
            prior = 0.24 if len(regions) == 1 else 0.06
            raw_score = audio_probability * (prior + (1.0 - prior) * visual)
            old_score = smoothed.get(track_id, raw_score)
            score = 0.62 * old_score + 0.38 * raw_score
            if audio_probability < 0.18:
                score *= 0.35
            smoothed[track_id] = score
            scores.append(float(np.clip(score, 0.0, 1.0)))
            current_regions[track_id] = region

        winner: int | None = None
        if scores and audio_probability >= 0.22:
            order = np.argsort(np.asarray(scores))[::-1]
            best_index = int(order[0])
            best_id = track_ids[best_index]
            best_score = scores[best_index]
            second_score = scores[int(order[1])] if len(order) > 1 else 0.0
            if best_score >= 0.16 and best_score - second_score >= 0.025:
                winner = best_id
                if previous_winner in track_ids:
                    previous_index = track_ids.index(previous_winner)
                    previous_score = scores[previous_index]
                    if best_id != previous_winner and best_score < previous_score + 0.12:
                        winner = previous_winner
        if winner is not None:
            speaking_frames += 1
            if previous_winner is not None and winner != previous_winner:
                switches += 1
            previous_winner = winner

        updated: list[SemanticRegion] = []
        for region, track_id, score in zip(regions, track_ids, scores, strict=True):
            is_winner = winner == track_id
            confidence = float(np.clip(audio_probability * (0.55 + 0.45 * score), 0.0, 1.0))
            updated.append(
                replace(
                    region,
                    speaker_score=(score if is_winner else score * 0.22),
                    speaker_confidence=(confidence if is_winner else confidence * 0.30),
                    speaker_track_id=track_id,
                )
            )
        evidence.semantic_regions = updated

        speaker_map, speaker_reliability = speaker_importance_map(
            (bundle.analysis_height, bundle.analysis_width), updated
        )
        coverage, centering, boundary = candidate_statistics(
            speaker_map,
            evidence.x_positions,
            evidence.crop_height if evidence.crop_axis == "y" else evidence.crop_width,
            axis=evidence.crop_axis,
        )
        evidence.signal_coverage["speaker"] = coverage
        evidence.signal_centering["speaker"] = centering
        evidence.signal_boundary["speaker"] = boundary
        evidence.signal_reliability["speaker"] = speaker_reliability
        evidence.evaluation_score, evidence.evaluation_coverage = combine_signals(
            x_positions=evidence.x_positions,
            crop_width=(evidence.crop_height if evidence.crop_axis == "y" else evidence.crop_width),
            frame_width=(bundle.analysis_height if evidence.crop_axis == "y" else bundle.analysis_width),
            weights=EVALUATION_WEIGHTS,
            coverage=evidence.signal_coverage,
            centering=evidence.signal_centering,
            boundary=evidence.signal_boundary,
            reliability=evidence.signal_reliability,
        )

        previous_regions = current_regions
        previous_faces = current_faces
        processed += 1

    return {
        "enabled": True,
        "backend": "opencv_yunet_mouth_flow+energy_vad",
        "seconds": time.perf_counter() - started,
        "processed_frames": processed,
        "speaking_frames": speaking_frames,
        "speaker_switches": switches,
        "audio_samples": int(audio_values.size),
    }
