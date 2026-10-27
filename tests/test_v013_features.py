from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from fastapi.testclient import TestClient

from reframelab import active_speaker
from reframelab.analysis_cache import load_analysis_bundle, save_analysis_bundle
from reframelab.corrections import FramingCorrection, parse_framing_corrections
from reframelab.models import AnalysisBundle, FrameEvidence, Profile, SemanticRegion, VideoInfo
from reframelab.planner import build_plan
from reframelab.synthetic import generate_sample
from reframelab.webapp import create_app


def _frame(
    *,
    index: int,
    time_seconds: float,
    shot_id: int,
    preferred: int,
    descriptor: np.ndarray,
    semantic_regions: list[SemanticRegion] | None = None,
) -> FrameEvidence:
    positions = np.asarray([60, 90, 120], dtype=np.int32)
    coverage = np.full(3, 0.2, dtype=np.float32)
    coverage[preferred] = 1.0
    return FrameEvidence(
        frame_index=index,
        time=time_seconds,
        shot_id=shot_id,
        shot_score=0.0,
        x_positions=positions,
        crop_width=100,
        signal_coverage={"saliency": coverage.copy()},
        signal_centering={"saliency": np.ones(3, dtype=np.float32)},
        signal_boundary={"saliency": np.zeros(3, dtype=np.float32)},
        signal_reliability={"saliency": 1.0},
        evaluation_score=coverage.copy(),
        evaluation_coverage=coverage.copy(),
        semantic_regions=list(semantic_regions or []),
        appearance_descriptor=descriptor.astype(np.float32),
    )


def _repeated_shot_bundle() -> AnalysisBundle:
    descriptor = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    descriptor /= np.linalg.norm(descriptor)
    frames = [
        _frame(index=0, time_seconds=0.0, shot_id=0, preferred=0, descriptor=descriptor),
        _frame(index=1, time_seconds=0.5, shot_id=0, preferred=0, descriptor=descriptor),
        _frame(index=2, time_seconds=1.0, shot_id=1, preferred=1, descriptor=descriptor),
        _frame(index=3, time_seconds=1.5, shot_id=1, preferred=1, descriptor=descriptor),
    ]
    return AnalysisBundle(
        video=VideoInfo("source.mp4", 300, 200, 2.0, 4, 2.0),
        analysis_width=300,
        analysis_height=200,
        target_width=100,
        target_height=200,
        sample_fps=2.0,
        frames=frames,
    )


def _simple_profile() -> Profile:
    return Profile(
        name="test",
        description="test",
        signal_weights={"saliency": 1.0},
        temporal=False,
        stabilize=False,
        allow_fallback=False,
    )


def test_consistent_framing_reduces_repeated_shot_offset() -> None:
    bundle = _repeated_shot_bundle()
    unlocked = build_plan(bundle, _simple_profile(), consistent_framing=False)
    locked = build_plan(bundle, _simple_profile(), consistent_framing=True)

    unlocked_centers = [point.center_x_norm for point in unlocked.points]
    locked_centers = [point.center_x_norm for point in locked.points]
    unlocked_gap = abs(np.mean(unlocked_centers[:2]) - np.mean(unlocked_centers[2:]))
    locked_gap = abs(np.mean(locked_centers[:2]) - np.mean(locked_centers[2:]))

    assert locked_gap < unlocked_gap * 0.25
    metadata = locked.implementation["consistent_framing"]
    assert metadata["repeated_group_count"] == 1
    assert metadata["locked_shots"] == 2


def test_matching_correction_copies_to_similar_shots_and_forces_layout() -> None:
    bundle = _repeated_shot_bundle()
    correction = FramingCorrection(
        time=0.2,
        center_x=0.72,
        center_y=0.5,
        scope="matching",
        mode="context_detail",
    )
    plan = build_plan(
        bundle,
        _simple_profile(),
        framing_corrections=[correction],
        consistent_framing=True,
    )
    assert all(point.center_x_norm == pytest.approx(0.72, abs=0.015) for point in plan.points)
    assert {point.mode for point in plan.points} == {"context_detail"}
    assert plan.implementation["framing_corrections"][0]["affected_samples"] == 4


def test_correction_parser_rejects_bad_coordinates_and_limits() -> None:
    parsed = parse_framing_corrections(
        json.dumps([{"time": 1.2, "x": 0.4, "y": 0.6, "scope": "shot"}])
    )
    assert parsed[0].center_x == pytest.approx(0.4)
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_framing_corrections([{"time": 0, "center_x": 2.0, "center_y": 0.5}])
    with pytest.raises(ValueError, match="at most 40"):
        parse_framing_corrections(
            [{"time": index, "center_x": 0.5, "center_y": 0.5} for index in range(41)]
        )


def test_analysis_cache_round_trip_preserves_appearance_and_semantics(tmp_path: Path) -> None:
    bundle = _repeated_shot_bundle()
    bundle.frames[0].semantic_regions = [
        SemanticRegion("person", 0.1, 0.1, 0.3, 0.8, 0.9, 0.15, 0.15, 0.1, 0.15)
    ]
    path = save_analysis_bundle(tmp_path / "analysis.pkl.gz", bundle)
    loaded = load_analysis_bundle(path)
    assert loaded.video.width == bundle.video.width
    np.testing.assert_allclose(
        loaded.frames[0].appearance_descriptor,
        bundle.frames[0].appearance_descriptor,
    )
    assert loaded.frames[0].semantic_regions[0].face_width == pytest.approx(0.1)


def test_mouth_motion_discounts_shared_head_motion() -> None:
    first = np.zeros((48, 48), dtype=np.uint8)
    second = first.copy()
    cv.rectangle(second, (14, 32), (34, 39), 255, -1)
    mouth_motion = active_speaker._mouth_motion(first, second)

    shifted = np.roll(first, 2, axis=1)
    rigid_motion = active_speaker._mouth_motion(first, shifted)
    assert mouth_motion > 0.15
    assert mouth_motion > rigid_motion


def test_active_speaker_adds_left_speaker_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    width, height = 200, 100
    left = SemanticRegion(
        "person", 0.05, 0.05, 0.40, 0.90, 0.95,
        0.12, 0.12, 0.20, 0.36,
    )
    right = SemanticRegion(
        "person", 0.55, 0.05, 0.40, 0.90, 0.95,
        0.62, 0.12, 0.20, 0.36,
    )
    positions = np.asarray([0, 100], dtype=np.int32)
    frames: list[FrameEvidence] = []
    video_frames: list[tuple[int, float, np.ndarray]] = []
    for index in range(6):
        evidence = FrameEvidence(
            frame_index=index,
            time=index * 0.2,
            shot_id=0,
            shot_score=0.0,
            x_positions=positions,
            crop_width=100,
            signal_coverage={},
            signal_centering={},
            signal_boundary={},
            signal_reliability={},
            evaluation_score=np.zeros(2, dtype=np.float32),
            evaluation_coverage=np.zeros(2, dtype=np.float32),
            semantic_regions=[left, right],
        )
        frames.append(evidence)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Alternate the lower part of the left face while leaving the right face still.
        if index % 2:
            cv.rectangle(image, (31, 35), (49, 44), (255, 255, 255), -1)
        video_frames.append((index, index * 0.2, image))

    bundle = AnalysisBundle(
        video=VideoInfo("speaker.mp4", width, height, 5.0, 6, 1.2),
        analysis_width=width,
        analysis_height=height,
        target_width=100,
        target_height=100,
        sample_fps=5.0,
        frames=frames,
    )
    monkeypatch.setattr(
        active_speaker,
        "_audio_activity",
        lambda *args, **kwargs: (
            np.arange(60, dtype=np.float64) * 0.02,
            np.ones(60, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        active_speaker,
        "iter_analysis_frames",
        lambda *args, **kwargs: iter(video_frames),
    )

    metadata = active_speaker.add_active_speaker_evidence("speaker.mp4", bundle)
    assert metadata["processed_frames"] == 6
    assert metadata["speaking_frames"] >= 2
    left_scores = [frame.semantic_regions[0].speaker_score for frame in bundle.frames]
    right_scores = [frame.semantic_regions[1].speaker_score for frame in bundle.frames]
    assert max(left_scores) > max(right_scores) + 0.05
    assert any(frame.signal_reliability.get("speaker", 0.0) > 0 for frame in bundle.frames)


def test_web_page_exposes_new_controls(tmp_path: Path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        html = client.get("/").text
    assert 'id="corrections"' in html
    assert 'data-mode="frame"' in html
    assert 'id="consistentFraming"' in html
    assert 'id="activeSpeaker"' in html
    assert "/correct" in html


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_web_correction_reuses_cached_analysis(tmp_path: Path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.4,
    )
    app = create_app(work_root=tmp_path / "runs", keep_days=0, max_workers=1)
    with TestClient(app) as client, source.open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("sample.mp4", video, "video/mp4")},
            data={
                "preset": "best",
                "quality": "quick",
                "output_size": "180x320",
                "active_speaker": "false",
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]

        deadline = time.monotonic() + 90
        job = response.json()
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "completed", job

        correction = [{
            "time": 0.2,
            "center_x": 0.25,
            "center_y": 0.5,
            "scope": "all",
            "mode": "crop",
        }]
        rerun = client.post(
            f"/api/jobs/{job_id}/correct",
            data={"corrections": json.dumps(correction)},
        )
        assert rerun.status_code == 202, rerun.text
        job = rerun.json()
        deadline = time.monotonic() + 90
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "completed", job
        performance = client.get(job["results"]["performance_url"]).json()
        assert performance["analysis_cache_reused"] is True
        assert job["corrections"][0]["scope"] == "all"
        plan = client.get(job["results"]["report_url"])
        assert plan.status_code == 200


def test_audio_activity_does_not_mark_flat_silence_as_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32).tobytes()

    class Result:
        returncode = 0
        stdout = silence

    monkeypatch.setattr(active_speaker, "require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(active_speaker.subprocess, "run", lambda *args, **kwargs: Result())
    _, activity = active_speaker._audio_activity("silent.mp4", 1.0, sample_rate=sample_rate)
    assert activity.size > 0
    assert float(np.max(activity)) == pytest.approx(0.0)


def test_audio_activity_keeps_flat_audible_test_signal_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 16000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.1 * np.sin(2.0 * np.pi * 440.0 * time_axis)).astype(np.float32).tobytes()

    class Result:
        returncode = 0
        stdout = tone

    monkeypatch.setattr(active_speaker, "require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(active_speaker.subprocess, "run", lambda *args, **kwargs: Result())
    _, activity = active_speaker._audio_activity("tone.mp4", 1.0, sample_rate=sample_rate)
    assert activity.size > 0
    assert 0.25 <= float(np.median(activity)) <= 0.65
