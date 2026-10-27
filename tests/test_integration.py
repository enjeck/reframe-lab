from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from reframelab.analyzer import analyze_video
from reframelab.detector_pool import DetectorBundle
from reframelab.ground_truth import evaluate_ground_truth
from reframelab.models import CropPlan, PlanPoint, SemanticRegion
from reframelab.pipeline import run_demo
from reframelab.planner import build_plan
from reframelab.profiles import DEVELOPMENT_DEMO_PROFILES, get_profile
from reframelab.semantic import PersonDetector
from reframelab.synthetic import generate_sample
from reframelab.video import prepare_working_video, probe_video


@pytest.fixture()
def short_sample(tmp_path):
    video, ground_truth = generate_sample(
        tmp_path / "sample.mp4",
        width=480,
        height=270,
        fps=10.0,
        shot_seconds=0.8,
    )
    return video, ground_truth


def test_sample_analysis_and_planning(short_sample) -> None:
    video, ground_truth = short_sample
    bundle = analyze_video(
        video,
        target_width=180,
        target_height=320,
        analysis_height=135,
        sample_fps=5.0,
        candidate_step=4,
        shot_threshold=0.35,
        minimum_shot_seconds=0.5,
    )
    assert 1 + max(frame.shot_id for frame in bundle.frames) == 4

    center = build_plan(bundle, get_profile("center"))
    temporal = build_plan(bundle, get_profile("hybrid_temporal"))
    safe = build_plan(bundle, get_profile("hybrid_safe"))

    center_gt = evaluate_ground_truth(center, ground_truth)
    temporal_gt = evaluate_ground_truth(temporal, ground_truth)
    safe_gt = evaluate_ground_truth(safe, ground_truth)

    assert temporal_gt["ground_truth_mean_center_error_crop_widths"] < center_gt[
        "ground_truth_mean_center_error_crop_widths"
    ]
    assert safe_gt["ground_truth_fallback_recall"] == pytest.approx(1.0)
    assert safe_gt["ground_truth_mode_accuracy"] >= 0.9
    assert 0.0 < safe.metrics["fallback_fraction"] < 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_full_demo_renders_video_and_audio(short_sample, tmp_path) -> None:
    video, _ = short_sample
    output_dir = tmp_path / "demo"
    result = run_demo(
        video,
        output_dir,
        ["center", "hybrid_safe"],
        output_size="180x320",
        analysis_height=135,
        analysis_fps=5.0,
        candidate_step=4,
        shot_threshold=0.35,
        create_comparison=True,
    )
    assert result.report_path.stat().st_size > 1000
    assert result.comparison_path is not None and result.comparison_path.stat().st_size > 1000
    assert all(path.stat().st_size > 1000 for path in result.rendered_paths.values())

    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(result.rendered_paths["hybrid_safe"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]}
    assert {"video", "audio"}.issubset(stream_types)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_render_preserves_short_audio_tail_with_last_frame_hold(tmp_path) -> None:
    """A source may keep playing audio after its final encoded video frame."""

    source = tmp_path / "audio-tail.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=10:d=1.0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.4",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )

    plan = CropPlan(
        profile="center",
        description="test",
        source=str(source),
        source_width=160,
        source_height=90,
        source_fps=10.0,
        target_width=50,
        target_height=90,
        sample_fps=2.0,
        points=[
            PlanPoint(0, 0.0, 0, 0.5, "crop", 1.0, 1.0),
            PlanPoint(9, 0.9, 0, 0.5, "crop", 1.0, 1.0),
        ],
    )
    output = tmp_path / "portrait.mp4"

    from reframelab.renderer import render_plan

    render_plan(source, plan, output)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    duration = float(payload["format"]["duration"])
    stream_types = {stream["codec_type"] for stream in payload["streams"]}
    assert duration >= 1.35
    assert duration <= 1.5
    assert {"video", "audio"}.issubset(stream_types)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_reliable_webm_source_uses_direct_decode(short_sample, tmp_path) -> None:
    source, _ = short_sample
    webm = tmp_path / "browser-recording.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-c:a",
            "libopus",
            str(webm),
        ],
        check=True,
    )

    with prepare_working_video(webm) as working:
        assert working.normalized is False
        assert working.reason is None
        assert working.path == webm.resolve()
        info = probe_video(working.path)
        assert info.width == 480
        assert info.height == 270
        assert info.frame_count > 0
        assert 1.0 < info.fps <= 60.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_target_guided_demo_tracks_selection_and_improves_coverage(short_sample, tmp_path) -> None:
    from reframelab.targets import TargetAnchor, TargetSpec

    video, _ = short_sample
    target = TargetSpec(
        id="product",
        label="Product",
        kind="point",
        role="follow",
        anchors=(TargetAnchor(0.4, 0.50, 0.55),),
    )
    result = run_demo(
        video,
        tmp_path / "target-demo",
        ["center", "hybrid_safe", "target_guided_safe"],
        output_size="180x320",
        analysis_height=135,
        analysis_fps=5.0,
        candidate_step=4,
        shot_threshold=0.35,
        create_comparison=True,
        target_specs=[target],
    )

    assert result.tracks_path is not None and result.tracks_path.stat().st_size > 500
    assert (
        result.tracking_preview_path is not None
        and result.tracking_preview_path.stat().st_size > 1000
    )
    center = CropPlan.load(result.plan_paths["center"])
    guided = CropPlan.load(result.plan_paths["target_guided_safe"])
    assert guided.metrics["target_in_crop_fraction"] > center.metrics["target_in_crop_fraction"]
    assert guided.metrics["mean_target_context_coverage"] > center.metrics[
        "mean_target_context_coverage"
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_all_development_profiles_complete_with_frozen_face_metadata(
    tmp_path, monkeypatch
) -> None:
    """Exercise the former FrozenInstanceError path through every crop profile."""

    source, _ = generate_sample(
        tmp_path / "faces.mp4",
        width=160,
        height=90,
        fps=4.0,
        shot_seconds=0.25,
    )
    detector = PersonDetector.__new__(PersonDetector)
    detector.progress = None
    detector.backend = "test_person+test_face"
    detector.batch_size = 4
    detector._person_batch_disabled = False
    detector._person_batch_failures = 0
    detector._person_frame_failures = 0
    detector._face_failures = 0
    detector._face_fallback_uses = 0
    detector._warnings_emitted = set()
    detector._last_person_error = None
    detector._last_face_error = None
    detector._load_seconds = 0.0
    detector._yunet = None
    detector._haar = None
    detector._haar_fallback = None

    body = SemanticRegion("person", 0.12, 0.05, 0.42, 0.90, 0.82)
    face = SemanticRegion(
        "person",
        0.18,
        0.10,
        0.18,
        0.55,
        0.94,
        face_x=0.22,
        face_y=0.12,
        face_width=0.10,
        face_height=0.15,
    )

    class FakePersonBackend:
        engine = "test"
        engine_benchmark_ms = {}
        load_seconds = 0.0

        def detect_batch(self, frames):
            return [[body] for _ in frames]

        def detect(self, frame):
            del frame
            return [body]

    detector._person = FakePersonBackend()
    monkeypatch.setattr(detector, "_face_regions", lambda frame: [face])

    import reframelab.analyzer as analyzer_module

    monkeypatch.setattr(
        analyzer_module,
        "get_detector_bundle",
        lambda **kwargs: DetectorBundle(
            person=detector,
            text=None,
            metadata={"cache_hit": False, "test": True},
        ),
    )

    result = run_demo(
        source,
        tmp_path / "all-profiles",
        list(DEVELOPMENT_DEMO_PROFILES),
        output_size="90x160",
        analysis_height=90,
        analysis_fps=2.0,
        candidate_step=8,
        create_comparison=True,
        allow_model_download=False,
        coarse_to_fine=False,
        detector_cadence=False,
        dnn_batch_size=4,
    )

    assert set(result.rendered_paths) == set(DEVELOPMENT_DEMO_PROFILES)
    assert all(
        path.is_file() and path.stat().st_size > 1000
        for path in result.rendered_paths.values()
    )
    assert result.comparison_path is not None
    assert result.comparison_path.stat().st_size > 1000


def test_semantic_detector_failure_is_non_fatal(tmp_path, monkeypatch) -> None:
    from dataclasses import FrozenInstanceError

    source, _ = generate_sample(
        tmp_path / "semantic-failure.mp4",
        width=160,
        height=90,
        fps=4.0,
        shot_seconds=0.25,
    )

    class ExplodingPersonDetector:
        backend = "broken_test_detector"

        def detect_batch(self, frames):
            del frames
            raise FrozenInstanceError("cannot assign to field 'face_x'")

        def runtime_metadata(self):
            return {"backend": self.backend, "face_failures": 1}

    import reframelab.analyzer as analyzer_module

    monkeypatch.setattr(
        analyzer_module,
        "get_detector_bundle",
        lambda **kwargs: DetectorBundle(
            person=ExplodingPersonDetector(),
            text=None,
            metadata={"cache_hit": False},
        ),
    )

    bundle = analyze_video(
        source,
        target_width=90,
        target_height=160,
        analysis_height=90,
        sample_fps=2.0,
        allow_model_download=False,
        coarse_to_fine=False,
        detector_cadence=False,
    )

    assert bundle.frames
    pass_stats = bundle.performance["passes"][0]
    assert pass_stats["counts"]["people_detector_errors"] >= 1
    assert pass_stats["detector_errors"]
