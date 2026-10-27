from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from reframelab.analyzer import analyze_video
from reframelab.pipeline import run_demo
from reframelab.renderer import render_plan
from reframelab.synthetic import generate_sample
from reframelab.video import iter_analysis_frames, probe_video

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")


@requires_ffmpeg
def test_ffmpeg_analysis_stream_samples_and_resizes_before_python(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4",
        width=320,
        height=180,
        fps=12.0,
        shot_seconds=0.5,
    )
    frames = list(
        iter_analysis_frames(
            source,
            width=160,
            height=90,
            sample_fps=3.0,
            use_ffmpeg_proxy=True,
        )
    )
    assert frames
    assert all(frame.shape == (90, 160, 3) for _, _, frame in frames)
    expected = round(probe_video(source).duration * 3.0)
    assert len(frames) == pytest.approx(expected, abs=1)
    assert frames[1][1] - frames[0][1] == pytest.approx(1.0 / 3.0, abs=1e-6)


@requires_ffmpeg
def test_detector_cadence_reduces_expensive_detection_calls(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4",
        width=320,
        height=180,
        fps=10.0,
        shot_seconds=1.0,
    )
    full = analyze_video(
        source,
        target_width=180,
        target_height=320,
        analysis_height=120,
        sample_fps=5.0,
        coarse_to_fine=False,
        detector_cadence=False,
        use_analysis_proxy=True,
        allow_model_download=False,
    )
    spaced = analyze_video(
        source,
        target_width=180,
        target_height=320,
        analysis_height=120,
        sample_fps=5.0,
        coarse_to_fine=False,
        detector_cadence=True,
        use_analysis_proxy=True,
        allow_model_download=False,
    )
    full_counts = full.performance["passes"][0]["counts"]
    spaced_counts = spaced.performance["passes"][0]["counts"]
    assert spaced_counts["text_dnn_calls"] < full_counts["text_dnn_calls"]
    assert spaced_counts["people_calls"] < full_counts["people_calls"]
    assert len(spaced.frames) == len(full.frames)


@requires_ffmpeg
def test_direct_ffmpeg_and_legacy_render_have_matching_timeline(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.5,
    )
    result = run_demo(
        source,
        tmp_path / "analysis",
        ["center"],
        output_size="180x320",
        analysis_height=90,
        analysis_fps=2.0,
        create_comparison=False,
        use_analysis_proxy=True,
        coarse_to_fine=False,
        detector_cadence=True,
        direct_ffmpeg=True,
        allow_model_download=False,
    )
    plan = next(iter(result.plan_paths.values()))
    from reframelab.models import CropPlan

    loaded = CropPlan.load(plan)
    direct = tmp_path / "direct.mp4"
    legacy = tmp_path / "legacy.mp4"
    render_plan(source, loaded, direct, direct_ffmpeg=True)
    render_plan(source, loaded, legacy, direct_ffmpeg=False)

    def probe(path):
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    direct_info = probe(direct)
    legacy_info = probe(legacy)
    assert float(direct_info["format"]["duration"]) == pytest.approx(
        float(legacy_info["format"]["duration"]), abs=0.15
    )
    direct_video = next(item for item in direct_info["streams"] if item["codec_type"] == "video")
    legacy_video = next(item for item in legacy_info["streams"] if item["codec_type"] == "video")
    assert abs(int(direct_video["nb_frames"]) - int(legacy_video["nb_frames"])) <= 1


@requires_ffmpeg
def test_run_manifest_records_each_speed_switch(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.4,
    )
    result = run_demo(
        source,
        tmp_path / "run",
        ["center"],
        output_size="180x320",
        analysis_height=90,
        analysis_fps=2.0,
        create_comparison=False,
        use_analysis_proxy=False,
        coarse_to_fine=False,
        detector_cadence=False,
        direct_ffmpeg=False,
        allow_model_download=False,
    )
    manifest = json.loads((result.output_dir / "run_manifest.json").read_text())
    assert manifest["speed_optimizations"] == {
        "analysis_proxy": False,
        "coarse_to_fine": False,
        "detector_cadence": False,
        "direct_ffmpeg": False,
        "static_scene_reuse": True,
        "persistent_models": True,
        "dnn_optimization": True,
        "dnn_batch_size": 4,
        "pipeline_queues": True,
    }
    performance = json.loads(result.performance_path.read_text())
    assert performance["settings"] == manifest["speed_optimizations"]

@requires_ffmpeg
def test_coarse_to_fine_refines_a_bounded_fraction(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "long-sample.mp4",
        width=240,
        height=136,
        fps=8.0,
        shot_seconds=6.25,
    )
    bundle = analyze_video(
        source,
        target_width=180,
        target_height=320,
        analysis_height=120,
        sample_fps=5.0,
        coarse_to_fine=True,
        detector_cadence=True,
        use_analysis_proxy=True,
        allow_model_download=False,
    )
    performance = bundle.performance
    assert performance["coarse_to_fine"] is True
    assert performance["refined_fraction"] <= 0.41
    assert performance["passes"][0]["kind"] == "coarse"
    assert performance["passes"][0]["counts"]["text_dnn_calls"] < performance["passes"][0]["counts"]["samples"]


@requires_ffmpeg
def test_static_scene_reuse_skips_repeated_visual_analysis(tmp_path) -> None:
    import cv2 as cv
    import numpy as np

    source = tmp_path / "static.mp4"
    writer = cv.VideoWriter(
        str(source), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
    )
    assert writer.isOpened()
    frame = np.full((180, 320, 3), 235, dtype=np.uint8)
    cv.putText(frame, "STATIC TITLE", (55, 95), cv.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    for _ in range(40):
        writer.write(frame)
    writer.release()

    reused = analyze_video(
        source,
        target_width=180,
        target_height=320,
        analysis_height=120,
        sample_fps=5.0,
        coarse_to_fine=False,
        detector_cadence=True,
        static_scene_reuse=True,
        persistent_models=False,
        dnn_optimization=False,
        pipeline_queues=False,
        allow_model_download=False,
    )
    full = analyze_video(
        source,
        target_width=180,
        target_height=320,
        analysis_height=120,
        sample_fps=5.0,
        coarse_to_fine=False,
        detector_cadence=True,
        static_scene_reuse=False,
        persistent_models=False,
        dnn_optimization=False,
        pipeline_queues=False,
        allow_model_download=False,
    )
    reused_counts = reused.performance["passes"][0]["counts"]
    full_counts = full.performance["passes"][0]["counts"]
    assert reused_counts["static_reuse_frames"] >= 10
    assert reused_counts["full_analysis_frames"] < full_counts["full_analysis_frames"]
    assert len(reused.frames) == len(full.frames)
    assert np.mean(reused.frames[-1].evaluation_score) == pytest.approx(
        np.mean(full.frames[-1].evaluation_score), abs=0.08
    )


def test_persistent_detector_pool_reuses_one_bundle(monkeypatch) -> None:
    from reframelab import detector_pool

    detector_pool.clear_detector_cache()
    calls = {"person": 0, "text": 0}

    class FakePerson:
        def __init__(self, **kwargs):
            del kwargs
            calls["person"] += 1

        def runtime_metadata(self):
            return {"backend": "fake_person"}

    class FakeText:
        def __init__(self, **kwargs):
            del kwargs
            calls["text"] += 1

        def runtime_metadata(self):
            return {"backend": "fake_text"}

    monkeypatch.setattr(detector_pool, "PersonDetector", FakePerson)
    monkeypatch.setattr(detector_pool, "TextDetector", FakeText)
    first = detector_pool.get_detector_bundle(
        semantic_people=True,
        semantic_text=True,
        model_dir=None,
        allow_download=False,
        persistent=True,
        dnn_optimization=True,
        dnn_batch_size=4,
        progress=None,
    )
    second = detector_pool.get_detector_bundle(
        semantic_people=True,
        semantic_text=True,
        model_dir=None,
        allow_download=False,
        persistent=True,
        dnn_optimization=True,
        dnn_batch_size=4,
        progress=None,
    )
    assert first.person is second.person
    assert first.text is second.text
    assert calls == {"person": 1, "text": 1}
    assert second.metadata["cache_hit"] is True
    detector_pool.clear_detector_cache()


def test_mediapipe_detector_batches_multiple_frames() -> None:
    import numpy as np

    from reframelab.semantic import _blazepose_anchors, _MediaPipePersonDetector

    class FakeNet:
        def setInput(self, blob) -> None:  # noqa: N802
            self.batch = blob.shape[0]

        def getUnconnectedOutLayersNames(self):  # noqa: N802
            return ["boxes", "scores"]

        def forward(self, names):
            del names
            regression = np.zeros((self.batch, 2254, 12), dtype=np.float32)
            scores = np.full((self.batch, 2254, 1), -20.0, dtype=np.float32)
            index = 28 * 28
            regression[:, index, :4] = [0.0, 0.0, 38.0, 60.0]
            regression[:, index, 4:12] = [0.0, 8.0, 0.0, 70.0, 0.0, -16.0, 0.0, 22.0]
            scores[:, index, 0] = 10.0
            return [regression, scores]

    detector = _MediaPipePersonDetector.__new__(_MediaPipePersonDetector)
    detector.net = FakeNet()
    detector.score_threshold = 0.48
    detector.nms_threshold = 0.30
    detector.anchors = _blazepose_anchors()
    results = detector.detect_batch([
        np.zeros((200, 320, 3), dtype=np.uint8),
        np.zeros((180, 240, 3), dtype=np.uint8),
    ])
    assert len(results) == 2
    assert all(len(regions) == 1 for regions in results)


@requires_ffmpeg
def test_pipeline_queues_can_be_disabled_without_changing_duration(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4", width=320, height=180, fps=8.0, shot_seconds=0.5
    )
    result = run_demo(
        source,
        tmp_path / "analysis",
        ["center"],
        output_size="180x320",
        analysis_height=90,
        analysis_fps=2.0,
        create_comparison=False,
        coarse_to_fine=False,
        pipeline_queues=True,
        allow_model_download=False,
    )
    from reframelab.models import CropPlan

    plan = CropPlan.load(next(iter(result.plan_paths.values())))
    queued = tmp_path / "queued.mp4"
    direct = tmp_path / "direct.mp4"
    render_plan(source, plan, queued, pipeline_queues=True)
    render_plan(source, plan, direct, pipeline_queues=False)
    queued_info = probe_video(queued)
    direct_info = probe_video(direct)
    assert queued_info.duration == pytest.approx(direct_info.duration, abs=0.15)
    assert abs(queued_info.frame_count - direct_info.frame_count) <= 1
