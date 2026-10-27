from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reframelab.analyzer import analyze_video
from reframelab.geometry import (
    choose_layout_split_axis,
    crop_geometry,
    panel_crop_fraction,
)
from reframelab.models import AnalysisBundle, FrameEvidence, Profile, VideoInfo
from reframelab.planner import build_plan
from reframelab.renderer import PlanInterpolator, _crop_to_size, render_plan
from reframelab.scoring import candidate_statistics
from reframelab.webapp import _validate_output_size, create_app


def test_crop_geometry_supports_portrait_square_landscape_and_ultrawide() -> None:
    portrait = crop_geometry(1920, 1080, 1080, 1920)
    assert portrait.axis == "x"
    assert (portrait.crop_width, portrait.crop_height) == (608, 1080)

    square = crop_geometry(1920, 1080, 1000, 1000)
    assert square.axis == "x"
    assert (square.crop_width, square.crop_height) == (1080, 1080)

    matching = crop_geometry(1920, 1080, 1280, 720)
    assert matching.axis == "none"
    assert (matching.crop_width, matching.crop_height) == (1920, 1080)

    landscape_from_portrait = crop_geometry(1080, 1920, 1920, 1080)
    assert landscape_from_portrait.axis == "y"
    assert (landscape_from_portrait.crop_width, landscape_from_portrait.crop_height) == (
        1080,
        608,
    )

    ultrawide = crop_geometry(1920, 1080, 2350, 1000)
    assert ultrawide.axis == "y"
    assert ultrawide.crop_width == 1920
    assert ultrawide.crop_height == 817



def test_split_orientation_adapts_when_the_natural_panel_would_duplicate_the_source() -> None:
    # A strong landscape-to-vertical conversion has enough horizontal crop in each
    # top/bottom panel, so the natural stacked layout remains useful.
    vertical_axis = choose_layout_split_axis(1920, 1080, 540, 960)
    assert vertical_axis == "y"
    vertical_fraction = panel_crop_fraction(
        1920,
        1080,
        540,
        960,
        active_axis="x",
        split_axis=vertical_axis,
        panel_fraction=0.5,
    )
    assert vertical_fraction == pytest.approx(0.6328125)

    # A 2:1 source converted to square would make top/bottom panels 2:1, showing
    # the full source twice. The layout switches to left/right detail panels.
    square_axis = choose_layout_split_axis(2000, 1000, 1000, 1000)
    assert square_axis == "x"
    square_fraction = panel_crop_fraction(
        2000,
        1000,
        1000,
        1000,
        active_axis="x",
        split_axis=square_axis,
        panel_fraction=0.5,
    )
    assert square_fraction == pytest.approx(0.25)

    # The symmetric case applies when a square source is converted to 2:1 output.
    wide_axis = choose_layout_split_axis(1000, 1000, 2000, 1000)
    assert wide_axis == "y"
    wide_fraction = panel_crop_fraction(
        1000,
        1000,
        2000,
        1000,
        active_axis="y",
        split_axis=wide_axis,
        panel_fraction=0.5,
    )
    assert wide_fraction == pytest.approx(0.25)

def test_candidate_statistics_operate_on_the_vertical_axis() -> None:
    evidence = np.zeros((100, 50), dtype=np.float32)
    evidence[:25, :] = 1.0
    positions = np.asarray([0, 25, 50], dtype=np.int32)

    coverage, centering, boundary = candidate_statistics(
        evidence,
        positions,
        crop_width=50,
        axis="y",
    )

    assert coverage[0] == pytest.approx(1.0)
    assert coverage[1] == pytest.approx(0.0)
    assert coverage[2] == pytest.approx(0.0)
    assert centering[0] > centering[1]
    assert boundary.shape == coverage.shape


def test_planner_moves_the_vertical_viewport_for_wide_output() -> None:
    positions = np.asarray([0, 200, 400, 600, 700], dtype=np.int32)
    scores = np.asarray([0.0, 0.1, 0.2, 0.7, 1.0], dtype=np.float32)
    zeros = np.zeros_like(scores)
    frames = [
        FrameEvidence(
            frame_index=index * 10,
            time=float(index),
            shot_id=0,
            shot_score=0.0,
            x_positions=positions,
            crop_width=600,
            crop_height=300,
            crop_axis="y",
            signal_coverage={"saliency": scores},
            signal_centering={"saliency": zeros},
            signal_boundary={"saliency": zeros},
            signal_reliability={"saliency": 1.0},
            evaluation_score=scores.copy(),
            evaluation_coverage=scores.copy(),
        )
        for index in range(3)
    ]
    bundle = AnalysisBundle(
        video=VideoInfo("portrait.mp4", 600, 1000, 10.0, 30, 3.0),
        analysis_width=600,
        analysis_height=1000,
        target_width=1000,
        target_height=500,
        sample_fps=1.0,
        frames=frames,
        crop_axis="y",
    )
    profile = Profile(
        name="vertical-test",
        description="test",
        signal_weights={"saliency": 1.0},
    )

    plan = build_plan(bundle, profile)

    assert plan.crop_axis == "y"
    assert all(point.center_x_norm == pytest.approx(0.5) for point in plan.points)
    assert all(point.center_y_norm == pytest.approx(0.85) for point in plan.points)
    assert plan.implementation["analysis_crop_height"] == 300


def test_renderer_uses_center_y_when_the_output_is_wider_than_the_source() -> None:
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[100:, :] = 255

    top = _crop_to_size(frame, 0.5, 0.15, 200, 100)
    bottom = _crop_to_size(frame, 0.5, 0.85, 200, 100)

    assert top.shape == (100, 200, 3)
    assert bottom.shape == (100, 200, 3)
    assert float(np.mean(top)) < 10.0
    assert float(np.mean(bottom)) > 245.0


def test_web_output_size_accepts_common_and_custom_formats() -> None:
    assert _validate_output_size("960x960") == "960x960"
    assert _validate_output_size("1280x720") == "1280x720"
    assert _validate_output_size("1281x721") == "1280x720"
    assert _validate_output_size("1280x544") == "1280x544"

    with pytest.raises(ValueError):
        _validate_output_size("50x100")
    with pytest.raises(ValueError):
        _validate_output_size("4096x4096")
    with pytest.raises(ValueError):
        _validate_output_size("4096x64")


def test_web_interface_exposes_presets_and_custom_dimensions(tmp_path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'value="960x960"' in response.text
    assert 'value="1280x720"' in response.text
    assert 'value="1280x544"' in response.text
    assert 'value="custom"' in response.text
    assert 'id="customWidth"' in response.text
    assert 'id="customHeight"' in response.text
    assert ">Reframe</button>" in response.text


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required",
)
def test_end_to_end_portrait_source_to_landscape_output(tmp_path) -> None:
    source = tmp_path / "portrait-source.mp4"
    output = tmp_path / "landscape-output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=180x320:rate=10:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    bundle = analyze_video(
        source,
        target_width=320,
        target_height=180,
        analysis_height=160,
        sample_fps=4.0,
        candidate_step=4,
        semantic_people=False,
        semantic_text=False,
    )
    plan = build_plan(bundle, Profile("center", "center"))
    render_plan(source, plan, output)

    assert bundle.crop_axis == "y"
    assert plan.crop_axis == "y"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (320, 180)
    assert stream["codec_name"] == "h264"


def test_plan_interpolator_smoothly_interpolates_vertical_centres() -> None:
    from reframelab.models import CropPlan, PlanPoint

    plan = CropPlan(
        profile="vertical",
        description="vertical",
        source="portrait.mp4",
        source_width=180,
        source_height=320,
        source_fps=30.0,
        target_width=320,
        target_height=180,
        sample_fps=1.0,
        crop_axis="y",
        points=[
            PlanPoint(0, 0.0, 0, 0.5, "crop", 1.0, 1.0, center_y_norm=0.2),
            PlanPoint(30, 1.0, 0, 0.5, "crop", 1.0, 1.0, center_y_norm=0.8),
        ],
    )
    state = PlanInterpolator(plan).at(0.5)
    assert state.center_x_norm == pytest.approx(0.5)
    assert state.center_y_norm == pytest.approx(0.5)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required",
)
def test_web_custom_square_size_runs_end_to_end(tmp_path) -> None:
    from reframelab.synthetic import generate_sample

    source, _ = generate_sample(
        tmp_path / "web-square-source.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.3,
    )
    app = create_app(work_root=tmp_path / "runs", keep_days=0, max_workers=1)

    with TestClient(app) as client, source.open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": (source.name, video, "video/mp4")},
            data={
                "preset": "best",
                "quality": "quick",
                "output_size": "241x241",
            },
        )
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["output_size"] == "240x240"

        import time

        deadline = time.monotonic() + 90.0
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job['id']}").json()

        assert job["status"] == "completed", job
        assert len(job["results"]["outputs"]) == 1
        rendered = client.get(job["results"]["outputs"][0]["url"])
        assert rendered.status_code == 200
        rendered_path = tmp_path / "web-square-output.mp4"
        rendered_path.write_bytes(rendered.content)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(rendered_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (240, 240)
    assert stream["codec_name"] == "h264"


def test_analysis_preserves_matching_source_axis_after_even_resize(tmp_path) -> None:
    from reframelab.synthetic import generate_sample

    source, _ = generate_sample(
        tmp_path / "matching-source.mp4",
        width=640,
        height=360,
        fps=8.0,
        shot_seconds=0.2,
    )
    bundle = analyze_video(
        source,
        target_width=320,
        target_height=180,
        analysis_height=120,
        sample_fps=2.0,
        candidate_step=4,
        semantic_people=False,
        semantic_text=False,
    )
    assert (bundle.analysis_width, bundle.analysis_height) == (212, 120)
    assert bundle.crop_axis == "none"
    assert all(frame.crop_axis == "none" for frame in bundle.frames)
    assert all(frame.crop_width == 212 for frame in bundle.frames)
    assert all(frame.crop_height == 120 for frame in bundle.frames)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required",
)
def test_comparison_grid_uses_rendered_output_aspect(tmp_path) -> None:
    from reframelab.comparison import build_comparison_video

    source = tmp_path / "source.mp4"
    square = tmp_path / "square.mp4"
    comparison = tmp_path / "comparison.mp4"
    for path, size in [(source, "320x180"), (square, "240x240")]:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={size}:rate=10:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )

    build_comparison_video(source, {"square": square}, comparison)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(comparison),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (960, 514)
    assert stream["codec_name"] == "h264"
