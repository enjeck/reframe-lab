from __future__ import annotations

import numpy as np
import pytest

from reframelab.models import (
    AnalysisBundle,
    CropPlan,
    FrameEvidence,
    PlanPoint,
    Profile,
    TextRegion,
    VideoInfo,
)
from reframelab.planner import (
    _dominant_text_requires_full_context,
    _text_split_centers,
    build_plan,
)
from reframelab.renderer import PlanInterpolator
from reframelab.scoring import combine_signals, text_region_statistics
from reframelab.text import _screen_text_prior


def _plan(points: list[PlanPoint], *, transition: float = 0.5) -> CropPlan:
    return CropPlan(
        profile="test",
        description="test",
        source="input.mp4",
        source_width=1920,
        source_height=1080,
        source_fps=30.0,
        target_width=540,
        target_height=960,
        sample_fps=1.0,
        points=points,
        implementation={"layout_transition_seconds": transition},
    )


def test_hold_to_pan_is_interpolated_without_a_snap() -> None:
    interpolator = PlanInterpolator(
        _plan(
            [
                PlanPoint(0, 0.0, 0, 0.25, "hold", 1.0, 1.0),
                PlanPoint(30, 1.0, 0, 0.75, "pan", 1.0, 1.0),
            ]
        )
    )
    before = interpolator.at(0.49)
    middle = interpolator.at(0.50)
    after = interpolator.at(0.51)
    assert before.blend_from is None
    assert middle.blend_from is None
    assert after.blend_from is None
    assert 0.25 < before.center_x_norm < middle.center_x_norm < after.center_x_norm < 0.75
    assert after.center_x_norm - before.center_x_norm < 0.04


def test_layout_changes_crossfade_instead_of_switching_in_one_frame() -> None:
    interpolator = PlanInterpolator(
        _plan(
            [
                PlanPoint(0, 0.0, 0, 0.25, "crop", 1.0, 1.0),
                PlanPoint(
                    30,
                    1.0,
                    0,
                    0.25,
                    "split",
                    1.0,
                    1.0,
                    secondary_center_x_norm=0.75,
                ),
            ],
            transition=0.6,
        )
    )
    state = interpolator.at(0.5)
    assert state.blend_from is not None
    assert state.blend_from.mode == "crop"
    assert state.point.mode == "split"
    assert state.blend_from.center_x_norm == pytest.approx(0.25)
    assert state.point.center_x_norm == pytest.approx(0.25)
    assert 0.35 < state.blend_alpha < 0.65



def test_layout_crossfade_does_not_pan_either_layout() -> None:
    interpolator = PlanInterpolator(
        _plan(
            [
                PlanPoint(0, 0.0, 0, 0.20, "split", 1.0, 1.0, secondary_center_x_norm=0.75),
                PlanPoint(30, 1.0, 0, 0.80, "fallback", 1.0, 1.0),
            ],
            transition=0.6,
        )
    )
    state = interpolator.at(0.5)
    assert state.blend_from is not None
    assert state.blend_from.center_x_norm == pytest.approx(0.20)
    assert state.point.center_x_norm == pytest.approx(0.80)
    assert 0.35 < state.blend_alpha < 0.65

def test_text_box_integrity_prefers_a_complete_line_over_a_cut_line() -> None:
    regions = [TextRegion(x=0.25, y=0.15, width=0.30, height=0.06, score=1.0)]
    positions = np.asarray([0, 200, 400], dtype=np.int32)
    coverage, centering, boundary = text_region_statistics(
        regions,
        positions,
        crop_width=400,
        frame_width=1000,
    )
    scores, _ = combine_signals(
        positions,
        crop_width=400,
        frame_width=1000,
        weights={"text_complete": 1.0},
        coverage={"text_complete": coverage},
        centering={"text_complete": centering},
        boundary={"text_complete": boundary},
        reliability={"text_complete": 1.0},
    )
    assert coverage[1] == pytest.approx(1.0)
    assert boundary[1] < boundary[0]
    assert boundary[1] < boundary[2]
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_separated_text_groups_use_a_stacked_layout() -> None:
    positions = np.asarray([0, 200, 400, 600], dtype=np.int32)
    poor_text_coverage = np.asarray([0.48, 0.50, 0.49, 0.47], dtype=np.float32)
    zeros = np.zeros_like(poor_text_coverage)
    text_regions = [
        TextRegion(0.04, 0.12, 0.18, 0.05, 0.95),
        TextRegion(0.06, 0.24, 0.20, 0.05, 0.90),
        TextRegion(0.74, 0.12, 0.18, 0.05, 0.94),
        TextRegion(0.72, 0.24, 0.20, 0.05, 0.91),
    ]
    frames = [
        FrameEvidence(
            frame_index=index * 10,
            time=index * 0.4,
            shot_id=0,
            shot_score=0.0,
            x_positions=positions,
            crop_width=400,
            signal_coverage={"text_complete": poor_text_coverage},
            signal_centering={"text_complete": zeros},
            signal_boundary={"text_complete": zeros},
            signal_reliability={"text_complete": 1.0},
            evaluation_score=poor_text_coverage.copy(),
            evaluation_coverage=poor_text_coverage.copy(),
            text_regions=text_regions,
            text_backend="opencv_dnn_ppocr_text",
        )
        for index in range(4)
    ]
    bundle = AnalysisBundle(
        video=VideoInfo("ui.mp4", 1000, 600, 25.0, 100, 4.0),
        analysis_width=1000,
        analysis_height=600,
        target_width=540,
        target_height=960,
        sample_fps=2.5,
        frames=frames,
    )
    profile = Profile(
        name="text_safe",
        description="test",
        signal_weights={"text_complete": 1.0},
        temporal=True,
        stabilize=True,
        allow_fallback=True,
        semantic_layouts=True,
        text_integrity_threshold=0.90,
        text_split_min_regions=4,
    )
    plan = build_plan(bundle, profile)
    assert {point.mode for point in plan.points} == {"split"}
    assert all(point.secondary_center_x_norm is not None for point in plan.points)
    assert plan.metrics["text_region_cut_fraction"] == pytest.approx(0.0)


def test_classical_text_prior_rejects_colourful_natural_texture() -> None:
    rng = np.random.default_rng(7)
    natural = rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
    ui = np.full((180, 320, 3), 245, dtype=np.uint8)
    ui[20:160, 20:145] = 228
    ui[20:160, 175:300] = 235
    ui[40:44, 35:130] = 20
    ui[70:74, 35:125] = 20
    ui[40:44, 190:285] = 20
    ui[70:74, 190:280] = 20
    assert _screen_text_prior(natural) < 0.35
    assert _screen_text_prior(ui) > 0.70


def test_low_confidence_classical_boxes_do_not_force_split_layout() -> None:
    frame = FrameEvidence(
        frame_index=0,
        time=0.0,
        shot_id=0,
        shot_score=0.0,
        x_positions=np.asarray([0, 200, 400, 600], dtype=np.int32),
        crop_width=400,
        signal_coverage={"text_complete": np.asarray([0.4, 0.5, 0.5, 0.4], dtype=np.float32)},
        signal_centering={"text_complete": np.zeros(4, dtype=np.float32)},
        signal_boundary={"text_complete": np.zeros(4, dtype=np.float32)},
        signal_reliability={"text_complete": 0.18},
        evaluation_score=np.zeros(4, dtype=np.float32),
        evaluation_coverage=np.zeros(4, dtype=np.float32),
        text_regions=[
            TextRegion(0.04, 0.12, 0.18, 0.05, 0.90),
            TextRegion(0.06, 0.24, 0.20, 0.05, 0.88),
            TextRegion(0.74, 0.12, 0.18, 0.05, 0.91),
            TextRegion(0.72, 0.24, 0.20, 0.05, 0.89),
        ],
        text_backend="opencv_classical_text",
    )
    assert _text_split_centers(frame, 0.4, 4, 0.9) is None


def test_large_natural_texture_blob_is_not_treated_as_a_title() -> None:
    frame = FrameEvidence(
        frame_index=0,
        time=0.0,
        shot_id=0,
        shot_score=0.0,
        x_positions=np.asarray([0, 200], dtype=np.int32),
        crop_width=300,
        signal_coverage={},
        signal_centering={},
        signal_boundary={},
        signal_reliability={"text_complete": 0.18},
        evaluation_score=np.zeros(2, dtype=np.float32),
        evaluation_coverage=np.zeros(2, dtype=np.float32),
        text_regions=[TextRegion(0.0, 0.15, 0.49, 0.24, 0.84)],
        text_backend="opencv_classical_text",
    )
    assert not _dominant_text_requires_full_context(frame, 0.31)


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None
    or __import__("shutil").which("ffprobe") is None,
    reason="FFmpeg is required",
)
def test_comparison_normalizes_near_equal_frame_rates(tmp_path) -> None:
    import json
    import subprocess
    from pathlib import Path

    from reframelab.comparison import build_comparison_video

    source = Path(tmp_path) / "source.mp4"
    portrait = Path(tmp_path) / "portrait.mp4"
    comparison = Path(tmp_path) / "comparison.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30000/1001:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=180x320:rate=2997/100:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(portrait),
        ],
        check=True,
    )

    build_comparison_video(
        source,
        {"portrait": portrait},
        comparison,
        tile_width=90,
        tile_height=160,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(comparison),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["r_frame_rate"] == "30000/1001"
    assert int(stream["nb_frames"]) in {36, 37}

    decode = subprocess.run(
        ["ffmpeg", "-v", "warning", "-i", str(comparison), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert decode.returncode == 0
    assert "non monotonically increasing" not in decode.stderr
