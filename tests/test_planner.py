from __future__ import annotations

import numpy as np

from reframelab.models import AnalysisBundle, FrameEvidence, Profile, VideoInfo
from reframelab.planner import _dynamic_programming_path, build_plan


def test_dynamic_programming_reduces_unnecessary_switches() -> None:
    scores = np.array(
        [
            [1.00, 0.20, 0.95],
            [0.95, 0.20, 1.00],
            [1.00, 0.20, 0.95],
            [0.95, 0.20, 1.00],
        ],
        dtype=np.float32,
    )
    positions = np.array([0, 50, 100], dtype=np.int32)
    raw = np.argmax(scores, axis=1)
    planned = _dynamic_programming_path(
        scores,
        positions,
        crop_width=50,
        movement_penalty=0.2,
        maximum_transition_fraction=3.0,
    )
    raw_distance = int(np.abs(np.diff(positions[raw])).sum())
    planned_distance = int(np.abs(np.diff(positions[planned])).sum())
    assert planned_distance < raw_distance


def _fallback_bundle() -> AnalysisBundle:
    video = VideoInfo("input.mp4", 300, 200, 25.0, 75, 3.0)
    positions = np.array([0, 50, 100], dtype=np.int32)
    frames = []
    for index in range(3):
        coverage = np.array([0.25, 0.30, 0.25], dtype=np.float32)
        frames.append(
            FrameEvidence(
                frame_index=index * 10,
                time=index * 0.4,
                shot_id=0,
                shot_score=0.0,
                x_positions=positions,
                crop_width=100,
                signal_coverage={"saliency": coverage},
                signal_centering={"saliency": np.ones(3, dtype=np.float32)},
                signal_boundary={"saliency": np.zeros(3, dtype=np.float32)},
                signal_reliability={"saliency": 1.0},
                evaluation_score=coverage.copy(),
                evaluation_coverage=coverage.copy(),
            )
        )
    return AnalysisBundle(video, 200, 200, 100, 200, 2.5, frames)


def test_safe_profile_falls_back_for_whole_unsafe_shot() -> None:
    bundle = _fallback_bundle()
    profile = Profile(
        name="safe",
        description="test",
        signal_weights={"saliency": 1.0},
        temporal=True,
        stabilize=True,
        allow_fallback=True,
        fallback_coverage_threshold=0.5,
    )
    plan = build_plan(bundle, profile)
    assert {point.mode for point in plan.points} == {"fallback"}
    assert plan.metrics["fallback_fraction"] == 1.0
