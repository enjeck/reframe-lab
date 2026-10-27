from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from reframelab.advanced_cv import DenseFlowEstimator, DepthEstimator, PromptSegmenter
from reframelab.analyzer import analyze_video
from reframelab.model_assets import (
    CORE_MODEL_NAMES,
    EXPERIMENTAL_MODEL_NAMES,
    MODEL_ASSETS,
)
from reframelab.pipeline import prepare_tracking_review, run_demo
from reframelab.profiles import (
    DEFAULT_DEMO_PROFILES,
    EVALUATION_WEIGHTS,
    PROFILES,
    TARGET_DEMO_PROFILES,
)
from reframelab.tracking import track_targets
from reframelab.webapp import _INDEX_HTML, QUALITY_PRESETS, RUN_PRESETS


def _default(callable_object, name: str):
    return inspect.signature(callable_object).parameters[name].default


def test_production_defaults_do_not_enable_experimental_models() -> None:
    assert _default(analyze_video, "advanced_vision") is False
    assert _default(analyze_video, "enable_dense_flow") is False
    assert _default(analyze_video, "enable_depth") is False
    assert _default(run_demo, "enable_learned_tracking") is False
    assert _default(run_demo, "advanced_vision") is False
    assert _default(run_demo, "enable_depth") is False
    assert _default(prepare_tracking_review, "enable_learned_tracking") is False
    assert _default(prepare_tracking_review, "advanced_vision") is False
    assert _default(track_targets, "enable_learned") is False
    assert _default(track_targets, "enable_masks") is False
    assert _default(track_targets, "enable_dense_flow") is False
    assert _default(DenseFlowEstimator, "enable_learned") is False
    assert _default(DepthEstimator, "enable_learned") is False
    assert _default(PromptSegmenter, "enable_learned") is False


def test_product_profiles_exclude_depth_and_keep_small_comparisons() -> None:
    assert "depth" not in EVALUATION_WEIGHTS
    assert "depth_raw" not in PROFILES
    assert all("depth" not in profile.signal_weights for profile in PROFILES.values())
    assert DEFAULT_DEMO_PROFILES == ["center", "hybrid_safe"]
    assert TARGET_DEMO_PROFILES == ["center", "hybrid_safe", "target_guided_safe"]


def test_model_groups_are_explicit_and_complete() -> None:
    core = set(CORE_MODEL_NAMES)
    experimental = set(EXPERIMENTAL_MODEL_NAMES)
    assert core == {"person_mediapipe", "text_ppocr", "yunet"}
    assert experimental == {
        "efficient_sam",
        "raft",
        "depth_anything_v2_small",
        "vittrack",
    }
    assert not core & experimental
    assert core | experimental == set(MODEL_ASSETS)


def test_web_ui_exposes_only_product_modes() -> None:
    assert "Compare all methods" not in _INDEX_HTML
    assert "adds depth" not in _INDEX_HTML
    assert "dense motion" not in _INDEX_HTML
    assert RUN_PRESETS["all"]["automatic_profiles"] == ["hybrid_safe", "hybrid_temporal", "center"]
    assert RUN_PRESETS["all"]["target_profiles"][0] == "target_guided_safe"
    assert RUN_PRESETS["fast_compare"]["automatic_profiles"] == [
        "center",
        "hybrid_safe",
    ]
    assert QUALITY_PRESETS["balanced"] == {
        "analysis_height": 240,
        "analysis_fps": 5.0,
        "candidate_step": 8,
        "adaptive_dense_flow": True,
    }


@pytest.mark.skipif(not Path("tests").is_dir(), reason="source tree test")
def test_experimental_switch_remains_available_for_research() -> None:
    assert "advanced_vision" in inspect.signature(run_demo).parameters
    assert "enable_depth" in inspect.signature(run_demo).parameters
    assert "enable_learned_tracking" in inspect.signature(run_demo).parameters
