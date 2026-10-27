from __future__ import annotations

from dataclasses import replace

import cv2 as cv
import numpy as np

from reframelab.advanced_cv import (
    DenseFlowEstimator,
    DepthEstimator,
    PromptSegmenter,
    mask_geometry,
)
from reframelab.planner import _minimum_jerk_centers, _stabilize_centers
from reframelab.profiles import get_profile


def test_dense_flow_removes_global_translation_and_keeps_local_motion() -> None:
    rng = np.random.default_rng(7)
    height, width = 120, 180
    texture = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    previous = cv.cvtColor(texture, cv.COLOR_GRAY2BGR)
    transform = np.float32([[1, 0, 4], [0, 1, 1]])
    current = cv.warpAffine(previous, transform, (width, height), borderMode=cv.BORDER_REFLECT)
    cv.rectangle(current, (112, 42), (145, 78), (255, 255, 255), -1)
    cv.rectangle(previous, (98, 42), (131, 78), (255, 255, 255), -1)

    estimator = DenseFlowEstimator(enable_learned=False, allow_download=False)
    result = estimator.residual_motion(previous, current)

    assert result.backend == "farneback_fallback"
    assert result.flow.shape == (height, width, 2)
    assert result.residual_map.shape == (height, width)
    object_strength = float(result.residual_map[38:82, 90:152].mean())
    background_strength = float(result.residual_map[8:32, 12:70].mean())
    assert object_strength > background_strength * 1.35
    assert result.reliability > 0.05


def test_grabcut_prompt_produces_mask_geometry() -> None:
    frame = np.full((140, 220, 3), 32, dtype=np.uint8)
    cv.ellipse(frame, (126, 73), (36, 24), 0, 0, 360, (30, 160, 240), -1)
    cv.circle(frame, (115, 68), 6, (245, 245, 245), -1)

    segmenter = PromptSegmenter(enable_learned=False, allow_download=False)
    result = segmenter.segment(frame, bbox=(84, 43, 84, 60))
    polygon, profile_x, profile_y, bbox = mask_geometry(result.mask)

    assert result.backend == "grabcut_fallback"
    assert bbox is not None
    assert len(polygon) >= 3
    assert len(profile_x) == 48
    assert len(profile_y) == 48
    assert abs(sum(profile_x) - 1.0) < 1e-4
    assert abs(sum(profile_y) - 1.0) < 1e-4
    assert np.count_nonzero(result.mask) > 900


class _FakeEfficientSamNet:
    def __init__(self) -> None:
        self.inputs: dict[str, np.ndarray] = {}
        self.forward_names: list[str] | None = None

    def setInput(self, blob: np.ndarray, name: str) -> None:  # noqa: N802
        self.inputs[name] = np.asarray(blob).copy()

    def forward(self, names: list[str]) -> list[np.ndarray]:
        self.forward_names = list(names)
        masks = np.full((1, 1, 4, 64, 64), -2.0, dtype=np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        masks[0, 0, 2, (xx - 36) ** 2 + (yy - 31) ** 2 <= 13 ** 2] = 3.0
        ious = np.array([[[0.08, 0.21, 0.91, 0.17]]], dtype=np.float32)
        return [masks, ious]

    def getUnconnectedOutLayersNames(self) -> list[str]:  # noqa: N802
        return ["output_masks", "iou_predictions"]


def test_efficient_sam_uses_opencv_model_tensor_contract() -> None:
    segmenter = PromptSegmenter.__new__(PromptSegmenter)
    segmenter.model_path = None
    fake_net = _FakeEfficientSamNet()
    segmenter.net = fake_net
    segmenter.backend = "mock_opencv_dnn_efficient_sam"
    frame = np.full((96, 160, 3), 40, dtype=np.uint8)

    result = segmenter.segment(frame, bbox=(60, 30, 50, 38))

    assert fake_net.forward_names == ["output_masks", "iou_predictions"]
    assert fake_net.inputs["batched_images"].shape == (1, 3, 1024, 1024)
    assert fake_net.inputs["batched_point_coords"].shape == (1, 1, 6, 2)
    assert fake_net.inputs["batched_point_labels"].shape == (1, 1, 6, 1)
    assert fake_net.inputs["batched_point_labels"][0, 0, 0, 0] == 2
    assert fake_net.inputs["batched_point_labels"][0, 0, 1, 0] == 3
    assert result.backend == "mock_opencv_dnn_efficient_sam"
    assert result.confidence > 0.9
    assert np.count_nonzero(result.mask) > 500


class _FakeDepthNet:
    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802
        self.shape = blob.shape

    def forward(self) -> np.ndarray:
        y, x = np.mgrid[0:64, 0:64].astype(np.float32)
        return (0.4 * x + 0.6 * y)[None, None, ...]


def test_depth_estimator_turns_relative_depth_into_advisory_map() -> None:
    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator.model_path = None
    estimator.net = _FakeDepthNet()
    estimator.backend = "mock_depth_anything"
    frame = np.zeros((90, 150, 3), dtype=np.uint8)

    result = estimator.infer(frame)

    assert result.relative_depth.shape == (90, 150)
    assert result.importance_map.shape == (90, 150)
    assert result.reliability > 0.2
    assert float(result.importance_map[-6, -6]) > float(result.importance_map[5, 5])


def _mean_jerk(path: np.ndarray) -> float:
    if len(path) < 4:
        return 0.0
    return float(np.mean(np.abs(np.diff(path, n=3))))


def test_minimum_jerk_solver_uses_lookahead_and_reduces_snap() -> None:
    profile = replace(
        get_profile("hybrid_temporal"),
        max_speed_crop_widths_per_second=1.9,
        max_acceleration_crop_widths_per_second2=5.5,
        max_jerk_crop_widths_per_second3=20.0,
    )
    times = np.arange(40, dtype=np.float64) / 5.0
    shot_ids = np.zeros(40, dtype=np.int32)
    desired = np.full(40, 300.0, dtype=np.float32)
    desired[20:] = 700.0

    local = _stabilize_centers(desired, times, shot_ids, 1000, 300, profile)
    planned = _minimum_jerk_centers(desired, times, shot_ids, 1000, 300, profile)

    assert np.allclose(planned[:14], 300.0, atol=1.0)
    assert planned[17] > 320.0  # The global movement window starts before the step.
    assert float(np.max(np.abs(np.diff(planned)))) < float(np.max(np.abs(np.diff(local))))
    assert _mean_jerk(planned) < _mean_jerk(local) * 0.35
    assert 660.0 <= planned[-1] <= 700.0
