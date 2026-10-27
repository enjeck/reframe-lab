from __future__ import annotations

import numpy as np

from reframelab.scoring import candidate_positions, candidate_statistics, combine_signals


def test_candidate_positions_include_last_valid_crop() -> None:
    positions = candidate_positions(frame_width=101, crop_width=30, step=16)
    assert positions[0] == 0
    assert positions[-1] == 71
    assert np.all(np.diff(positions) > 0)


def test_candidate_statistics_selects_bright_region() -> None:
    importance = np.zeros((20, 100), dtype=np.float32)
    importance[:, 70:90] = 1.0
    positions = candidate_positions(100, 30, 5)
    coverage, centering, boundary = candidate_statistics(importance, positions, 30)
    best = int(np.argmax(coverage + 0.1 * centering - 0.18 * boundary))
    assert 60 <= int(positions[best]) <= 70
    assert float(coverage[best]) > 0.95


def test_combine_signals_uses_reliability() -> None:
    x_positions = np.array([0, 10], dtype=np.int32)
    coverage = {
        "saliency": np.array([0.9, 0.1], dtype=np.float32),
        "motion": np.array([0.1, 0.9], dtype=np.float32),
    }
    centering = {name: np.ones(2, dtype=np.float32) for name in coverage}
    boundary = {name: np.zeros(2, dtype=np.float32) for name in coverage}
    score, _ = combine_signals(
        x_positions=x_positions,
        crop_width=10,
        frame_width=20,
        weights={"saliency": 0.5, "motion": 0.5},
        coverage=coverage,
        centering=centering,
        boundary=boundary,
        reliability={"saliency": 1.0, "motion": 0.0},
    )
    assert int(np.argmax(score)) == 0
