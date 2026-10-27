from __future__ import annotations

import cv2 as cv
import numpy as np

from reframelab.signals import (
    residual_motion_map,
    shot_change_score,
    spectral_residual_saliency,
    textness_map,
)


def test_spectral_saliency_finds_distinct_region() -> None:
    frame = np.full((120, 220, 3), 30, dtype=np.uint8)
    cv.rectangle(frame, (160, 35), (205, 90), (245, 245, 245), -1)
    saliency = spectral_residual_saliency(frame)
    assert saliency.shape == frame.shape[:2]
    assert np.isfinite(saliency).all()
    assert float(saliency[:, 145:].mean()) > float(saliency[:, :90].mean())


def test_textness_map_detects_horizontal_text() -> None:
    frame = np.full((120, 320, 3), 245, dtype=np.uint8)
    cv.putText(frame, "IMPORTANT RESULT", (18, 72), cv.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv.LINE_AA)
    text_map, reliability = textness_map(frame)
    assert text_map.shape == frame.shape[:2]
    assert reliability > 0.05
    assert float(text_map.max()) > 0.5


def test_residual_motion_highlights_changed_object() -> None:
    previous = np.full((120, 220, 3), 45, dtype=np.uint8)
    current = previous.copy()
    for x in range(10, 220, 30):
        cv.line(previous, (x, 0), (x, 119), (60, 60, 60), 1)
        cv.line(current, (x, 0), (x, 119), (60, 60, 60), 1)
    cv.rectangle(previous, (45, 45), (80, 80), (240, 80, 30), -1)
    cv.rectangle(current, (105, 45), (140, 80), (240, 80, 30), -1)
    motion, reliability = residual_motion_map(previous, current)
    changed = np.concatenate((motion[35:90, 35:90].ravel(), motion[35:90, 95:150].ravel()))
    background = motion[:, 165:210].ravel()
    assert reliability > 0.05
    assert float(changed.mean()) > float(background.mean()) + 0.05


def test_shot_change_score_separates_identical_and_different_frames() -> None:
    first = np.zeros((80, 120, 3), dtype=np.uint8)
    second = np.full((80, 120, 3), 240, dtype=np.uint8)
    assert shot_change_score(first, first.copy()) < 0.01
    assert shot_change_score(first, second) > 0.4
