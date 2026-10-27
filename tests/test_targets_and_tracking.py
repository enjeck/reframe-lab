from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from reframelab.targets import TargetAnchor, TargetSpec, parse_target_specs
from reframelab.tracking import track_targets


def _writer(path: Path, width: int, height: int, fps: float) -> cv.VideoWriter:
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("The OpenCV runtime cannot create the synthetic tracking video")
    return writer


def _background(width: int, height: int, frame_index: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (36, 42, 49)
    for x in range(0, width, 24):
        tone = 50 + ((x // 24 + frame_index // 9) % 3) * 8
        cv.line(frame, (x, 0), (x, height), (tone, tone + 3, tone + 6), 1)
    for y in range(0, height, 22):
        cv.line(frame, (0, y), (width, y), (45, 50, 57), 1)
    cv.putText(
        frame,
        f"FRAME {frame_index:03d}",
        (8, height - 9),
        cv.FONT_HERSHEY_SIMPLEX,
        0.32,
        (90, 96, 104),
        1,
        cv.LINE_AA,
    )
    return frame


def _cursor_position(frame_index: int, frame_count: int, width: int, height: int) -> tuple[int, int]:
    progress = frame_index / max(frame_count - 1, 1)
    x = int(round(width * (0.16 + 0.68 * progress)))
    y = int(round(height * (0.30 + 0.22 * math.sin(progress * math.pi * 2.0))))
    return x, y


def _draw_cursor(frame: np.ndarray, center: tuple[int, int]) -> None:
    x, y = center
    points = np.asarray(
        [
            [x - 4, y - 10],
            [x + 8, y + 1],
            [x + 2, y + 2],
            [x + 6, y + 10],
            [x + 1, y + 12],
            [x - 3, y + 4],
            [x - 8, y + 8],
        ],
        dtype=np.int32,
    )
    cv.fillPoly(frame, [points], (18, 18, 18), cv.LINE_AA)
    cv.polylines(frame, [points], True, (250, 250, 250), 2, cv.LINE_AA)


def _make_cursor_video(path: Path) -> tuple[Path, float, int, int, int, tuple[int, int]]:
    width, height, fps, frame_count = 240, 136, 10.0, 60
    hidden = range(24, 35)
    writer = _writer(path, width, height, fps)
    for frame_index in range(frame_count):
        frame = _background(width, height, frame_index)
        if frame_index not in hidden:
            _draw_cursor(frame, _cursor_position(frame_index, frame_count, width, height))
        writer.write(frame)
    writer.release()
    return path, fps, frame_count, width, height, (hidden.start, hidden.stop)


def _box_position(frame_index: int, frame_count: int, width: int, height: int) -> tuple[int, int, int, int]:
    progress = frame_index / max(frame_count - 1, 1)
    box_width, box_height = 42, 30
    x = int(round(width * (0.10 + 0.72 * progress)))
    y = int(round(height * (0.52 + 0.12 * math.sin(progress * math.pi * 2.0))))
    return x, y, box_width, box_height


def _draw_box_target(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    x, y, width, height = bbox
    cv.rectangle(frame, (x, y), (x + width, y + height), (42, 175, 235), -1)
    cv.rectangle(frame, (x + 4, y + 4), (x + width - 4, y + height - 4), (245, 245, 245), -1)
    cv.line(frame, (x + 5, y + 5), (x + width - 5, y + height - 5), (25, 65, 210), 2)
    cv.line(frame, (x + width - 5, y + 5), (x + 5, y + height - 5), (25, 65, 210), 2)
    cv.circle(frame, (x + width // 2, y + height // 2), 4, (20, 20, 25), -1)


def _make_box_video(path: Path) -> tuple[Path, float, int, int, int, tuple[int, int]]:
    width, height, fps, frame_count = 260, 146, 10.0, 60
    hidden = range(25, 34)
    writer = _writer(path, width, height, fps)
    for frame_index in range(frame_count):
        frame = _background(width, height, frame_index)
        if frame_index not in hidden:
            _draw_box_target(frame, _box_position(frame_index, frame_count, width, height))
        else:
            # A similarly sized, high-contrast distractor should not be accepted as the target.
            cv.rectangle(frame, (180, 28), (224, 60), (235, 235, 235), -1)
            cv.putText(
                frame,
                "UI",
                (188, 52),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (24, 24, 30),
                2,
                cv.LINE_AA,
            )
        writer.write(frame)
    writer.release()
    return path, fps, frame_count, width, height, (hidden.start, hidden.stop)


def test_parse_target_specs_accepts_points_boxes_and_keyframes() -> None:
    targets = parse_target_specs(
        [
            {
                "id": "cursor",
                "label": "Cursor",
                "kind": "point",
                "role": "follow",
                "anchors": [
                    {"time": 1.0, "x": 0.2, "y": 0.3},
                    {"time": 4.0, "x": 0.7, "y": 0.5},
                ],
            },
            {
                "id": "chart",
                "kind": "box",
                "role": "keep_visible",
                "anchors": [
                    {"time": 2.0, "x": 0.5, "y": 0.2, "width": 0.4, "height": 0.6}
                ],
            },
        ]
    )
    assert [target.id for target in targets] == ["cursor", "chart"]
    assert targets[0].anchors[0].center_x == pytest.approx(0.2)
    assert targets[1].anchors[0].center_x == pytest.approx(0.7)
    assert targets[1].role == "keep_visible"


def test_parse_target_specs_rejects_invalid_selection() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_target_specs(
            [
                {
                    "id": "bad",
                    "kind": "point",
                    "role": "follow",
                    "anchors": [{"time": 0.0, "x": 1.2, "y": 0.5}],
                }
            ]
        )
    with pytest.raises(ValueError, match="non-zero"):
        parse_target_specs(
            [
                {
                    "id": "bad-box",
                    "kind": "box",
                    "role": "follow",
                    "anchors": [{"time": 0.0, "x": 0.2, "y": 0.2}],
                }
            ]
        )


def test_point_tracker_reacquires_cursor_after_occlusion(tmp_path) -> None:
    path, fps, frame_count, width, height, hidden = _make_cursor_video(tmp_path / "cursor.mp4")
    anchor_frame = 8
    anchor_x, anchor_y = _cursor_position(anchor_frame, frame_count, width, height)
    target = TargetSpec(
        id="cursor",
        label="Cursor",
        kind="point",
        role="follow",
        anchors=(TargetAnchor(anchor_frame / fps, anchor_x / width, anchor_y / height),),
    )

    track = track_targets(path, [target], analysis_height=height, sample_fps=fps)[0]
    errors: list[float] = []
    visible_expected = 0
    visible_found = 0
    hidden_found = 0
    hidden_total = 0
    for observation in track.observations:
        frame_index = int(round(observation.time * fps))
        frame_index = min(max(frame_index, 0), frame_count - 1)
        is_hidden = hidden[0] <= frame_index < hidden[1]
        if is_hidden:
            hidden_total += 1
            hidden_found += int(observation.visible)
            continue
        visible_expected += 1
        if observation.visible:
            visible_found += 1
            expected_x, expected_y = _cursor_position(frame_index, frame_count, width, height)
            error = math.hypot(
                observation.center_x * width - expected_x,
                observation.center_y * height - expected_y,
            )
            errors.append(error)

    assert visible_found / visible_expected >= 0.85
    assert hidden_found / max(hidden_total, 1) <= 0.20
    assert errors and float(np.mean(errors)) < 5.0
    assert any(
        item.visible and item.time > hidden[1] / fps
        for item in track.observations
    )


def test_box_tracker_reacquires_object_after_occlusion(tmp_path) -> None:
    path, fps, frame_count, width, height, hidden = _make_box_video(tmp_path / "box.mp4")
    anchor_frame = 8
    x, y, box_width, box_height = _box_position(anchor_frame, frame_count, width, height)
    target = TargetSpec(
        id="product",
        label="Product",
        kind="box",
        role="keep_visible",
        anchors=(
            TargetAnchor(
                anchor_frame / fps,
                x / width,
                y / height,
                box_width / width,
                box_height / height,
            ),
        ),
    )

    track = track_targets(path, [target], analysis_height=height, sample_fps=fps)[0]
    errors: list[float] = []
    visible_expected = 0
    visible_found = 0
    hidden_found = 0
    hidden_total = 0
    for observation in track.observations:
        frame_index = int(round(observation.time * fps))
        frame_index = min(max(frame_index, 0), frame_count - 1)
        is_hidden = hidden[0] <= frame_index < hidden[1]
        if is_hidden:
            hidden_total += 1
            hidden_found += int(observation.visible)
            continue
        visible_expected += 1
        if observation.visible:
            visible_found += 1
            expected_x, expected_y, expected_width, expected_height = _box_position(
                frame_index,
                frame_count,
                width,
                height,
            )
            expected_center_x = expected_x + expected_width / 2.0
            expected_center_y = expected_y + expected_height / 2.0
            errors.append(
                math.hypot(
                    observation.center_x * width - expected_center_x,
                    observation.center_y * height - expected_center_y,
                )
            )

    assert visible_found / visible_expected >= 0.80
    assert hidden_found / max(hidden_total, 1) <= 0.25
    assert errors and float(np.mean(errors)) < 8.0
    assert track.metrics["reacquisition_count"] >= 1.0


def test_tracker_rejects_keyframe_after_video_end(tmp_path) -> None:
    path, _, _, _, _, _ = _make_cursor_video(tmp_path / "short.mp4")
    target = TargetSpec(
        id="late",
        label="Late target",
        kind="point",
        role="follow",
        anchors=(TargetAnchor(99.0, 0.5, 0.5),),
    )
    with pytest.raises(ValueError, match="after the video ends"):
        track_targets(path, [target], analysis_height=136, sample_fps=5.0)
