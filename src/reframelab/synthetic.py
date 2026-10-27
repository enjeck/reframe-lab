from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

from .video import even, require_ffmpeg, run_command


def _draw_header(frame: np.ndarray, text: str) -> None:
    cv.rectangle(frame, (0, 0), (frame.shape[1], 48), (18, 18, 22), -1)
    cv.putText(
        frame,
        text,
        (18, 33),
        cv.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv.LINE_AA,
    )


def _shot_moving_product(frame: np.ndarray, progress: float) -> float:
    height, width = frame.shape[:2]
    frame[:] = (32, 38, 45)
    for x in range(0, width, 80):
        cv.line(frame, (x, 48), (x, height), (45, 52, 61), 1)
    for y in range(70, height, 70):
        cv.line(frame, (0, y), (width, y), (45, 52, 61), 1)
    center_x = int(round(width * (0.16 + 0.68 * progress)))
    center_y = int(round(height * 0.55))
    box_width = int(width * 0.20)
    box_height = int(height * 0.32)
    x0, y0 = center_x - box_width // 2, center_y - box_height // 2
    x1, y1 = center_x + box_width // 2, center_y + box_height // 2
    cv.rectangle(frame, (x0 + 8, y0 + 10), (x1 + 8, y1 + 10), (10, 10, 12), -1)
    cv.rectangle(frame, (x0, y0), (x1, y1), (42, 188, 236), -1)
    cv.rectangle(frame, (x0 + 16, y0 + 18), (x1 - 16, y1 - 18), (248, 248, 248), -1)
    cv.circle(frame, (center_x, center_y - 18), int(box_height * 0.19), (40, 70, 220), -1)
    cv.putText(frame, "PRODUCT", (x0 + 24, y1 - 34), cv.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 30), 2, cv.LINE_AA)
    return center_x / width


def _shot_screen_text(frame: np.ndarray, progress: float) -> float:
    del progress
    height, width = frame.shape[:2]
    frame[:] = (224, 226, 230)
    cv.rectangle(frame, (0, 48), (int(width * 0.19), height), (39, 45, 55), -1)
    for index, label in enumerate(["HOME", "DATA", "REPORTS", "SETTINGS"]):
        y = 105 + index * 62
        cv.putText(frame, label, (22, y), cv.FONT_HERSHEY_SIMPLEX, 0.58, (210, 216, 225), 1, cv.LINE_AA)
    cv.rectangle(frame, (int(width * 0.23), 82), (int(width * 0.55), height - 45), (250, 250, 250), -1)
    cv.putText(frame, "BACKGROUND", (int(width * 0.27), 135), cv.FONT_HERSHEY_SIMPLEX, 0.62, (120, 125, 132), 2, cv.LINE_AA)
    for row in range(6):
        y = 180 + row * 45
        cv.line(frame, (int(width * 0.27), y), (int(width * 0.51), y), (190, 194, 200), 2)

    x0, y0 = int(width * 0.60), int(height * 0.14)
    x1, y1 = int(width * 0.94), int(height * 0.88)
    cv.rectangle(frame, (x0 + 7, y0 + 8), (x1 + 7, y1 + 8), (120, 120, 125), -1)
    cv.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv.putText(frame, "KEY RESULT", (x0 + 22, y0 + 58), cv.FONT_HERSHEY_SIMPLEX, 0.72, (32, 36, 46), 2, cv.LINE_AA)
    cv.putText(frame, "+42%", (x0 + 34, y0 + 145), cv.FONT_HERSHEY_SIMPLEX, 1.55, (34, 145, 72), 4, cv.LINE_AA)
    cv.putText(frame, "CONVERSION", (x0 + 24, y0 + 204), cv.FONT_HERSHEY_SIMPLEX, 0.60, (60, 64, 74), 2, cv.LINE_AA)
    chart_left, chart_top = x0 + 30, y0 + 250
    chart_right, chart_bottom = x1 - 24, y1 - 38
    points = []
    for index in range(7):
        px = int(chart_left + (chart_right - chart_left) * index / 6)
        py = int(chart_bottom - (chart_bottom - chart_top) * (0.18 + 0.72 * index / 6))
        points.append((px, py))
    cv.polylines(frame, [np.array(points, dtype=np.int32)], False, (220, 90, 45), 4, cv.LINE_AA)
    return (x0 + x1) / 2.0 / width


def _shot_camera_motion(frame: np.ndarray, progress: float, frame_number: int) -> float:
    height, width = frame.shape[:2]
    frame[:] = (52, 66, 62)
    shift = (frame_number * 9) % 120
    for x in range(-120 + shift, width + 120, 120):
        cv.rectangle(frame, (x, 48), (x + 54, height), (64, 82, 75), -1)
        cv.line(frame, (x + 78, 48), (x + 78, height), (84, 98, 90), 2)
    for y in range(100, height, 90):
        cv.line(frame, (0, y), (width, y), (75, 88, 82), 1)

    center_x = int(round(width * (0.24 + 0.40 * progress)))
    center_y = int(height * (0.58 + 0.04 * np.sin(progress * np.pi * 4)))
    cv.ellipse(frame, (center_x, center_y), (92, 58), 0, 0, 360, (32, 116, 235), -1)
    cv.circle(frame, (center_x - 34, center_y + 48), 19, (18, 18, 22), -1)
    cv.circle(frame, (center_x + 42, center_y + 48), 19, (18, 18, 22), -1)
    cv.putText(frame, "ROVER", (center_x - 62, center_y + 8), cv.FONT_HERSHEY_SIMPLEX, 0.62, (250, 250, 250), 2, cv.LINE_AA)
    return center_x / width


def _shot_impossible_crop(frame: np.ndarray, progress: float) -> None:
    del progress
    height, width = frame.shape[:2]
    frame[:] = (28, 31, 42)
    left = (int(width * 0.04), int(height * 0.18), int(width * 0.35), int(height * 0.84))
    right = (int(width * 0.65), int(height * 0.18), int(width * 0.96), int(height * 0.84))
    for bounds, title, value, color in [
        (left, "INPUT", "18.2", (204, 112, 38)),
        (right, "OUTPUT", "31.7", (42, 166, 88)),
    ]:
        x0, y0, x1, y1 = bounds
        cv.rectangle(frame, (x0 + 6, y0 + 7), (x1 + 6, y1 + 7), (8, 8, 10), -1)
        cv.rectangle(frame, (x0, y0), (x1, y1), (246, 246, 248), -1)
        cv.putText(frame, title, (x0 + 24, y0 + 62), cv.FONT_HERSHEY_SIMPLEX, 0.78, (45, 48, 58), 2, cv.LINE_AA)
        cv.putText(frame, value, (x0 + 25, y0 + 170), cv.FONT_HERSHEY_SIMPLEX, 1.55, color, 4, cv.LINE_AA)
        cv.putText(frame, "KEEP BOTH", (x0 + 24, y1 - 42), cv.FONT_HERSHEY_SIMPLEX, 0.56, (65, 68, 78), 2, cv.LINE_AA)
    cv.arrowedLine(frame, (int(width * 0.39), int(height * 0.53)), (int(width * 0.61), int(height * 0.53)), (235, 235, 235), 5, tipLength=0.18)


def generate_sample(
    output_path: str | Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: float = 24.0,
    shot_seconds: float = 3.0,
) -> tuple[Path, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = even(width), even(height)
    total_shots = 4
    total_seconds = shot_seconds * total_shots
    total_frames = int(round(total_seconds * fps))

    with tempfile.TemporaryDirectory(prefix="reframe-sample-") as temp_dir:
        silent_path = Path(temp_dir) / "sample_silent.mp4"
        writer = cv.VideoWriter(
            str(silent_path),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the synthetic sample video")

        for frame_index in range(total_frames):
            time_seconds = frame_index / fps
            shot = min(total_shots - 1, int(time_seconds // shot_seconds))
            local_time = time_seconds - shot * shot_seconds
            progress = float(np.clip(local_time / shot_seconds, 0.0, 1.0))
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if shot == 0:
                _shot_moving_product(frame, progress)
            elif shot == 1:
                _shot_screen_text(frame, progress)
            elif shot == 2:
                _shot_camera_motion(frame, progress, frame_index)
            else:
                _shot_impossible_crop(frame, progress)
            writer.write(frame)
        writer.release()

        ffmpeg = require_ffmpeg()
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent_path),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=330:sample_rate=48000:duration={total_seconds}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
        run_command(command)

    ground_truth = output.with_suffix(".ground_truth.json")
    ground_truth.write_text(
        json.dumps(
            {
                "version": 1,
                "source": output.name,
                "target_aspect": 9 / 16,
                "segments": [
                    {
                        "start": 0.0,
                        "end": shot_seconds,
                        "mode": "crop",
                        "center_start": 0.16,
                        "center_end": 0.84,
                        "label": "moving product",
                    },
                    {
                        "start": shot_seconds,
                        "end": shot_seconds * 2,
                        "mode": "fallback",
                        "label": "screen layout with separated context and result",
                    },
                    {
                        "start": shot_seconds * 2,
                        "end": shot_seconds * 3,
                        "mode": "crop",
                        "center_start": 0.24,
                        "center_end": 0.64,
                        "label": "independent moving rover",
                    },
                    {
                        "start": shot_seconds * 3,
                        "end": shot_seconds * 4 + 0.001,
                        "mode": "fallback",
                        "label": "two required regions",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, ground_truth
