from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2 as cv
import numpy as np

from .runtime import opencv_runtime_info


def _load_frames(path: Path | None, count: int, width: int, height: int) -> list[np.ndarray]:
    if path is None:
        rng = np.random.default_rng(2601)
        frames: list[np.ndarray] = []
        for index in range(count):
            image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            cv.rectangle(
                image,
                (int(width * (0.08 + 0.55 * index / max(count - 1, 1))), int(height * 0.28)),
                (int(width * (0.25 + 0.55 * index / max(count - 1, 1))), int(height * 0.72)),
                (245, 245, 245),
                -1,
            )
            cv.putText(
                image,
                f"Frame {index}",
                (40, 70),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2,
                cv.LINE_AA,
            )
            frames.append(image)
        return frames

    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open benchmark video: {path}")
    total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for index in range(count):
            if total > 0:
                position = int(round(index * max(total - 1, 0) / max(count - 1, 1)))
                capture.set(cv.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError("Benchmark video did not provide at least two frames")
    return frames


def _measure(operation: Callable[[], None], iterations: int) -> dict[str, float]:
    for _ in range(2):
        operation()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        values.append((time.perf_counter() - start) * 1000.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(np.mean(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "iterations": float(iterations),
    }


def benchmark_opencv(
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    frame_count: int = 12,
    iterations: int = 8,
    synthetic_width: int = 1280,
    synthetic_height: int = 720,
) -> dict[str, object]:
    """Benchmark OpenCV kernels used by the production reframing pipeline."""

    frames = _load_frames(
        Path(input_path) if input_path is not None else None,
        max(3, frame_count),
        synthetic_width,
        synthetic_height,
    )
    resized = [cv.resize(frame, (640, 360), interpolation=cv.INTER_AREA) for frame in frames]
    grays = [cv.cvtColor(frame, cv.COLOR_BGR2GRAY) for frame in resized]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    points = cv.goodFeaturesToTrack(grays[0], maxCorners=600, qualityLevel=0.01, minDistance=5)

    def resize_op() -> None:
        for frame in frames:
            cv.resize(frame, (640, 360), interpolation=cv.INTER_AREA)

    def color_op() -> None:
        for frame in resized:
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def blur_sobel_op() -> None:
        for gray in grays:
            smooth = cv.GaussianBlur(gray, (5, 5), 0)
            cv.Sobel(smooth, cv.CV_32F, 1, 0, ksize=3)

    def threshold_morphology_op() -> None:
        for gray in grays:
            binary = cv.adaptiveThreshold(
                gray,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                31,
                7,
            )
            cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    def contours_op() -> None:
        for gray in grays:
            edges = cv.Canny(gray, 80, 160)
            cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    def sparse_flow_op() -> None:
        if points is None:
            return
        for previous, current in zip(grays[:-1], grays[1:], strict=False):
            cv.calcOpticalFlowPyrLK(previous, current, points, None)

    def farneback_op() -> None:
        for previous, current in zip(grays[:-1], grays[1:], strict=False):
            cv.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )

    def production_mix_op() -> None:
        previous_gray = None
        for frame in frames:
            small = cv.resize(frame, (640, 360), interpolation=cv.INTER_AREA)
            gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
            smooth = cv.GaussianBlur(gray, (5, 5), 0)
            gradient = cv.Sobel(smooth, cv.CV_32F, 1, 0, ksize=3)
            magnitude = cv.convertScaleAbs(gradient)
            _, binary = cv.threshold(magnitude, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if previous_gray is not None:
                local_points = cv.goodFeaturesToTrack(
                    previous_gray,
                    maxCorners=450,
                    qualityLevel=0.01,
                    minDistance=5,
                )
                if local_points is not None:
                    cv.calcOpticalFlowPyrLK(previous_gray, gray, local_points, None)
            previous_gray = gray

    operations = {
        "resize": resize_op,
        "color_conversion": color_op,
        "gaussian_sobel": blur_sobel_op,
        "adaptive_threshold_morphology": threshold_morphology_op,
        "contours": contours_op,
        "sparse_lk_flow": sparse_flow_op,
        "farneback_flow": farneback_op,
        "production_mix": production_mix_op,
    }
    results = {name: _measure(operation, max(1, iterations)) for name, operation in operations.items()}
    mix_ms = float(results["production_mix"]["median_ms"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "runtime": opencv_runtime_info(),
        "input": str(Path(input_path).resolve()) if input_path is not None else "synthetic",
        "frame_count": len(frames),
        "source_shape": list(frames[0].shape),
        "operations": results,
        "production_mix_frames_per_second": float(len(frames) * 1000.0 / max(mix_ms, 1e-9)),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_benchmarks(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    rows: dict[str, dict[str, float]] = {}
    for name, baseline_values in baseline.get("operations", {}).items():
        candidate_values = candidate.get("operations", {}).get(name)
        if not candidate_values:
            continue
        baseline_ms = float(baseline_values["median_ms"])
        candidate_ms = float(candidate_values["median_ms"])
        rows[name] = {
            "baseline_median_ms": baseline_ms,
            "candidate_median_ms": candidate_ms,
            "speedup_x": float(baseline_ms / max(candidate_ms, 1e-9)),
            "latency_reduction_fraction": float(1.0 - candidate_ms / max(baseline_ms, 1e-9)),
        }
    payload = {
        "schema_version": 1,
        "baseline_runtime": baseline.get("runtime", {}),
        "candidate_runtime": candidate.get("runtime", {}),
        "operations": rows,
        "production_mix_speedup_x": float(
            float(candidate.get("production_mix_frames_per_second", 0.0))
            / max(float(baseline.get("production_mix_frames_per_second", 0.0)), 1e-9)
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
