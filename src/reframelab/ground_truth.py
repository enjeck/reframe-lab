from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import crop_geometry
from .models import CropPlan


def find_ground_truth(input_path: str | Path) -> Path | None:
    source = Path(input_path)
    candidates = [
        source.with_suffix(".ground_truth.json"),
        Path(str(source) + ".ground_truth.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def evaluate_ground_truth(plan: CropPlan, ground_truth_path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    geometry = crop_geometry(
        plan.source_width,
        plan.source_height,
        plan.target_width,
        plan.target_height,
    )
    axis = str(payload.get("crop_axis", plan.crop_axis or geometry.axis))
    if axis not in {"x", "y"}:
        axis = geometry.axis if geometry.axis in {"x", "y"} else "x"
    crop_fraction = geometry.crop_fraction

    errors: list[float] = []
    mode_matches: list[float] = []
    fallback_expected = 0
    fallback_correct = 0

    for point in plan.points:
        segment = next(
            (
                item
                for item in segments
                if float(item["start"]) <= point.time < float(item["end"])
            ),
            None,
        )
        if segment is None:
            continue
        expected_mode = str(segment["mode"])
        actual_fallback = point.mode in {"fallback", "context_detail", "split"}
        expected_fallback = expected_mode == "fallback"
        mode_matches.append(float(actual_fallback == expected_fallback))

        if expected_fallback:
            fallback_expected += 1
            fallback_correct += int(actual_fallback)
            continue

        duration = max(float(segment["end"]) - float(segment["start"]), 1e-6)
        alpha = float(np.clip((point.time - float(segment["start"])) / duration, 0.0, 1.0))
        start_key = "center_y_start" if axis == "y" else "center_start"
        end_key = "center_y_end" if axis == "y" else "center_end"
        if start_key not in segment or end_key not in segment:
            # Legacy fixtures use center_start/center_end for horizontal paths.
            start_key, end_key = "center_start", "center_end"
        expected_center = (
            float(segment[start_key]) * (1.0 - alpha)
            + float(segment[end_key]) * alpha
        )
        actual_center = point.center_y_norm if axis == "y" else point.center_x_norm
        errors.append(abs(actual_center - expected_center) / max(crop_fraction, 1e-6))

    mean_error = float(np.mean(errors)) if errors else 0.0
    p90_error = float(np.percentile(errors, 90.0)) if errors else 0.0
    return {
        "ground_truth_mean_center_error_viewports": mean_error,
        "ground_truth_p90_center_error_viewports": p90_error,
        # Backward-compatible aliases used by older reports and tests.
        "ground_truth_mean_center_error_crop_widths": mean_error,
        "ground_truth_p90_center_error_crop_widths": p90_error,
        "ground_truth_mode_accuracy": float(np.mean(mode_matches)) if mode_matches else 0.0,
        "ground_truth_fallback_recall": (
            float(fallback_correct / fallback_expected) if fallback_expected else 1.0
        ),
    }
