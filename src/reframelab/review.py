from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .targets import TargetTrack


@dataclass(frozen=True)
class ReviewPoint:
    time: float
    target_id: str
    reason: str
    confidence: float


def _periodic_point_checkpoints(
    track: TargetTrack,
    *,
    interval_seconds: float,
) -> list[ReviewPoint]:
    if track.target.kind != "point":
        return []
    visible = [observation for observation in track.observations if observation.visible]
    if not visible:
        return []
    start = visible[0].time
    end = visible[-1].time
    checkpoints: list[ReviewPoint] = []
    target_time = start + interval_seconds
    while target_time < end:
        observation = min(visible, key=lambda item: abs(item.time - target_time))
        checkpoints.append(
            ReviewPoint(
                time=observation.time,
                target_id=track.target.id,
                reason="identity checkpoint",
                confidence=observation.confidence,
            )
        )
        target_time += interval_seconds
    return checkpoints


def build_tracking_review(
    tracks: list[TargetTrack],
    *,
    minimum_spacing_seconds: float = 1.25,
    point_checkpoint_seconds: float = 12.0,
    maximum_points: int = 24,
) -> dict[str, object]:
    candidates: list[ReviewPoint] = []
    for track in tracks:
        previous = None
        for observation in track.observations:
            reasons: list[str] = []
            if observation.needs_review:
                reasons.append("tracker uncertainty")
            if not observation.visible and previous is not None and previous.visible:
                reasons.append("target lost")
            if observation.visible and observation.confidence < 0.56:
                reasons.append("low confidence")
            if observation.visible and observation.uncertainty > 0.42:
                reasons.append("ambiguous match")
            if observation.visible and previous is not None and previous.visible:
                jump = float(
                    np.hypot(
                        observation.center_x - previous.center_x,
                        observation.center_y - previous.center_y,
                    )
                )
                if jump > 0.18:
                    reasons.append("large jump")
            if reasons:
                candidates.append(
                    ReviewPoint(
                        time=observation.time,
                        target_id=track.target.id,
                        reason=", ".join(dict.fromkeys(reasons)),
                        confidence=observation.confidence,
                    )
                )
            previous = observation
        candidates.extend(
            _periodic_point_checkpoints(
                track,
                interval_seconds=max(4.0, point_checkpoint_seconds),
            )
        )

    # Cluster independently per target so two simultaneous targets are both reviewable.
    selected_by_target: dict[str, list[ReviewPoint]] = {}
    for candidate in sorted(candidates, key=lambda item: (item.target_id, item.time)):
        selected = selected_by_target.setdefault(candidate.target_id, [])
        if selected and candidate.time - selected[-1].time < minimum_spacing_seconds:
            existing = selected[-1]
            candidate_is_diagnostic = candidate.reason != "identity checkpoint"
            existing_is_diagnostic = existing.reason != "identity checkpoint"
            replace = (
                candidate_is_diagnostic and not existing_is_diagnostic
            ) or (
                candidate_is_diagnostic == existing_is_diagnostic
                and candidate.confidence < existing.confidence
            )
            if replace:
                selected[-1] = candidate
            continue
        selected.append(candidate)

    selected = sorted(
        (point for values in selected_by_target.values() for point in values),
        key=lambda item: (item.time, item.target_id),
    )[:maximum_points]

    return {
        "review_required": bool(tracks),
        "points": [asdict(point) for point in selected],
        "track_count": len(tracks),
        "flagged_count": len(selected),
    }


def save_tracking_review(path: str | Path, review: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return target
