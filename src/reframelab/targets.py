from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_TARGET_KINDS = {"point", "box"}
_TARGET_ROLES = {"follow", "keep_visible"}


@dataclass(frozen=True)
class TargetAnchor:
    """A user annotation on one video frame.

    Coordinates and dimensions are normalised to the source frame. Point anchors may
    leave width and height at zero; the tracker then chooses a small patch around the
    point. Additional anchors act as appearance examples and correction keyframes.
    """

    time: float
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0

    def validate(self) -> None:
        if not np.isfinite(self.time) or self.time < 0.0:
            raise ValueError("Target anchor time must be a non-negative number")
        for name, value in (("x", self.x), ("y", self.y)):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Target anchor {name} must be between 0 and 1")
        for name, value in (("width", self.width), ("height", self.height)):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Target anchor {name} must be between 0 and 1")
        if self.width > 0.0 and self.x + self.width > 1.000001:
            raise ValueError("Target anchor extends beyond the right edge")
        if self.height > 0.0 and self.y + self.height > 1.000001:
            raise ValueError("Target anchor extends beyond the bottom edge")

    @property
    def center_x(self) -> float:
        return self.x if self.width <= 0.0 else self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y if self.height <= 0.0 else self.y + self.height / 2.0


@dataclass(frozen=True)
class TargetSpec:
    id: str
    label: str
    kind: str
    role: str
    anchors: tuple[TargetAnchor, ...]

    def validate(self) -> None:
        if not self.id or len(self.id) > 80:
            raise ValueError("Each target needs a short id")
        if self.kind not in _TARGET_KINDS:
            raise ValueError(f"Unknown target kind {self.kind!r}")
        if self.role not in _TARGET_ROLES:
            raise ValueError(f"Unknown target role {self.role!r}")
        if not self.anchors:
            raise ValueError("Each target needs at least one anchor")
        if len(self.anchors) > 12:
            raise ValueError("A target may contain at most 12 correction anchors")
        for anchor in self.anchors:
            anchor.validate()
            if self.kind == "box" and (anchor.width <= 0.0 or anchor.height <= 0.0):
                raise ValueError("Box targets need non-zero width and height")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetObservation:
    frame_index: int
    time: float
    visible: bool
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float
    source: str = "none"
    event: str = ""
    needs_review: bool = False
    uncertainty: float = 0.0
    priority: float = 1.0
    mask_polygon: list[list[float]] = field(default_factory=list)
    mask_x_profile: list[float] = field(default_factory=list)
    mask_y_profile: list[float] = field(default_factory=list)
    mask_confidence: float = 0.0
    mask_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetTrack:
    target: TargetSpec
    observations: list[TargetObservation]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "observations": [observation.to_dict() for observation in self.observations],
            "metrics": dict(self.metrics),
        }

    def nearest(self, time_seconds: float, tolerance: float | None = None) -> TargetObservation | None:
        if not self.observations:
            return None
        times = np.fromiter((item.time for item in self.observations), dtype=np.float64)
        index = int(np.argmin(np.abs(times - time_seconds)))
        result = self.observations[index]
        if tolerance is not None and abs(result.time - time_seconds) > tolerance:
            return None
        return result


def parse_target_specs(value: str | bytes | list[dict[str, Any]] | None) -> list[TargetSpec]:
    if value is None or value == "":
        return []
    if isinstance(value, (str, bytes)):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Targets must be valid JSON") from exc
    else:
        payload = value
    if not isinstance(payload, list):
        raise ValueError("Targets must be a JSON list")
    if len(payload) > 8:
        raise ValueError("Select at most 8 targets")

    result: list[TargetSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each target must be a JSON object")
        raw_anchors = item.get("anchors")
        if raw_anchors is None:
            # Accept the compact shape produced by early clients.
            raw_anchors = [item.get("anchor", item)]
        if not isinstance(raw_anchors, list):
            raise ValueError("Target anchors must be a list")
        anchors: list[TargetAnchor] = []
        for raw_anchor in raw_anchors:
            if not isinstance(raw_anchor, dict):
                raise ValueError("Each target anchor must be an object")
            anchors.append(
                TargetAnchor(
                    time=float(raw_anchor.get("time", 0.0)),
                    x=float(raw_anchor.get("x", raw_anchor.get("x_norm", 0.0))),
                    y=float(raw_anchor.get("y", raw_anchor.get("y_norm", 0.0))),
                    width=float(raw_anchor.get("width", raw_anchor.get("width_norm", 0.0))),
                    height=float(raw_anchor.get("height", raw_anchor.get("height_norm", 0.0))),
                )
            )
        target_id = str(item.get("id") or f"target-{index}").strip()
        if target_id in seen_ids:
            raise ValueError(f"Duplicate target id {target_id!r}")
        seen_ids.add(target_id)
        spec = TargetSpec(
            id=target_id,
            label=str(item.get("label") or f"Target {index}").strip()[:80],
            kind=str(item.get("kind") or "point"),
            role=str(item.get("role") or "follow"),
            anchors=tuple(sorted(anchors, key=lambda anchor: anchor.time)),
        )
        spec.validate()
        result.append(spec)
    return result


def load_tracks(path: str | Path) -> list[TargetTrack]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_tracks = payload.get("tracks", []) if isinstance(payload, dict) else []
    result: list[TargetTrack] = []
    for raw in raw_tracks:
        target = parse_target_specs([raw["target"]])[0]
        observations = [TargetObservation(**item) for item in raw.get("observations", [])]
        result.append(
            TargetTrack(
                target=target,
                observations=observations,
                metrics=dict(raw.get("metrics", {})),
            )
        )
    return result


def save_tracks(path: str | Path, tracks: Iterable[TargetTrack]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"tracks": [track.to_dict() for track in tracks]}, indent=2),
        encoding="utf-8",
    )
    return target
