from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    signal_weights: dict[str, float] = field(default_factory=dict)
    temporal: bool = False
    stabilize: bool = False
    allow_fallback: bool = False
    movement_penalty: float = 1.0
    max_transition_crop_fraction: float = 0.75
    dead_zone_fraction: float = 0.35
    max_speed_crop_widths_per_second: float = 1.15
    max_acceleration_crop_widths_per_second2: float = 3.2
    max_jerk_crop_widths_per_second3: float = 10.0
    layout_transition_seconds: float = 0.45
    minimum_layout_seconds: float = 0.90
    fallback_coverage_threshold: float = 0.46
    fallback_static_coverage_threshold: float = 0.62
    fallback_motion_reliability_threshold: float = 0.12
    fallback_min_seconds: float = 1.5
    fallback_merge_gap_seconds: float = 1.5
    fallback_padding_seconds: float = 0.5
    target_weight: float = 0.0
    target_only: bool = False
    target_context_fraction: float = 0.58
    target_min_coverage: float = 0.82
    semantic_layouts: bool = False
    context_detail_fraction: float = 0.68
    text_integrity_threshold: float = 0.88
    text_split_min_regions: int = 4
    camera_solver: str = "minimum_jerk"
    minimum_jerk_position_weight: float = 1.0
    minimum_jerk_velocity_weight: float = 0.003
    minimum_jerk_acceleration_weight: float = 0.012
    minimum_jerk_jerk_weight: float = 0.03


@dataclass(frozen=True)
class SemanticRegion:
    kind: str
    x: float
    y: float
    width: float
    height: float
    score: float
    face_x: float = 0.0
    face_y: float = 0.0
    face_width: float = 0.0
    face_height: float = 0.0
    speaker_score: float = 0.0
    speaker_confidence: float = 0.0
    speaker_track_id: int = 0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass(frozen=True)
class TextRegion:
    x: float
    y: float
    width: float
    height: float
    score: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass
class FrameEvidence:
    frame_index: int
    time: float
    shot_id: int
    shot_score: float
    x_positions: np.ndarray
    crop_width: int
    signal_coverage: dict[str, np.ndarray]
    signal_centering: dict[str, np.ndarray]
    signal_boundary: dict[str, np.ndarray]
    signal_reliability: dict[str, float]
    evaluation_score: np.ndarray
    evaluation_coverage: np.ndarray
    crop_axis: str = "x"
    crop_height: int = 0
    semantic_regions: list[SemanticRegion] = field(default_factory=list)
    semantic_backend: str = "none"
    text_regions: list[TextRegion] = field(default_factory=list)
    text_backend: str = "classical"
    motion_backend: str = "sparse_lk"
    depth_backend: str = "disabled"
    appearance_descriptor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class AnalysisBundle:
    video: VideoInfo
    analysis_width: int
    analysis_height: int
    target_width: int
    target_height: int
    sample_fps: float
    frames: list[FrameEvidence]
    crop_axis: str = "x"
    performance: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanPoint:
    frame_index: int
    time: float
    shot_id: int
    center_x_norm: float
    mode: str
    content_coverage: float
    confidence: float
    secondary_center_x_norm: float | None = None
    layout_detail_fraction: float = 0.68
    center_y_norm: float = 0.5
    secondary_center_y_norm: float | None = None
    split_primary_fraction: float = 0.5


@dataclass
class CropPlan:
    profile: str
    description: str
    source: str
    source_width: int
    source_height: int
    source_fps: float
    target_width: int
    target_height: int
    sample_fps: float
    points: list[PlanPoint]
    metrics: dict[str, float] = field(default_factory=dict)
    implementation: dict[str, Any] = field(default_factory=dict)
    crop_axis: str = "x"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropPlan:
        payload = dict(data)
        point_payloads = payload.pop("points")
        points = []
        for point in point_payloads:
            item = dict(point)
            item.setdefault("center_y_norm", 0.5)
            item.setdefault("secondary_center_y_norm", None)
            item.setdefault("split_primary_fraction", 0.5)
            points.append(PlanPoint(**item))
        payload.setdefault("crop_axis", "x")
        return cls(points=points, **payload)

    def save(self, path: str | Path) -> None:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CropPlan:
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
