from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_CORRECTION_SCOPES = {"shot", "matching", "all"}
_CORRECTION_MODES = {"auto", "crop", "split", "context_detail", "fallback"}


@dataclass(frozen=True)
class FramingCorrection:
    """A user-authored framing decision at one source timestamp.

    Coordinates are normalized to the source frame. ``scope`` controls where the
    correction is copied. ``mode`` can leave the automatic layout intact or force one
    of the renderer's supported layout families.
    """

    time: float
    center_x: float
    center_y: float
    scope: str = "shot"
    mode: str = "auto"
    secondary_center_x: float | None = None
    secondary_center_y: float | None = None
    label: str = "Framing correction"

    def validate(self) -> None:
        if not np.isfinite(self.time) or self.time < 0.0:
            raise ValueError("Framing correction time must be a non-negative number")
        for name, value in (("center_x", self.center_x), ("center_y", self.center_y)):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Framing correction {name} must be between 0 and 1")
        for name, value in (
            ("secondary_center_x", self.secondary_center_x),
            ("secondary_center_y", self.secondary_center_y),
        ):
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"Framing correction {name} must be between 0 and 1")
        if self.scope not in _CORRECTION_SCOPES:
            raise ValueError(f"Unknown framing correction scope {self.scope!r}")
        if self.mode not in _CORRECTION_MODES:
            raise ValueError(f"Unknown framing correction mode {self.mode!r}")
        if len(self.label) > 100:
            raise ValueError("Framing correction label is too long")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_framing_corrections(
    value: str | bytes | list[dict[str, Any]] | None,
) -> list[FramingCorrection]:
    if value is None or value == "":
        return []
    if isinstance(value, (str, bytes)):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Framing corrections must be valid JSON") from exc
    else:
        payload = value
    if not isinstance(payload, list):
        raise ValueError("Framing corrections must be a JSON list")
    if len(payload) > 40:
        raise ValueError("Use at most 40 framing corrections")

    corrections: list[FramingCorrection] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each framing correction must be an object")
        correction = FramingCorrection(
            time=float(item.get("time", 0.0)),
            center_x=float(item.get("center_x", item.get("x", 0.5))),
            center_y=float(item.get("center_y", item.get("y", 0.5))),
            scope=str(item.get("scope", "shot")),
            mode=str(item.get("mode", "auto")),
            secondary_center_x=(
                None
                if item.get("secondary_center_x") is None
                else float(item["secondary_center_x"])
            ),
            secondary_center_y=(
                None
                if item.get("secondary_center_y") is None
                else float(item["secondary_center_y"])
            ),
            label=str(item.get("label") or f"Framing correction {index}").strip()[:100],
        )
        correction.validate()
        corrections.append(correction)
    return sorted(corrections, key=lambda item: item.time)


def save_framing_corrections(
    path: str | Path,
    corrections: Iterable[FramingCorrection],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"corrections": [item.to_dict() for item in corrections]}, indent=2),
        encoding="utf-8",
    )
    return target
