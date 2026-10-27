from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CropAxis = Literal["x", "y", "none"]


LayoutAxis = Literal["x", "y"]


def _panel_target_size(
    target_width: int,
    target_height: int,
    split_axis: LayoutAxis,
    fraction: float,
) -> tuple[int, int]:
    fraction = float(np.clip(fraction, 0.05, 0.95))
    if split_axis == "x":
        return max(2, int(round(target_width * fraction))), target_height
    return target_width, max(2, int(round(target_height * fraction)))


def panel_crop_fraction(
    frame_width: int,
    frame_height: int,
    target_width: int,
    target_height: int,
    *,
    active_axis: CropAxis,
    split_axis: LayoutAxis,
    panel_fraction: float,
) -> float:
    """Return the source span used by one detail panel on the active axis."""

    panel_width, panel_height = _panel_target_size(
        target_width, target_height, split_axis, panel_fraction
    )
    panel = crop_geometry(frame_width, frame_height, panel_width, panel_height)
    if active_axis == "x":
        return float(panel.crop_width / max(frame_width, 1))
    if active_axis == "y":
        return float(panel.crop_height / max(frame_height, 1))
    return 1.0


def choose_layout_split_axis(
    frame_width: int,
    frame_height: int,
    target_width: int,
    target_height: int,
    *,
    panel_fraction: float = 0.5,
    maximum_preferred_crop_fraction: float = 0.94,
) -> LayoutAxis:
    """Choose a panel division that still produces useful detail crops.

    The visually natural orientation divides the output perpendicular to the source
    separation: x-cropped sources normally become top/bottom panels, and y-cropped
    sources normally become left/right panels. For less aggressive output ratios that
    orientation can make each panel show almost the complete source, duplicating the
    same image twice. In that case the division switches axes so each panel can still
    focus on its assigned region.
    """

    geometry = crop_geometry(frame_width, frame_height, target_width, target_height)
    if geometry.axis == "none":
        return "y" if target_height >= target_width else "x"

    preferred: LayoutAxis = "y" if geometry.axis == "x" else "x"
    preferred_fraction = panel_crop_fraction(
        frame_width,
        frame_height,
        target_width,
        target_height,
        active_axis=geometry.axis,
        split_axis=preferred,
        panel_fraction=panel_fraction,
    )
    if preferred_fraction <= maximum_preferred_crop_fraction:
        return preferred
    return "x" if preferred == "y" else "y"


@dataclass(frozen=True)
class CropGeometry:
    """Maximum source crop matching an output aspect ratio.

    Reframe Lab does not zoom beyond the largest crop that fills the requested
    output. Therefore only one source axis can move:

    - ``x`` when the source is wider than the requested output;
    - ``y`` when the source is taller than the requested output;
    - ``none`` when the aspect ratios already match.
    """

    frame_width: int
    frame_height: int
    target_width: int
    target_height: int
    crop_width: int
    crop_height: int
    axis: CropAxis

    @property
    def target_aspect(self) -> float:
        return self.target_width / max(self.target_height, 1)

    @property
    def source_aspect(self) -> float:
        return self.frame_width / max(self.frame_height, 1)

    @property
    def frame_span(self) -> int:
        if self.axis == "y":
            return self.frame_height
        return self.frame_width

    @property
    def crop_span(self) -> int:
        if self.axis == "y":
            return self.crop_height
        return self.crop_width

    @property
    def crop_fraction(self) -> float:
        return self.crop_span / max(self.frame_span, 1)

    def center_norms(self, active_center_norm: float) -> tuple[float, float]:
        center = float(np.clip(active_center_norm, 0.0, 1.0))
        if self.axis == "y":
            return 0.5, center
        if self.axis == "x":
            return center, 0.5
        return 0.5, 0.5

    def active_center(self, center_x_norm: float, center_y_norm: float) -> float:
        if self.axis == "y":
            return float(np.clip(center_y_norm, 0.0, 1.0))
        if self.axis == "x":
            return float(np.clip(center_x_norm, 0.0, 1.0))
        return 0.5


def _even_at_most(value: int, maximum: int) -> int:
    value = min(maximum, max(2, int(value)))
    if value % 2 and value > 2:
        value -= 1
    return max(2, value)


def crop_geometry(
    frame_width: int,
    frame_height: int,
    target_width: int,
    target_height: int,
    *,
    even_dimensions: bool = False,
) -> CropGeometry:
    """Return the largest source rectangle with the requested output aspect.

    This function is the single source of truth for analysis, planning, metrics,
    and rendering. It supports portrait, square, landscape, ultrawide, and custom
    output sizes, as well as source videos in any orientation.
    """

    if min(frame_width, frame_height, target_width, target_height) <= 0:
        raise ValueError("Frame and target dimensions must be positive")

    # Compare aspect ratios with integer cross-products so mathematically matching
    # ratios are not separated by floating-point rounding.
    source_scaled = frame_width * target_height
    target_scaled = target_width * frame_height

    if source_scaled > target_scaled:
        crop_height = frame_height
        crop_width = int(round(frame_height * target_width / target_height))
    elif source_scaled < target_scaled:
        crop_width = frame_width
        crop_height = int(round(frame_width * target_height / target_width))
    else:
        crop_width = frame_width
        crop_height = frame_height

    if even_dimensions:
        crop_width = _even_at_most(crop_width, frame_width)
        crop_height = _even_at_most(crop_height, frame_height)
    else:
        crop_width = min(frame_width, max(1, crop_width))
        crop_height = min(frame_height, max(1, crop_height))

    horizontal_room = frame_width - crop_width
    vertical_room = frame_height - crop_height
    if horizontal_room > 0 and horizontal_room >= vertical_room:
        axis: CropAxis = "x"
    elif vertical_room > 0:
        axis = "y"
    else:
        axis = "none"

    return CropGeometry(
        frame_width=frame_width,
        frame_height=frame_height,
        target_width=target_width,
        target_height=target_height,
        crop_width=crop_width,
        crop_height=crop_height,
        axis=axis,
    )

def scale_crop_geometry(
    source: CropGeometry,
    frame_width: int,
    frame_height: int,
    *,
    even_dimensions: bool = False,
) -> CropGeometry:
    """Scale a source-derived crop into a resized analysis frame.

    Analysis frames are commonly rounded to even dimensions for OpenCV and H.264.
    That rounding can alter the sampled aspect by a fraction of a percent. The active
    crop axis must still come from the real source dimensions; otherwise a matching
    16:9 source/output pair can be misclassified as a vertical crop at analysis time.
    """

    if min(frame_width, frame_height) <= 0:
        raise ValueError("Frame dimensions must be positive")

    if source.axis == "x":
        crop_width = int(round(frame_width * source.crop_width / source.frame_width))
        crop_height = frame_height
    elif source.axis == "y":
        crop_width = frame_width
        crop_height = int(round(frame_height * source.crop_height / source.frame_height))
    else:
        crop_width = frame_width
        crop_height = frame_height

    if even_dimensions:
        crop_width = _even_at_most(crop_width, frame_width)
        crop_height = _even_at_most(crop_height, frame_height)
    else:
        crop_width = min(frame_width, max(1, crop_width))
        crop_height = min(frame_height, max(1, crop_height))

    return CropGeometry(
        frame_width=frame_width,
        frame_height=frame_height,
        target_width=source.target_width,
        target_height=source.target_height,
        crop_width=crop_width,
        crop_height=crop_height,
        axis=source.axis,
    )

