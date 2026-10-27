from __future__ import annotations

import bisect
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import cv2 as cv
import numpy as np

from .geometry import choose_layout_split_axis, crop_geometry
from .models import CropPlan, PlanPoint
from .video import (
    capture_timestamp_seconds,
    even,
    probe_presentation_timing,
    probe_video,
    require_ffmpeg,
    run_command,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class RenderState:
    point: PlanPoint
    blend_from: PlanPoint | None = None
    blend_alpha: float = 1.0

    @property
    def center_x_norm(self) -> float:
        return self.point.center_x_norm

    @property
    def center_y_norm(self) -> float:
        return self.point.center_y_norm

    @property
    def mode(self) -> str:
        return self.point.mode


def _layout_family(mode: str) -> str:
    # hold and pan are camera labels, not different rendered layouts. Treating them
    # as different previously disabled interpolation and caused visible snaps.
    if mode in {"hold", "pan", "crop"}:
        return "crop"
    return mode


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class PlanInterpolator:
    def __init__(self, plan: CropPlan) -> None:
        if not plan.points:
            raise ValueError("Crop plan contains no points")
        self.points = plan.points
        self.times = [point.time for point in plan.points]
        self.transition_seconds = float(
            np.clip(plan.implementation.get("layout_transition_seconds", 0.50), 0.0, 1.5)
        )
        self.transitions: list[tuple[float, float, PlanPoint, PlanPoint]] = []
        for left, right in zip(self.points, self.points[1:], strict=False):
            if left.shot_id != right.shot_id:
                continue
            if _layout_family(left.mode) == _layout_family(right.mode):
                continue
            boundary = (left.time + right.time) / 2.0
            duration = min(
                self.transition_seconds,
                max(0.08, (right.time - left.time) * 2.5),
            )
            self.transitions.append(
                (boundary - duration / 2.0, boundary + duration / 2.0, left, right)
            )
        self.transition_starts = [item[0] for item in self.transitions]

    def _active_transition(
        self, time_seconds: float
    ) -> tuple[float, float, PlanPoint, PlanPoint] | None:
        if not self.transitions:
            return None
        index = bisect.bisect_right(self.transition_starts, time_seconds) - 1
        for candidate_index in (index, index + 1):
            if 0 <= candidate_index < len(self.transitions):
                transition = self.transitions[candidate_index]
                if transition[0] <= time_seconds <= transition[1]:
                    return transition
        return None

    def _tangent(self, index: int, attribute: str, *, forward: bool) -> float:
        point = self.points[index]
        family = _layout_family(point.mode)
        if forward:
            other_index = min(len(self.points) - 1, index + 1)
        else:
            other_index = max(0, index - 1)
        other = self.points[other_index]
        if (
            other_index == index
            or other.shot_id != point.shot_id
            or _layout_family(other.mode) != family
        ):
            return 0.0
        first, second = (point, other) if forward else (other, point)
        first_value = getattr(first, attribute)
        second_value = getattr(second, attribute)
        if first_value is None or second_value is None:
            return 0.0
        dt = max(second.time - first.time, 1e-6)
        return float(second_value - first_value) / dt

    def _hermite(
        self,
        left_index: int,
        right_index: int,
        attribute: str,
        alpha: float,
    ) -> float | None:
        left = self.points[left_index]
        right = self.points[right_index]
        y0 = getattr(left, attribute)
        y1 = getattr(right, attribute)
        if y0 is None or y1 is None:
            return None
        duration = max(right.time - left.time, 1e-6)
        secant = float(y1 - y0) / duration
        m0 = self._tangent(left_index, attribute, forward=False)
        m1 = self._tangent(right_index, attribute, forward=True)
        if abs(secant) <= 1e-9:
            m0 = m1 = 0.0
        else:
            limit = abs(secant) * 2.5
            if m0 * secant <= 0.0:
                m0 = 0.0
            if m1 * secant <= 0.0:
                m1 = 0.0
            m0 = float(np.clip(m0, -limit, limit))
            m1 = float(np.clip(m1, -limit, limit))
        a2 = alpha * alpha
        a3 = a2 * alpha
        value = (
            (2.0 * a3 - 3.0 * a2 + 1.0) * float(y0)
            + (a3 - 2.0 * a2 + alpha) * duration * m0
            + (-2.0 * a3 + 3.0 * a2) * float(y1)
            + (a3 - a2) * duration * m1
        )
        return float(np.clip(value, min(float(y0), float(y1)), max(float(y0), float(y1))))

    def _interpolated_point(
        self, left_index: int, right_index: int, time_seconds: float
    ) -> PlanPoint:
        left = self.points[left_index]
        right = self.points[right_index]
        duration = max(right.time - left.time, 1e-6)
        alpha = float(np.clip((time_seconds - left.time) / duration, 0.0, 1.0))
        center_x = self._hermite(left_index, right_index, "center_x_norm", alpha)
        center_y = self._hermite(left_index, right_index, "center_y_norm", alpha)
        secondary_x = self._hermite(
            left_index, right_index, "secondary_center_x_norm", alpha
        )
        secondary_y = self._hermite(
            left_index, right_index, "secondary_center_y_norm", alpha
        )
        detail = self._hermite(left_index, right_index, "layout_detail_fraction", alpha)
        split_fraction = self._hermite(left_index, right_index, "split_primary_fraction", alpha)
        return PlanPoint(
            frame_index=left.frame_index,
            time=time_seconds,
            shot_id=left.shot_id,
            center_x_norm=float(center_x if center_x is not None else left.center_x_norm),
            mode=left.mode if _layout_family(left.mode) == _layout_family(right.mode) else right.mode,
            content_coverage=(
                left.content_coverage * (1.0 - alpha) + right.content_coverage * alpha
            ),
            confidence=left.confidence * (1.0 - alpha) + right.confidence * alpha,
            secondary_center_x_norm=secondary_x,
            layout_detail_fraction=float(
                detail if detail is not None else left.layout_detail_fraction
            ),
            center_y_norm=float(center_y if center_y is not None else left.center_y_norm),
            secondary_center_y_norm=secondary_y,
            split_primary_fraction=float(
                split_fraction if split_fraction is not None else left.split_primary_fraction
            ),
        )

    def at(self, time_seconds: float) -> RenderState:
        transition = self._active_transition(time_seconds)
        if transition is not None:
            start, end, left, right = transition
            alpha = _smoothstep((time_seconds - start) / max(end - start, 1e-6))
            # Keep both layouts spatially stable during the dissolve. Interpolating
            # one shared centre here made a split or detail panel sweep sideways
            # while it faded, which still looked like a jump. The visual transition
            # is now carried entirely by the smooth alpha blend.
            from_point = replace(left, time=time_seconds)
            to_point = replace(right, time=time_seconds)
            return RenderState(point=to_point, blend_from=from_point, blend_alpha=alpha)

        right_index = bisect.bisect_right(self.times, time_seconds)
        if right_index <= 0:
            return RenderState(self.points[0])
        if right_index >= len(self.points):
            return RenderState(self.points[-1])

        left_index = right_index - 1
        left_point = self.points[left_index]
        right_point = self.points[right_index]
        if left_point.shot_id != right_point.shot_id:
            return RenderState(left_point if time_seconds < right_point.time else right_point)

        if _layout_family(left_point.mode) != _layout_family(right_point.mode):
            # Outside the explicit transition window, retain the nearest stable layout.
            midpoint = (left_point.time + right_point.time) / 2.0
            return RenderState(left_point if time_seconds < midpoint else right_point)

        return RenderState(self._interpolated_point(left_index, right_index, time_seconds))


def _fit_with_background(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    # Build the blurred fill from the source crop at low resolution. This produces
    # the same visual treatment as a large Gaussian kernel but is substantially
    # faster on long fallback segments.
    source_height, source_width = frame.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = source_width / max(source_height, 1)
    if source_aspect >= target_aspect:
        background_crop_width = max(2, int(round(source_height * target_aspect)))
        background_x = max(0, (source_width - background_crop_width) // 2)
        background_crop = frame[:, background_x : background_x + background_crop_width]
    else:
        background_crop_height = max(2, int(round(source_width / target_aspect)))
        background_y = max(0, (source_height - background_crop_height) // 2)
        background_crop = frame[background_y : background_y + background_crop_height, :]

    small_size = (max(24, target_width // 12), max(24, target_height // 12))
    background = cv.resize(background_crop, small_size, interpolation=cv.INTER_AREA)
    background = cv.resize(background, (target_width, target_height), interpolation=cv.INTER_LINEAR)
    background = cv.GaussianBlur(background, (0, 0), 1.2)
    background = cv.convertScaleAbs(background, alpha=0.58, beta=0)

    foreground_scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
    foreground_size = (
        max(2, even(int(round(frame.shape[1] * foreground_scale)))),
        max(2, even(int(round(frame.shape[0] * foreground_scale)))),
    )
    foreground = cv.resize(frame, foreground_size, interpolation=cv.INTER_AREA)
    fx = (target_width - foreground.shape[1]) // 2
    fy = (target_height - foreground.shape[0]) // 2
    background[fy : fy + foreground.shape[0], fx : fx + foreground.shape[1]] = foreground
    return background



def _crop_to_size(
    frame: np.ndarray,
    center_x_norm: float,
    center_y_norm: float,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Fill any output rectangle with a centre-controlled, aspect-correct crop."""

    source_height, source_width = frame.shape[:2]
    geometry = crop_geometry(source_width, source_height, target_width, target_height)
    if geometry.axis == "x":
        center_x = float(np.clip(center_x_norm, 0.0, 1.0)) * source_width
        x0 = int(round(center_x - geometry.crop_width / 2.0))
        x0 = int(np.clip(x0, 0, source_width - geometry.crop_width))
        crop = frame[:, x0 : x0 + geometry.crop_width]
    elif geometry.axis == "y":
        center_y = float(np.clip(center_y_norm, 0.0, 1.0)) * source_height
        y0 = int(round(center_y - geometry.crop_height / 2.0))
        y0 = int(np.clip(y0, 0, source_height - geometry.crop_height))
        crop = frame[y0 : y0 + geometry.crop_height, :]
    else:
        crop = frame
    interpolation = (
        cv.INTER_AREA
        if crop.shape[1] >= target_width and crop.shape[0] >= target_height
        else cv.INTER_CUBIC
    )
    return cv.resize(crop, (target_width, target_height), interpolation=interpolation)


def _fit_contain(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
    width = max(2, int(round(frame.shape[1] * scale)))
    height = max(2, int(round(frame.shape[0] * scale)))
    interpolation = cv.INTER_AREA if scale <= 1.0 else cv.INTER_CUBIC
    resized = cv.resize(frame, (width, height), interpolation=interpolation)
    x0 = (target_width - width) // 2
    y0 = (target_height - height) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = resized
    return canvas


def _layout_split_axis(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    *,
    panel_fraction: float = 0.5,
) -> str:
    source_height, source_width = frame.shape[:2]
    return choose_layout_split_axis(
        source_width,
        source_height,
        target_width,
        target_height,
        panel_fraction=panel_fraction,
    )


def _context_detail_layout(
    frame: np.ndarray,
    point: PlanPoint,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    detail_fraction = float(np.clip(point.layout_detail_fraction, 0.55, 0.80))
    split_axis = _layout_split_axis(
        frame, target_width, target_height, panel_fraction=detail_fraction
    )
    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if split_axis == "y":
        divider = max(2, target_height // 320)
        detail_height = max(2, int(round((target_height - divider) * detail_fraction)))
        context_height = target_height - divider - detail_height
        detail = _crop_to_size(
            frame,
            point.center_x_norm,
            point.center_y_norm,
            target_width,
            detail_height,
        )
        context = _fit_contain(frame, target_width, context_height)
        output[:detail_height] = detail
        output[detail_height + divider :] = context
    else:
        divider = max(2, target_width // 320)
        detail_width = max(2, int(round((target_width - divider) * detail_fraction)))
        context_width = target_width - divider - detail_width
        detail = _crop_to_size(
            frame,
            point.center_x_norm,
            point.center_y_norm,
            detail_width,
            target_height,
        )
        context = _fit_contain(frame, context_width, target_height)
        output[:, :detail_width] = detail
        output[:, detail_width + divider :] = context
    return output


def _split_layout(
    frame: np.ndarray,
    point: PlanPoint,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    secondary_x = (
        point.secondary_center_x_norm
        if point.secondary_center_x_norm is not None
        else point.center_x_norm
    )
    secondary_y = (
        point.secondary_center_y_norm
        if point.secondary_center_y_norm is not None
        else point.center_y_norm
    )
    primary_fraction = float(np.clip(point.split_primary_fraction, 0.36, 0.68))
    split_axis = _layout_split_axis(
        frame, target_width, target_height, panel_fraction=primary_fraction
    )
    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if split_axis == "y":
        divider = max(2, target_height // 320)
        available = target_height - divider
        first_height = max(2, int(round(available * primary_fraction)))
        second_height = max(2, available - first_height)
        first = _crop_to_size(
            frame,
            point.center_x_norm,
            point.center_y_norm,
            target_width,
            first_height,
        )
        second = _crop_to_size(
            frame,
            secondary_x,
            secondary_y,
            target_width,
            second_height,
        )
        output[:first_height] = first
        output[first_height + divider :] = second
    else:
        divider = max(2, target_width // 320)
        available = target_width - divider
        first_width = max(2, int(round(available * primary_fraction)))
        second_width = max(2, available - first_width)
        first = _crop_to_size(
            frame,
            point.center_x_norm,
            point.center_y_norm,
            first_width,
            target_height,
        )
        second = _crop_to_size(
            frame,
            secondary_x,
            secondary_y,
            second_width,
            target_height,
        )
        output[:, :first_width] = first
        output[:, first_width + divider :] = second
    return output


def _render_point(
    frame: np.ndarray,
    point: PlanPoint,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if point.mode == "fallback":
        return _fit_with_background(frame, target_width, target_height)
    if point.mode == "context_detail":
        return _context_detail_layout(frame, point, target_width, target_height)
    if point.mode == "split":
        return _split_layout(frame, point, target_width, target_height)
    return _crop_to_size(
        frame,
        point.center_x_norm,
        point.center_y_norm,
        target_width,
        target_height,
    )

def render_frame(
    frame: np.ndarray,
    state: RenderState | PlanPoint,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if isinstance(state, PlanPoint):
        return _render_point(frame, state, target_width, target_height)
    target = _render_point(frame, state.point, target_width, target_height)
    if state.blend_from is None or state.blend_alpha >= 0.999:
        return target
    source = _render_point(frame, state.blend_from, target_width, target_height)
    alpha = float(np.clip(state.blend_alpha, 0.0, 1.0))
    if alpha <= 0.001:
        return source
    return cv.addWeighted(source, 1.0 - alpha, target, alpha, 0.0)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv.VideoWriter:
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open temporary video writer for {path}")
    return writer


def _probe_mux_timing(path: str | Path) -> tuple[float | None, bool]:
    return probe_presentation_timing(path)


def _mux_and_encode(
    silent_video: Path,
    source_video: str | Path,
    output_path: Path,
    crf: int = 19,
    preset: str = "medium",
) -> None:
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    silent_duration, _ = _probe_mux_timing(silent_video)
    source_duration, has_audio = _probe_mux_timing(source_video)
    durations = [
        duration
        for duration in (silent_duration, source_duration)
        if duration is not None and duration > 0.0
    ]
    # probe_presentation_timing filters unrelated subtitle/metadata tails. Preserve the
    # complete video/audio presentation, even when the source intentionally ends with
    # several seconds of audio over a held final frame.
    target_duration = max(durations) if durations else None

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "1:a:0"])
        if target_duration is not None:
            command.extend(
                [
                    "-vf",
                    f"tpad=stop_mode=clone:stop_duration={target_duration:.6f}",
                    "-af",
                    f"apad=whole_dur={target_duration:.6f}",
                ]
            )
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "160k"])
    if target_duration is not None:
        command.extend(["-t", f"{target_duration:.6f}"])
    command.extend(
        [
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    run_command(command)


def _render_plans_legacy(
    input_path: str | Path,
    plans: dict[str, CropPlan],
    output_paths: dict[str, str | Path],
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Path]:
    if set(plans) != set(output_paths):
        raise ValueError("plans and output_paths must contain the same profile names")
    if not plans:
        return {}

    first_plan = next(iter(plans.values()))
    target_size = (even(first_plan.target_width), even(first_plan.target_height))
    for plan in plans.values():
        if (even(plan.target_width), even(plan.target_height)) != target_size:
            raise ValueError("All plans rendered in one pass must have the same output size")

    cap = cv.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not decode {input_path}")
    source_info = probe_video(input_path)
    fps = source_info.fps
    frame_count = source_info.frame_count

    temporary_root = Path(tempfile.mkdtemp(prefix="reframe-lab-"))
    writers: dict[str, cv.VideoWriter] = {}
    temporary_files: dict[str, Path] = {}
    interpolators = {name: PlanInterpolator(plan) for name, plan in plans.items()}

    try:
        for name in plans:
            temporary_path = temporary_root / f"{name}.silent.mp4"
            writers[name] = _open_writer(temporary_path, fps, target_size)
            temporary_files[name] = temporary_path

        frame_index = 0
        previous_frame_time: float | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_seconds = capture_timestamp_seconds(
                cap,
                frame_index,
                fps,
                previous_frame_time,
            )
            previous_frame_time = time_seconds
            for name, plan in plans.items():
                point = interpolators[name].at(time_seconds)
                output_frame = render_frame(
                    frame,
                    point,
                    even(plan.target_width),
                    even(plan.target_height),
                )
                writers[name].write(output_frame)
            frame_index += 1
            if progress and (frame_index == 1 or frame_index % max(1, int(fps * 4)) == 0):
                progress(f"Rendered {frame_index}/{frame_count or '?'} source frames")
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()

    rendered: dict[str, Path] = {}
    try:
        total_outputs = len(temporary_files)
        for output_index, (name, temporary_path) in enumerate(temporary_files.items(), start=1):
            if progress:
                progress(f"Encoding profile {output_index}/{total_outputs}: {name}")
            output_path = Path(output_paths[name])
            _mux_and_encode(temporary_path, input_path, output_path)
            rendered[name] = output_path
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    return rendered



def _open_ffmpeg_pipe(
    *,
    source_video: str | Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    target_duration: float | None,
    audio_duration: float | None,
    pad_duration: float,
    has_audio: bool,
    crf: int = 19,
    preset: str = "medium",
) -> subprocess.Popen[bytes]:
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.12g}",
        "-i",
        "pipe:0",
    ]
    if has_audio:
        command.extend(["-i", str(source_video)])
    command.extend(["-map", "0:v:0"])
    if has_audio:
        command.extend(["-map", "1:a:0?"])
    if pad_duration > 1e-4:
        command.extend(["-vf", f"tpad=stop_mode=clone:stop_duration={pad_duration:.6f}"])
    if has_audio and audio_duration is not None and audio_duration > 0.0:
        command.extend(["-af", f"apad=whole_dur={audio_duration:.6f}"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "160k"])
    if target_duration is not None and target_duration > 0.0:
        command.extend(["-t", f"{target_duration:.6f}"])
    command.extend(["-movflags", "+faststart", str(output_path)])
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


class _AsyncPipeWriter:
    def __init__(self, name: str, process: subprocess.Popen[bytes], maxsize: int = 4) -> None:
        self.name = name
        self.process = process
        self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=max(1, maxsize))
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name=f"reframe-encode-{name}", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            if self.process.stdin is None:
                raise RuntimeError(f"FFmpeg stdin unavailable for profile {self.name}")
            while True:
                payload = self.queue.get()
                if payload is None:
                    break
                self.process.stdin.write(payload)
        except BaseException as exc:
            self.error = exc
        finally:
            if self.process.stdin is not None and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass

    def submit(self, payload: bytes) -> None:
        while True:
            if self.error is not None:
                raise RuntimeError(f"Encoder writer failed for {self.name}: {self.error}")
            try:
                self.queue.put(payload, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        if self.thread.is_alive():
            self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"Encoder writer failed for {self.name}: {self.error}")


def _prefetched_source_frames(
    input_path: str | Path, fps: float, *, enabled: bool, maxsize: int = 6
):
    def decode():
        cap = cv.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not decode {input_path}")
        frame_index = 0
        previous_frame_time: float | None = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                time_seconds = capture_timestamp_seconds(
                    cap, frame_index, fps, previous_frame_time
                )
                previous_frame_time = time_seconds
                yield frame_index, time_seconds, frame
                frame_index += 1
        finally:
            cap.release()

    iterator = decode()
    if not enabled:
        yield from iterator
        return
    from .video import prefetch_iterator

    yield from prefetch_iterator(iterator, maxsize=maxsize)


def _render_plans_direct(
    input_path: str | Path,
    plans: dict[str, CropPlan],
    output_paths: dict[str, str | Path],
    *,
    pipeline_queues: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Path], dict[str, float]]:
    if set(plans) != set(output_paths):
        raise ValueError("plans and output_paths must contain the same profile names")
    if not plans:
        return {}, {"render_seconds": 0.0, "encode_seconds": 0.0}

    first_plan = next(iter(plans.values()))
    target_size = (even(first_plan.target_width), even(first_plan.target_height))
    for plan in plans.values():
        if (even(plan.target_width), even(plan.target_height)) != target_size:
            raise ValueError("All plans rendered in one pass must have the same output size")

    source_info = probe_video(input_path)
    fps = source_info.fps
    frame_count = source_info.frame_count
    presentation_duration, has_audio = _probe_mux_timing(input_path)
    target_duration = presentation_duration or source_info.duration
    interpolators = {name: PlanInterpolator(plan) for name, plan in plans.items()}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    outputs = {name: Path(path) for name, path in output_paths.items()}

    start = time.perf_counter()
    try:
        total_outputs = len(plans)
        for output_index, name in enumerate(plans, start=1):
            if progress:
                progress(f"Encoding profile {output_index}/{total_outputs}: {name}")
            processes[name] = _open_ffmpeg_pipe(
                source_video=input_path,
                output_path=outputs[name],
                width=target_size[0],
                height=target_size[1],
                fps=fps,
                target_duration=target_duration,
                audio_duration=target_duration,
                pad_duration=(
                    max(0.0, target_duration - source_info.duration)
                    + (1.0 / max(fps, 1e-6) if target_duration > source_info.duration else 0.0)
                ),
                has_audio=has_audio,
            )

        # A background writer only helps when several encoders compete for the
        # same composed frame. For a single output, FFmpeg already encodes in a
        # separate process while Python composes the next frame; adding another
        # Python queue creates an extra full-frame copy and can be slower.
        use_async_writers = bool(pipeline_queues and len(processes) > 1)
        writers = (
            {name: _AsyncPipeWriter(name, process) for name, process in processes.items()}
            if use_async_writers
            else {}
        )
        frame_index = 0
        for frame_index, time_seconds, frame in _prefetched_source_frames(
            input_path, fps, enabled=pipeline_queues
        ):
            for name, plan in plans.items():
                state = interpolators[name].at(time_seconds)
                output_frame = render_frame(
                    frame, state, even(plan.target_width), even(plan.target_height)
                )
                payload = output_frame.tobytes()
                if use_async_writers:
                    writers[name].submit(payload)
                else:
                    process = processes[name]
                    if process.stdin is None:
                        raise RuntimeError(f"FFmpeg stdin unavailable for profile {name}")
                    try:
                        process.stdin.write(payload)
                    except BrokenPipeError as exc:
                        stderr = (
                            process.stderr.read().decode("utf-8", errors="replace")
                            if process.stderr is not None else ""
                        )
                        raise RuntimeError(
                            f"FFmpeg stopped while encoding {name}: {stderr[-3000:]}"
                        ) from exc
            completed = frame_index + 1
            if progress and (completed == 1 or completed % max(1, int(fps * 4)) == 0):
                progress(f"Rendered {completed}/{frame_count or '?'} source frames")
    finally:
        if locals().get("use_async_writers", False):
            for writer in locals().get("writers", {}).values():
                writer.close()
        else:
            for process in processes.values():
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()

    render_done = time.perf_counter()
    failures: list[str] = []
    for name, process in processes.items():
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if process.stderr is not None:
            process.stderr.close()
        if return_code != 0:
            failures.append(f"{name}: exit {return_code}: {stderr[-2000:]}")
    if failures:
        raise RuntimeError("Direct FFmpeg encoding failed: " + " | ".join(failures))
    end = time.perf_counter()
    return outputs, {
        "render_seconds": render_done - start,
        "encode_flush_seconds": end - render_done,
        "total_seconds": end - start,
        "pipeline_queues": bool(pipeline_queues),
    }


def render_plans(
    input_path: str | Path,
    plans: dict[str, CropPlan],
    output_paths: dict[str, str | Path],
    *,
    direct_ffmpeg: bool = True,
    pipeline_queues: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Path]:
    """Render one or more plans.

    The default path pipes OpenCV-composed BGR frames directly into FFmpeg, avoiding
    the old temporary MP4 and second video encode. ``direct_ffmpeg=False`` retains the
    legacy path for benchmarks and regression testing.
    """

    if direct_ffmpeg:
        rendered, _ = _render_plans_direct(
            input_path,
            plans,
            output_paths,
            pipeline_queues=pipeline_queues,
            progress=progress,
        )
        return rendered
    return _render_plans_legacy(
        input_path,
        plans,
        output_paths,
        progress=progress,
    )

def render_plan(
    input_path: str | Path,
    plan: CropPlan,
    output_path: str | Path,
    *,
    direct_ffmpeg: bool = True,
    pipeline_queues: bool = True,
    progress: ProgressCallback | None = None,
) -> Path:
    rendered = render_plans(
        input_path,
        {plan.profile: plan},
        {plan.profile: output_path},
        direct_ffmpeg=direct_ffmpeg,
        pipeline_queues=pipeline_queues,
        progress=progress,
    )
    return rendered[plan.profile]
