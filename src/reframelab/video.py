from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .models import VideoInfo


class VideoError(RuntimeError):
    """Raised when a video cannot be probed, decoded, or encoded."""




def prefetch_iterator(iterator, *, maxsize: int = 8):
    """Read an iterator on a background thread with bounded memory.

    FFmpeg frame reads release the GIL while waiting on the pipe. Prefetching lets
    that I/O overlap OpenCV work without changing frame order. Producer exceptions
    are raised in the consumer thread.
    """

    items: queue.Queue[object] = queue.Queue(maxsize=max(1, int(maxsize)))
    sentinel = object()

    def produce() -> None:
        try:
            for item in iterator:
                items.put((True, item))
        except BaseException as exc:  # propagate exact producer failure
            items.put((False, exc))
        finally:
            items.put((True, sentinel))

    thread = threading.Thread(target=produce, name="reframe-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            ok, payload = items.get()
            if ok and payload is sentinel:
                break
            if not ok:
                raise payload
            yield payload
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        thread.join(timeout=2.0)

def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid size {value!r}; expected WIDTHxHEIGHT, for example 720x1280") from exc
    if width < 2 or height < 2:
        raise ValueError("Both output dimensions must be at least 2 pixels")
    return width, height


def even(value: int) -> int:
    return max(2, value - (value % 2))


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number <= 0:
        return None
    return number


def _parse_rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", maxsplit=1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            if denominator == 0.0:
                return None
            rate = numerator / denominator
        else:
            rate = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(rate) or rate <= 0.01 or rate > 240.0:
        return None
    return rate


def _ffprobe_video_metadata(source: Path) -> dict[str, object] | None:
    """Return lightweight video-stream metadata without decoding frames."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,"
                "duration,start_time:format=duration"
            ),
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams", [])
    if not streams:
        return None
    return {
        "stream": streams[0],
        "format": payload.get("format", {}),
    }


def _rate_is_variable(metadata: dict[str, object] | None) -> bool:
    if not metadata:
        return False
    stream = metadata.get("stream", {})
    if not isinstance(stream, dict):
        return False
    average = _parse_rate(stream.get("avg_frame_rate"))
    nominal = _parse_rate(stream.get("r_frame_rate"))
    duration = _positive_float(stream.get("duration"))
    frame_count = _positive_int(stream.get("nb_frames"))

    if nominal is not None and duration is not None and frame_count is not None:
        # MP4 commonly reports the video stream duration as the span from the first
        # frame timestamp to the last one, excluding the final frame interval. A true
        # CFR stream can therefore have avg_frame_rate slightly above r_frame_rate.
        # Accept either convention before applying the looser metadata heuristics.
        expected_span = max(frame_count - 1, 0) / nominal
        expected_full = frame_count / nominal
        timing_tolerance = max(0.02, 1.5 / nominal)
        if min(abs(duration - expected_span), abs(duration - expected_full)) <= timing_tolerance:
            return False

    if average is not None and nominal is not None:
        relative_gap = abs(average - nominal) / max(average, nominal, 1e-6)
        if relative_gap > 0.015:
            return True

    if average is not None and duration is not None and frame_count is not None:
        effective_span = max(frame_count - 1, 1) / duration
        relative_gap = abs(effective_span - average) / max(effective_span, average, 1e-6)
        if relative_gap > 0.015:
            return True
    return False


def probe_presentation_timing(path: str | Path) -> tuple[float | None, bool]:
    """Return the full video/audio presentation duration and audio presence.

    Only video and audio streams are considered. Subtitle or metadata tracks must not
    make the renderer clone the last frame for an unrelated stream duration.
    """

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None, False
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration,start_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, False

    streams = [
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type") in {"video", "audio"}
    ]
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    candidates: list[float] = []
    missing_types: list[tuple[str, float]] = []
    seen_types: set[str] = set()
    for stream in streams:
        codec_type = str(stream.get("codec_type"))
        # The renderer maps only the first video and first audio stream.
        if codec_type in seen_types:
            continue
        seen_types.add(codec_type)
        try:
            start_time = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            start_time = 0.0
        if not math.isfinite(start_time):
            start_time = 0.0
        start_time = max(0.0, start_time)
        duration = _positive_float(stream.get("duration"))
        if duration is None:
            selector = "v:0" if codec_type == "video" else "a:0"
            missing_types.append((selector, start_time))
            continue
        candidates.append(start_time + duration)

    format_duration = _positive_float(payload.get("format", {}).get("duration"))
    unresolved_stream_duration = bool(missing_types)
    ffprobe_path = shutil.which("ffprobe")
    if missing_types and ffprobe_path is not None:
        unresolved_stream_duration = False
        for selector, start_time in missing_types:
            packet_timing = _packet_timing(ffprobe_path, Path(path), selector)
            if packet_timing is None:
                unresolved_stream_duration = True
                continue
            candidates.append(start_time + packet_timing[1])

    stream_duration = max(candidates) if candidates else None
    if format_duration is None:
        return stream_duration, has_audio
    if stream_duration is None or unresolved_stream_duration:
        # WebM and Ogg frequently omit per-stream durations. Use the container only
        # when packet timing could not recover every mapped video/audio stream.
        return max(stream_duration or 0.0, format_duration), has_audio

    # When every video/audio stream has a duration, ignore a much longer container
    # duration. It can be extended by subtitles or malformed metadata. A small gap is
    # normal because muxers round the final packet end differently.
    tolerance = max(2.0, stream_duration * 0.05)
    if format_duration <= stream_duration + tolerance:
        stream_duration = max(stream_duration, format_duration)
    return stream_duration, has_audio


def _packet_timing(
    ffprobe: str,
    source: Path,
    stream_selector: str = "v:0",
) -> tuple[int, float] | None:
    """Scan one stream's packet timing without assuming packets arrive in PTS order."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        stream_selector,
        "-show_entries",
        "packet=pts_time,dts_time,duration_time",
        "-of",
        "csv=p=0",
        str(source),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None

    minimum_timestamp: float | None = None
    maximum_timestamp: float | None = None
    maximum_end: float | None = None
    packet_count = 0
    recent_timestamps: deque[float] = deque(maxlen=512)

    def parse_number(value: str) -> float | None:
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    for line in process.stdout:
        fields = line.strip().split(",")
        pts = parse_number(fields[0]) if len(fields) > 0 else None
        dts = parse_number(fields[1]) if len(fields) > 1 else None
        packet_duration = parse_number(fields[2]) if len(fields) > 2 else None
        timestamp = pts if pts is not None else dts
        if timestamp is None:
            continue

        minimum_timestamp = (
            timestamp if minimum_timestamp is None else min(minimum_timestamp, timestamp)
        )
        maximum_timestamp = (
            timestamp if maximum_timestamp is None else max(maximum_timestamp, timestamp)
        )
        if packet_duration is not None and packet_duration > 0.0:
            end = timestamp + packet_duration
            maximum_end = end if maximum_end is None else max(maximum_end, end)
        recent_timestamps.append(timestamp)
        packet_count += 1

    return_code = process.wait()
    if (
        return_code != 0
        or packet_count <= 0
        or minimum_timestamp is None
        or maximum_timestamp is None
    ):
        return None

    if maximum_end is not None:
        return packet_count, max(0.0, maximum_end - minimum_timestamp)
    if packet_count == 1:
        return packet_count, 0.0

    ordered = np.sort(np.unique(np.asarray(recent_timestamps, dtype=np.float64)))
    intervals = np.diff(ordered)
    intervals = intervals[(intervals > 1e-6) & (intervals < 10.0)]
    terminal_interval = float(np.median(intervals)) if intervals.size else 0.0
    span = max(0.0, maximum_timestamp - minimum_timestamp)
    return packet_count, span + terminal_interval


def probe_video(path: str | Path) -> VideoInfo:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    cap = cv.VideoCapture(str(source))
    if cap.isOpened():
        width_value = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height_value = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps_value = cap.get(cv.CAP_PROP_FPS)
        frame_count_value = cap.get(cv.CAP_PROP_FRAME_COUNT)
        cap.release()
    else:
        # Upload validation must not reject a codec merely because this OpenCV build
        # cannot decode it. FFprobe can still provide metadata and the normalisation
        # pass can transcode it into the H.264 working format used by the analyser.
        cap.release()
        width_value = height_value = fps_value = frame_count_value = float("nan")

    cv_width = _positive_int(round(width_value)) if math.isfinite(width_value) else None
    cv_height = _positive_int(round(height_value)) if math.isfinite(height_value) else None
    cv_fps = _parse_rate(fps_value)
    cv_frame_count = (
        _positive_int(round(frame_count_value))
        if math.isfinite(frame_count_value) and frame_count_value < 1_000_000_000
        else None
    )

    metadata = _ffprobe_video_metadata(source)
    if metadata is None:
        if cv_width and cv_height and cv_fps and cv_frame_count:
            return VideoInfo(
                path=str(source.resolve()),
                width=cv_width,
                height=cv_height,
                fps=cv_fps,
                frame_count=cv_frame_count,
                duration=cv_frame_count / cv_fps,
            )
        raise VideoError(
            "The video metadata was incomplete and ffprobe is unavailable. Install FFmpeg."
        )

    stream = metadata["stream"]
    format_info = metadata["format"]
    assert isinstance(stream, dict)
    assert isinstance(format_info, dict)

    width = _positive_int(stream.get("width")) or cv_width
    height = _positive_int(stream.get("height")) or cv_height
    frame_count = _positive_int(stream.get("nb_frames")) or cv_frame_count
    stream_duration = _positive_float(stream.get("duration"))
    format_duration = _positive_float(format_info.get("duration"))
    average_fps = _parse_rate(stream.get("avg_frame_rate")) or cv_fps
    nominal_fps = _parse_rate(stream.get("r_frame_rate"))

    # A CFR stream's declared duration often ends at the last frame timestamp and
    # omits that frame's display interval. Recover the full video duration from the
    # nominal rate when the metadata is consistent with either convention.
    if frame_count and nominal_fps and stream_duration is not None:
        expected_span = max(frame_count - 1, 0) / nominal_fps
        expected_full = frame_count / nominal_fps
        timing_tolerance = max(0.02, 1.5 / nominal_fps)
        closest_duration = min(
            abs(stream_duration - expected_span),
            abs(stream_duration - expected_full),
        )
        if closest_duration <= timing_tolerance:
            stream_duration = expected_full

    packet_timing: tuple[int, float] | None = None
    # Container duration can include a longer audio tail. When the video stream does
    # not publish its own duration, packet timestamps give the actual video timeline.
    if stream_duration is None or frame_count is None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is not None:
            packet_timing = _packet_timing(ffprobe, source)
    if packet_timing is not None:
        packet_count, packet_duration = packet_timing
        frame_count = frame_count or packet_count
        stream_duration = stream_duration or _positive_float(packet_duration)

    duration = stream_duration
    if duration is not None and format_duration is not None:
        reference_fps = nominal_fps or average_fps or cv_fps
        frame_slack = 0.05 if reference_fps is None else max(0.05, 1.5 / reference_fps)
        # MP4 stream duration often excludes the final frame interval. Prefer the
        # container duration only when the difference is small enough to be that tail,
        # not when a substantially longer audio stream extends the presentation.
        if 0.0 <= format_duration - duration <= frame_slack:
            duration = format_duration
    if duration is None:
        duration = format_duration
    if duration is None and frame_count and average_fps:
        duration = frame_count / average_fps
    if frame_count is None and duration and average_fps:
        frame_count = max(1, int(round(duration * average_fps)))

    fps = average_fps or nominal_fps
    if frame_count and duration:
        effective_fps = frame_count / duration
        if 0.01 < effective_fps <= 240.0:
            fps = effective_fps

    if not width or not height:
        raise VideoError(f"Could not determine video dimensions for {source}")
    if fps is None or frame_count is None or duration is None:
        raise VideoError(
            f"Could not determine a reliable timeline for {source}. "
            "Try remuxing or transcoding the source with FFmpeg."
        )

    return VideoInfo(
        path=str(source.resolve()),
        width=width,
        height=height,
        fps=float(fps),
        frame_count=int(frame_count),
        duration=float(duration),
    )


def capture_timestamp_seconds(
    capture: cv.VideoCapture,
    frame_index: int,
    fallback_fps: float,
    previous_time: float | None = None,
) -> float:
    """Return the decoded frame timestamp, with a monotonic CFR fallback.

    CAP_PROP_POS_MSEC reflects container timestamps on FFmpeg-backed captures and is
    important for browser-recorded variable-frame-rate WebM files. Some backends report
    zero or repeated values, so the fallback remains deterministic and monotonic.
    """

    timestamp_ms = float(capture.get(cv.CAP_PROP_POS_MSEC))
    timestamp = timestamp_ms / 1000.0 if math.isfinite(timestamp_ms) else -1.0
    frame_interval = 1.0 / max(float(fallback_fps), 1e-6)
    fallback = frame_index / max(float(fallback_fps), 1e-6)

    if timestamp >= 0.0:
        # Preserve exact frame-index timing for ordinary CFR files. This avoids
        # backend-specific floating-point noise while still using real timestamps
        # when a VFR recording materially departs from its average cadence.
        if abs(timestamp - fallback) <= frame_interval * 0.25:
            timestamp = fallback
        if previous_time is None:
            return timestamp
        if timestamp > previous_time + 1e-6:
            return timestamp

    if previous_time is None:
        return fallback
    return max(fallback, previous_time + frame_interval)



@dataclass(frozen=True)
class WorkingVideo:
    path: Path
    normalized: bool
    reason: str | None = None


def _opencv_timeline_is_reliable(source: Path) -> bool:
    capture = cv.VideoCapture(str(source))
    if not capture.isOpened():
        return False
    try:
        fps_value = float(capture.get(cv.CAP_PROP_FPS))
        count_value = float(capture.get(cv.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    fps = _parse_rate(fps_value)
    count_ok = (
        math.isfinite(count_value)
        and 0.0 < count_value < 1_000_000_000
    )
    return fps is not None and count_ok




def _opencv_decode_is_reliable(source: Path) -> bool:
    """Quickly verify that OpenCV can decode representative video frames.

    Reliable WebM files should not be transcoded merely because of their container.
    A first-frame and mid-stream probe catches unsupported or badly indexed sources
    without paying the cost of a full-resolution normalization pass.
    """

    capture = cv.VideoCapture(str(source))
    if not capture.isOpened():
        return False
    try:
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            return False
        frame_count = float(capture.get(cv.CAP_PROP_FRAME_COUNT))
        if math.isfinite(frame_count) and frame_count >= 8:
            capture.set(cv.CAP_PROP_POS_FRAMES, max(1.0, frame_count * 0.5))
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                return False
        return True
    finally:
        capture.release()

def _probe_codec_name(source: Path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    codec = result.stdout.strip().lower()
    return codec or None


def normalization_reason(path: str | Path) -> str | None:
    """Explain why a source should be converted to a predictable working MP4."""

    source = Path(path)
    metadata = _ffprobe_video_metadata(source)
    if _rate_is_variable(metadata):
        return "variable frame rate"
    if not _opencv_timeline_is_reliable(source):
        return "missing or irregular timeline metadata"
    codec = _probe_codec_name(source)
    if codec == "av1":
        return "AV1 decoding compatibility"
    if not _opencv_decode_is_reliable(source):
        label = codec.upper() if codec else source.suffix.lstrip(".").upper() or "video"
        return f"{label} decoding compatibility"
    return None



def _quick_nominal_fps(source: Path) -> float:
    metadata = _ffprobe_video_metadata(source)
    if metadata is not None:
        stream = metadata.get("stream", {})
        if isinstance(stream, dict):
            average = _parse_rate(stream.get("avg_frame_rate"))
            nominal = _parse_rate(stream.get("r_frame_rate"))
            if _rate_is_variable(metadata):
                # Keep the fastest published cadence when it is reasonable. This
                # preserves short-lived frames while FFmpeg converts the source to CFR.
                if nominal is not None and nominal <= 120.0:
                    return nominal
            if average is not None:
                return average
            if nominal is not None:
                return nominal

    capture = cv.VideoCapture(str(source))
    if capture.isOpened():
        try:
            detected = _parse_rate(capture.get(cv.CAP_PROP_FPS))
        finally:
            capture.release()
        if detected is not None:
            return detected

    # Browser recordings with missing rate metadata are overwhelmingly near 30 fps;
    # normalization still preserves their presentation duration through generated PTS.
    return 30.0



def _normalization_rate(fps: float) -> str:
    common_rates = [
        (23.976, "24000/1001"),
        (24.0, "24"),
        (25.0, "25"),
        (29.97, "30000/1001"),
        (30.0, "30"),
        (50.0, "50"),
        (59.94, "60000/1001"),
        (60.0, "60"),
    ]
    nearest_value, nearest_text = min(common_rates, key=lambda item: abs(item[0] - fps))
    if abs(nearest_value - fps) <= 0.08:
        return nearest_text
    return f"{fps:.6f}"


def normalize_video(
    source_path: str | Path,
    output_path: str | Path,
    *,
    fps: float | None = None,
) -> Path:
    """Create a CFR H.264/AAC working copy without shortening the presentation."""

    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rate = _normalization_rate(fps or _quick_nominal_fps(source))
    presentation_duration, has_audio = probe_presentation_timing(source)
    if presentation_duration is None:
        # Live/browser WebM files may omit both stream and container duration. Packet
        # timestamps still define a finite timeline and probe_video recovers it without
        # decoding the full source. This also prevents an unbounded ``apad`` filter.
        try:
            presentation_duration = probe_video(source).duration
        except (FileNotFoundError, VideoError):
            presentation_duration = None
    ffmpeg = require_ffmpeg()
    video_filter = f"fps={rate},setpts=PTS-STARTPTS"
    if presentation_duration is not None:
        video_filter += (
            f",tpad=stop_mode=clone:stop_duration={presentation_duration:.6f}"
        )

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0"])
    command.extend(
        [
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if has_audio:
        audio_filter = "asetpts=PTS-STARTPTS"
        if presentation_duration is not None:
            audio_filter += f",apad=whole_dur={presentation_duration:.6f}"
        command.extend(["-af", audio_filter, "-c:a", "aac", "-b:a", "160k"])
    if presentation_duration is not None:
        command.extend(["-t", f"{presentation_duration:.6f}"])
    command.extend(["-movflags", "+faststart", str(output)])
    run_command(command)
    return output


@contextmanager
def prepare_working_video(
    path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> Iterator[WorkingVideo]:
    """Yield a source OpenCV can decode predictably, cleaning temporary media afterward."""

    source = Path(path).resolve()
    reason = normalization_reason(source)
    if reason is None:
        yield WorkingVideo(path=source, normalized=False, reason=None)
        return

    if progress:
        progress(f"Normalizing source ({reason})")
    with tempfile.TemporaryDirectory(prefix="reframe-source-") as temporary_directory:
        working_path = Path(temporary_directory) / "working.mp4"
        normalize_video(source, working_path)
        if progress:
            progress("Source normalization complete")
        yield WorkingVideo(path=working_path, normalized=True, reason=reason)


def _read_exact(stream, size: int) -> bytes:
    """Read exactly ``size`` bytes unless the stream reaches EOF."""

    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_analysis_frames(
    path: str | Path,
    *,
    width: int,
    height: int,
    sample_fps: float,
    start_time: float = 0.0,
    end_time: float | None = None,
    use_ffmpeg_proxy: bool = True,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield low-resolution sampled BGR frames for visual analysis.

    The optimized path asks FFmpeg to sample and resize before bytes enter Python.
    The legacy path decodes the source with OpenCV, skips unwanted frames, and then
    resizes them. Both paths expose source-relative timestamps.
    """

    source = Path(path)
    if width <= 0 or height <= 0 or sample_fps <= 0.0:
        raise ValueError("Analysis dimensions and sample_fps must be positive")
    start_time = max(0.0, float(start_time))
    if end_time is not None and end_time <= start_time:
        return

    if use_ffmpeg_proxy:
        ffmpeg = require_ffmpeg()
        command = [ffmpeg, "-v", "error"]
        if start_time > 0.0:
            command.extend(["-ss", f"{start_time:.6f}"])
        command.extend(["-i", str(source)])
        if end_time is not None:
            command.extend(["-t", f"{end_time - start_time:.6f}"])
        command.extend(
            [
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"fps={sample_fps:.9f},scale={width}:{height}:flags=area",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        # A pipe for stderr can fill while a background frame-prefetch thread is
        # blocked on its bounded queue. Use a temporary file so FFmpeg diagnostics
        # never stall the raw-video stream.
        stderr_file = tempfile.TemporaryFile()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        assert process.stdout is not None
        frame_bytes = width * height * 3
        local_index = 0
        try:
            while True:
                payload = _read_exact(process.stdout, frame_bytes)
                if not payload:
                    break
                if len(payload) != frame_bytes:
                    raise VideoError(
                        f"FFmpeg analysis stream ended with a partial frame "
                        f"({len(payload)}/{frame_bytes} bytes)"
                    )
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
                time_seconds = start_time + local_index / sample_fps
                yield local_index, time_seconds, frame
                local_index += 1
        finally:
            if process.stdout is not None:
                process.stdout.close()
            return_code = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")
            stderr_file.close()
        if return_code != 0:
            raise VideoError(
                f"FFmpeg analysis proxy failed with exit code {return_code}: "
                f"{stderr[-3000:]}"
            )
        return

    info = probe_video(source)
    capture = cv.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoError(f"Could not decode {source}")
    try:
        if start_time > 0.0:
            capture.set(cv.CAP_PROP_POS_MSEC, start_time * 1000.0)
        source_index = max(0, int(round(start_time * info.fps)))
        previous_time: float | None = None
        next_sample = start_time
        sample_interval = 1.0 / sample_fps
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_seconds = capture_timestamp_seconds(
                capture, source_index, info.fps, previous_time
            )
            previous_time = time_seconds
            if end_time is not None and time_seconds > end_time + 0.5 / info.fps:
                break
            if time_seconds + 0.5 / info.fps < next_sample:
                source_index += 1
                continue
            resized = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
            yield source_index, time_seconds, resized
            next_sample = max(next_sample + sample_interval, time_seconds + sample_interval)
            source_index += 1
    finally:
        capture.release()

def require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise VideoError("FFmpeg is required for final H.264 encoding and audio preservation")
    return executable


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        rendered = " ".join(command)
        raise VideoError(
            f"Command failed with exit code {result.returncode}: {rendered}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
