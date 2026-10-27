from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from reframelab.models import CropPlan, PlanPoint
from reframelab.pipeline import run_demo
from reframelab.renderer import render_plan
from reframelab.video import (
    _packet_timing,
    normalization_reason,
    normalize_video,
    probe_presentation_timing,
    probe_video,
)
from reframelab.webapp import _INDEX_HTML, _safe_filename

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="FFmpeg and ffprobe are required",
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _probe(path: Path) -> dict[str, object]:
    assert FFPROBE is not None
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _center_plan(source: Path, *, fps: float, frames: int) -> CropPlan:
    return CropPlan(
        profile="center",
        description="regression test",
        source=str(source),
        source_width=160,
        source_height=90,
        source_fps=fps,
        target_width=90,
        target_height=160,
        sample_fps=2.0,
        points=[
            PlanPoint(0, 0.0, 0, 0.5, "crop", 1.0, 1.0),
            PlanPoint(frames - 1, (frames - 1) / fps, 0, 0.5, "crop", 1.0, 1.0),
        ],
    )


def test_long_upload_filename_keeps_video_suffix() -> None:
    value = _safe_filename("x" * 500 + ".MP4")
    assert len(value) <= 180
    assert value.endswith(".MP4")


def test_preview_selection_accounts_for_letterboxing() -> None:
    assert "function videoContentRect()" in _INDEX_HTML
    assert "if(!clampOutside&&(x<area.x" in _INDEX_HTML
    assert "const area=videoContentRect(),w=area.width,h=area.height" in _INDEX_HTML


@requires_ffmpeg
def test_probe_uses_ffprobe_when_opencv_cannot_open(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
    )

    class ClosedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    import reframelab.video as video_module

    monkeypatch.setattr(video_module.cv, "VideoCapture", lambda _: ClosedCapture())
    info = probe_video(source)
    assert (info.width, info.height) == (160, 90)
    assert info.frame_count == 10
    assert info.fps == pytest.approx(10.0, abs=0.01)
    assert info.duration == pytest.approx(1.0, abs=0.02)


@requires_ffmpeg
def test_normalization_does_not_truncate_video_when_audio_is_shorter(tmp_path) -> None:
    source = tmp_path / "short-audio.mp4"
    output = tmp_path / "normalised.mp4"
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )

    normalize_video(source, output)
    payload = _probe(output)
    streams = {stream["codec_type"]: stream for stream in payload["streams"]}
    assert float(payload["format"]["duration"]) == pytest.approx(3.0, abs=0.04)
    assert int(streams["video"]["nb_frames"]) == 90
    assert float(streams["audio"]["duration"]) == pytest.approx(3.0, abs=0.04)


@requires_ffmpeg
def test_render_preserves_long_audio_tail(tmp_path) -> None:
    source = tmp_path / "long-audio.mp4"
    output = tmp_path / "portrait.mp4"
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=5",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )

    render_plan(source, _center_plan(source, fps=30.0, frames=30), output)
    payload = _probe(output)
    streams = {stream["codec_type"]: stream for stream in payload["streams"]}
    assert float(payload["format"]["duration"]) == pytest.approx(5.0, abs=0.05)
    assert int(streams["video"]["nb_frames"]) == 150
    assert float(streams["audio"]["duration"]) == pytest.approx(5.0, abs=0.05)


@requires_ffmpeg
def test_vfr_source_is_detected_and_normalized_to_cfr(tmp_path) -> None:
    source = tmp_path / "vfr.mp4"
    output = tmp_path / "cfr.mp4"
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-vf",
            "select='if(lt(t,1),not(mod(n,3)),1)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )

    source_info = probe_video(source)
    assert source_info.frame_count == 40
    assert source_info.duration == pytest.approx(2.0, abs=0.03)
    assert normalization_reason(source) == "variable frame rate"

    normalize_video(source, output)
    output_info = probe_video(output)
    assert output_info.frame_count == 60
    assert output_info.fps == pytest.approx(30.0, abs=0.02)
    assert output_info.duration == pytest.approx(2.0, abs=0.03)
    assert normalization_reason(output) is None


@requires_ffmpeg
def test_durationless_webm_packet_timing_and_normalization_are_finite(tmp_path) -> None:
    source = tmp_path / "live.webm"
    output = tmp_path / "working.mp4"
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "-live",
            "1",
            str(source),
        ]
    )

    duration, has_audio = probe_presentation_timing(source)
    assert has_audio is True
    assert duration == pytest.approx(3.021, abs=0.04)
    assert normalization_reason(source) == "missing or irregular timeline metadata"

    normalize_video(source, output)
    output_duration, output_has_audio = probe_presentation_timing(output)
    assert output_has_audio is True
    assert output_duration == pytest.approx(3.021, abs=0.05)
    assert output.stat().st_size > 1000

    result = run_demo(
        source,
        tmp_path / "demo",
        ["center"],
        output_size="90x160",
        analysis_height=90,
        analysis_fps=2.0,
        candidate_step=4,
        create_comparison=False,
        advanced_vision=False,
        enable_depth=False,
        allow_model_download=False,
    )
    rendered = _probe(result.rendered_paths["center"])
    assert float(rendered["format"]["duration"]) == pytest.approx(3.1, abs=0.11)
    rendered_streams = {item["codec_type"]: item for item in rendered["streams"]}
    assert int(rendered_streams["video"]["nb_frames"]) >= 30
    assert float(rendered_streams["audio"]["duration"]) == pytest.approx(3.021, abs=0.05)


@requires_ffmpeg
def test_packet_scan_handles_b_frame_pts_reordering(tmp_path) -> None:
    source = tmp_path / "bframes.mp4"
    assert FFMPEG is not None and FFPROBE is not None
    _run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=3",
            "-c:v",
            "libx264",
            "-bf",
            "3",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
    )
    timing = _packet_timing(FFPROBE, source)
    assert timing is not None
    packet_count, duration = timing
    assert packet_count == 90
    assert duration == pytest.approx(3.0, abs=0.01)
