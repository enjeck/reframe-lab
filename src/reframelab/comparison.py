from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from .video import even, probe_video, require_ffmpeg, run_command


def _filter_label(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("%", "\\%")
    )


def build_comparison_video(
    source_path: str | Path,
    rendered_paths: dict[str, Path],
    output_path: str | Path,
    *,
    tile_width: int | None = None,
    tile_height: int | None = None,
) -> Path:
    """Build a labelled comparison grid in FFmpeg.

    FFmpeg decodes and composes all streams in native code. This is substantially
    faster than the earlier Python frame loop, especially for long VP9 or H.264
    recordings, while preserving the source audio track.
    """

    source = Path(source_path)
    items = [("SOURCE", source)] + [
        (name, Path(path)) for name, path in rendered_paths.items()
    ]
    source_fps = probe_video(source).fps
    fps_fraction = Fraction(source_fps).limit_denominator(1001)
    fps_filter = f"{fps_fraction.numerator}/{fps_fraction.denominator}"

    columns = 3 if len(items) > 4 else 2
    rows = int(math.ceil(len(items) / columns))
    label_height = 34
    if tile_width is None or tile_height is None:
        reference = probe_video(next(iter(rendered_paths.values()))) if rendered_paths else probe_video(source)
        aspect = reference.width / max(reference.height, 1)
        if aspect >= 1.0:
            tile_width = 480
            tile_height = max(180, int(round(tile_width / aspect)))
        else:
            tile_height = 480
            tile_width = max(180, int(round(tile_height * aspect)))
    tile_width = even(tile_width)
    tile_height = even(tile_height)
    labelled_height = even(tile_height + label_height)
    canvas_width = even(columns * tile_width)
    canvas_height = even(rows * labelled_height)

    filters: list[str] = []
    labels: list[str] = []
    for index, (name, _) in enumerate(items):
        output_label = f"v{index}"
        escaped_name = _filter_label(name)
        scale = (
            f"scale={tile_width}:{tile_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        filters.append(
            f"[{index}:v]fps=fps={fps_filter}:round=near,setpts=PTS-STARTPTS,{scale},"
            f"pad={tile_width}:{labelled_height}:0:{label_height}:black,"
            "drawbox=x=0:y=0:w=iw:h="
            f"{label_height}:color=black:t=fill,"
            f"drawtext=text='{escaped_name}':x=10:y=8:fontsize=16:fontcolor=white"
            f"[{output_label}]"
        )
        labels.append(f"[{output_label}]")

    layout = "|".join(
        f"{(index % columns) * tile_width}_{(index // columns) * labelled_height}"
        for index in range(len(items))
    )
    filters.append(
        "".join(labels)
        + f"xstack=inputs={len(items)}:layout={layout}:fill=black,"
        + f"pad={canvas_width}:{canvas_height}:0:0:black[comparison]"
    )

    ffmpeg = require_ffmpeg()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for _, path in items:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[comparison]",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            fps_filter,
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run_command(command)
    return output
