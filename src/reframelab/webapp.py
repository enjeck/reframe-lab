from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .corrections import FramingCorrection, parse_framing_corrections
from .pipeline import prepare_tracking_review, run_demo
from .profiles import (
    DEFAULT_DEMO_PROFILES,
    RECOMMENDED_PROFILES,
    TARGET_DEMO_PROFILES,
    TARGET_VARIANT_PROFILES,
    VARIANT_PROFILES,
    get_profile,
    variant_name,
)
from .targets import TargetSpec, load_tracks, parse_target_specs
from .video import parse_size, probe_video

VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ogv",
    ".ts",
    ".vob",
    ".webm",
    ".wmv",
}

RUN_PRESETS: dict[str, dict[str, Any]] = {
    # Default: render every user-facing variant so the person picks the edit.
    "all": {
        "automatic_profiles": list(VARIANT_PROFILES),
        "target_profiles": list(TARGET_VARIANT_PROFILES),
        "comparison": False,
    },
    # Retained as an API-compatible alias for v0.8.x clients. It is no longer
    # shown in the browser UI because the public comparison is intentionally small.
    "compare": {
        "automatic_profiles": list(DEFAULT_DEMO_PROFILES),
        "target_profiles": list(TARGET_DEMO_PROFILES),
        "comparison": True,
    },
    "fast_compare": {
        "automatic_profiles": list(DEFAULT_DEMO_PROFILES),
        "target_profiles": list(TARGET_DEMO_PROFILES),
        "comparison": True,
    },
    "best": {
        "automatic_profiles": ["hybrid_safe"],
        "target_profiles": ["target_guided_safe"],
        "comparison": False,
    },
}


def _profiles_for_run(preset: str, has_targets: bool) -> list[str]:
    settings = RUN_PRESETS[preset]
    key = "target_profiles" if has_targets else "automatic_profiles"
    return list(settings[key])


QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "analysis_height": 180,
        "analysis_fps": 3.0,
        "candidate_step": 8,
        "adaptive_dense_flow": False,
    },
    "balanced": {
        "analysis_height": 240,
        "analysis_fps": 5.0,
        "candidate_step": 8,
        "adaptive_dense_flow": True,
    },
    "careful": {
        "analysis_height": 360,
        "analysis_fps": 8.0,
        "candidate_step": 6,
        "adaptive_dense_flow": True,
    },
}

_OUTPUT_PIXEL_LIMIT = 3840 * 2160
_OUTPUT_DIMENSION_LIMIT = 4096
_OUTPUT_DIMENSION_MINIMUM = 64
_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_PROGRESS_TRACK = re.compile(r"Tracked\s+(\d+)/(\d+)\s+sampled frames")
_PROGRESS_ANALYSIS = re.compile(r"Analysed\s+(\d+)/(\d+)\s+sampled frames")
_PROGRESS_RENDER = re.compile(r"Rendered\s+(\d+)/(\d+|\?)\s+source frames")
_PROGRESS_ENCODE = re.compile(r"Encoding profile\s+(\d+)/(\d+):")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename: str | None) -> str:
    raw = Path(filename or "video.mp4").name.strip() or "video.mp4"
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(raw).suffix)
    stem = Path(raw).stem if suffix else raw
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).lstrip(".") or "video"
    suffix = suffix[:24]
    maximum_stem = max(1, 180 - len(suffix))
    return stem[:maximum_stem] + suffix


def _validate_output_size(value: str) -> str:
    width, height = parse_size(value)
    if width < _OUTPUT_DIMENSION_MINIMUM or height < _OUTPUT_DIMENSION_MINIMUM:
        raise ValueError(f"Each output dimension must be at least {_OUTPUT_DIMENSION_MINIMUM} pixels")
    if width > _OUTPUT_DIMENSION_LIMIT or height > _OUTPUT_DIMENSION_LIMIT:
        raise ValueError(f"Each output dimension must be at most {_OUTPUT_DIMENSION_LIMIT} pixels")
    if width * height > _OUTPUT_PIXEL_LIMIT:
        raise ValueError("The web output may contain at most 8.3 million pixels")
    aspect = width / height
    if not 0.125 <= aspect <= 8.0:
        raise ValueError("Choose an aspect ratio between 1:8 and 8:1")
    # H.264 yuv420 output requires even dimensions. Normalising here keeps custom
    # input easy while preserving the requested aspect to within one pixel.
    width -= width % 2
    height -= height % 2
    return f"{width}x{height}"


def _archive_results(output_dir: Path, archive_path: Path) -> None:
    archive_resolved = archive_path.resolve()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_resolved:
                archive.write(path, path.relative_to(output_dir))


@dataclass
class JobRecord:
    id: str
    status: str
    progress: float
    message: str
    original_name: str
    stored_name: str
    preset: str
    quality: str
    output_size: str
    profiles: list[str]
    speed_options: dict[str, Any]
    created_at: str
    updated_at: str
    targets: list[dict[str, Any]] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["progress"] = round(float(data["progress"]), 4)
        return data


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._load_existing()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def source_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "source"

    def output_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "output"

    def state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _load_existing(self) -> None:
        for state_path in self.root.glob("*/job.json"):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                data.setdefault(
                    "speed_options",
                    {
                        "analysis_proxy": True,
                        "coarse_to_fine": True,
                        "detector_cadence": True,
                        "direct_ffmpeg": True,
                        "static_scene_reuse": True,
                        "persistent_models": True,
                        "dnn_optimization": True,
                        "dnn_batch_size": 4,
                        "pipeline_queues": True,
                        "consistent_framing": True,
                        "active_speaker": True,
                    },
                )
                data.setdefault("corrections", [])
                record = JobRecord(**data)
            except (OSError, ValueError, TypeError):
                continue
            if record.status in {"uploading", "queued", "running"}:
                record.status = "failed"
                record.error = "The server stopped before this run completed."
                record.message = "Stopped"
                record.updated_at = _utc_now()
                self._persist(record)
            self._records[record.id] = record

    def _persist(self, record: JobRecord) -> None:
        path = self.state_path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.replace(path)

    def create(
        self,
        *,
        original_name: str,
        stored_name: str,
        preset: str,
        quality: str,
        output_size: str,
        profiles: list[str],
        speed_options: dict[str, Any],
        targets: list[dict[str, Any]],
        corrections: list[dict[str, Any]] | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        record = JobRecord(
            id=job_id,
            status="uploading",
            progress=0.0,
            message="Uploading",
            original_name=original_name,
            stored_name=stored_name,
            preset=preset,
            quality=quality,
            output_size=output_size,
            profiles=list(profiles),
            speed_options=dict(speed_options),
            created_at=now,
            targets=list(targets),
            corrections=list(corrections or []),
            updated_at=now,
        )
        with self._lock:
            self.source_dir(job_id).mkdir(parents=True, exist_ok=True)
            self.output_dir(job_id).mkdir(parents=True, exist_ok=True)
            self._records[job_id] = record
            self._persist(record)
        return record

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(job_id)
            for key, value in changes.items():
                if not hasattr(record, key):
                    raise AttributeError(key)
                setattr(record, key, value)
            record.updated_at = _utc_now()
            self._persist(record)
            return record

    def append_log(
        self,
        job_id: str,
        message: str,
        *,
        progress: float | None = None,
        status: str | None = None,
    ) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(job_id)
            record.logs.append(message)
            record.logs = record.logs[-100:]
            record.message = message
            if progress is not None:
                record.progress = max(record.progress, min(1.0, float(progress)))
            if status is not None:
                record.status = status
            record.updated_at = _utc_now()
            self._persist(record)
            return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def public(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            return record.public_dict() if record is not None else None

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._records.pop(job_id, None)
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def clean_old(self, max_age_days: int) -> None:
        if max_age_days <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        for record in list(self._records.values()):
            try:
                updated = datetime.fromisoformat(record.updated_at)
            except ValueError:
                continue
            if record.status in {"completed", "failed", "review"} and updated < cutoff:
                self.delete(record.id)


class JobRunner:
    def __init__(self, store: JobStore, max_workers: int = 1) -> None:
        self.store = store
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="reframe-web",
        )

    def submit_initial(
        self,
        job_id: str,
        input_path: Path,
        *,
        profiles: list[str],
        output_size: str,
        quality: str,
        create_comparison: bool,
        target_specs: list[TargetSpec],
        speed_options: dict[str, Any],
        corrections: list[FramingCorrection] | None = None,
    ) -> None:
        if target_specs:
            self.submit_review(
                job_id,
                input_path,
                quality=quality,
                target_specs=target_specs,
            )
        else:
            self.submit_render(
                job_id,
                input_path,
                profiles=profiles,
                output_size=output_size,
                quality=quality,
                create_comparison=create_comparison,
                target_specs=target_specs,
                precomputed_tracks=None,
                framing_corrections=corrections or [],
                precomputed_bundle=None,
                speed_options=speed_options,
            )

    def submit_review(
        self,
        job_id: str,
        input_path: Path,
        *,
        quality: str,
        target_specs: list[TargetSpec],
    ) -> None:
        self.store.update(
            job_id,
            status="queued",
            progress=0.02,
            message="Queued tracking review",
            results={},
            error=None,
        )
        self._executor.submit(
            self._run_review,
            job_id,
            input_path,
            quality,
            target_specs,
        )

    def submit_render(
        self,
        job_id: str,
        input_path: Path,
        *,
        profiles: list[str],
        output_size: str,
        quality: str,
        create_comparison: bool,
        target_specs: list[TargetSpec],
        precomputed_tracks: Path | None,
        speed_options: dict[str, Any],
        framing_corrections: list[FramingCorrection] | None = None,
        precomputed_bundle: Path | None = None,
    ) -> None:
        self.store.update(
            job_id,
            status="queued",
            progress=0.02,
            message="Queued render",
            profiles=list(profiles),
            targets=[target.to_dict() for target in target_specs],
            corrections=[item.to_dict() for item in (framing_corrections or [])],
            results={},
            error=None,
        )
        self._executor.submit(
            self._run_render,
            job_id,
            input_path,
            profiles,
            output_size,
            quality,
            create_comparison,
            target_specs,
            precomputed_tracks,
            framing_corrections or [],
            precomputed_bundle,
            speed_options,
        )

    def shutdown(self) -> None:
        # ThreadPoolExecutor workers are non-daemon threads. Waiting here prevents
        # finished web tests and graceful server shutdowns from hanging after the
        # application lifespan has ended. Pending jobs are cancelled; running jobs
        # are allowed to finish so files are not left half-written.
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run_review(
        self,
        job_id: str,
        input_path: Path,
        quality: str,
        target_specs: list[TargetSpec],
    ) -> None:
        settings = QUALITY_PRESETS[quality]
        output_dir = self.store.output_dir(job_id)
        current_progress = 0.03

        def report(message: str) -> None:
            nonlocal current_progress
            track_match = _PROGRESS_TRACK.search(message)
            if track_match:
                done, total = (int(value) for value in track_match.groups())
                current_progress = max(current_progress, 0.06 + 0.76 * done / max(total, 1))
            elif message.startswith("Building the pre-render"):
                current_progress = max(current_progress, 0.86)
            elif message.startswith("Encoding"):
                current_progress = max(current_progress, 0.92)
            self.store.append_log(
                job_id,
                message,
                progress=current_progress,
                status="running",
            )

        try:
            self.store.append_log(
                job_id,
                "Preparing tracking review",
                progress=0.03,
                status="running",
            )
            review_result = prepare_tracking_review(
                input_path,
                output_dir,
                target_specs=target_specs,
                analysis_height=max(360, int(settings["analysis_height"])),
                tracking_fps=max(8.0, float(settings["analysis_fps"])),
                enable_learned_tracking=False,
                advanced_vision=False,
                progress=report,
            )
            base_url = f"/runs/{job_id}"
            results = {
                "tracking_preview_url": (
                    f"{base_url}/{review_result.tracking_preview_path.name}"
                ),
                "tracks_url": f"{base_url}/{review_result.tracks_path.name}",
                "tracking_review_url": f"{base_url}/{review_result.review_path.name}",
                "review": review_result.review,
                "primary_video_url": (
                    f"{base_url}/{review_result.tracking_preview_path.name}"
                ),
                "source_video_url": f"/source/{job_id}",
                "outputs": [],
            }
            self.store.update(
                job_id,
                status="review",
                progress=1.0,
                message="Review tracking",
                results=results,
                error=None,
            )
        except Exception as exc:
            self.store.update(
                job_id,
                status="failed",
                message="Failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _clear_render_outputs(output_dir: Path) -> None:
        for name in ("plans", "renders"):
            shutil.rmtree(output_dir / name, ignore_errors=True)
        for name in (
            "report.html",
            "comparison.mp4",
            "results.zip",
            "run_manifest.json",
            "auto_fix_trace.json",
            "performance.json",
        ):
            (output_dir / name).unlink(missing_ok=True)

    def _run_render(
        self,
        job_id: str,
        input_path: Path,
        profiles: list[str],
        output_size: str,
        quality: str,
        create_comparison: bool,
        target_specs: list[TargetSpec],
        precomputed_tracks: Path | None,
        framing_corrections: list[FramingCorrection],
        precomputed_bundle: Path | None,
        speed_options: dict[str, Any],
    ) -> None:
        settings = QUALITY_PRESETS[quality]
        output_dir = self.store.output_dir(job_id)
        self._clear_render_outputs(output_dir)
        current_progress = 0.03
        planning_index = 0
        has_guidance = bool(target_specs or precomputed_tracks)

        def report(message: str) -> None:
            nonlocal current_progress, planning_index
            track_match = _PROGRESS_TRACK.search(message)
            analysis_match = _PROGRESS_ANALYSIS.search(message)
            render_match = _PROGRESS_RENDER.search(message)
            encode_match = _PROGRESS_ENCODE.search(message)

            if message.startswith("Tracking selected"):
                current_progress = max(current_progress, 0.04)
            elif track_match:
                done, total = (int(value) for value in track_match.groups())
                current_progress = max(current_progress, 0.04 + 0.18 * done / max(total, 1))
            elif message.startswith("Analysing the source"):
                current_progress = max(current_progress, 0.23 if has_guidance else 0.04)
            elif analysis_match:
                done, total = (int(value) for value in analysis_match.groups())
                base = 0.24 if has_guidance else 0.05
                span = 0.25 if has_guidance else 0.40
                current_progress = max(current_progress, base + span * done / max(total, 1))
            elif message.startswith("Planning profile:"):
                planning_index += 1
                current_progress = max(
                    current_progress,
                    0.50 + 0.05 * planning_index / max(len(profiles), 1),
                )
            elif message.startswith("Reviewing "):
                current_progress = max(current_progress, 0.56)
            elif message.startswith("Rendering all profiles"):
                current_progress = max(current_progress, 0.58)
            elif render_match and render_match.group(2) != "?":
                done = int(render_match.group(1))
                total = int(render_match.group(2))
                current_progress = max(current_progress, 0.59 + 0.25 * done / max(total, 1))
            elif encode_match:
                done = int(encode_match.group(1))
                total = int(encode_match.group(2))
                current_progress = max(current_progress, 0.85 + 0.06 * done / max(total, 1))
            elif message.startswith("Building the selected-target"):
                current_progress = max(current_progress, 0.92)
            elif message.startswith("Building the labelled"):
                current_progress = max(current_progress, 0.94)
            elif message.startswith("Writing the report"):
                current_progress = max(current_progress, 0.97)

            self.store.append_log(
                job_id,
                message,
                progress=current_progress,
                status="running",
            )

        try:
            self.store.append_log(job_id, "Starting render", progress=0.03, status="running")
            tracks = load_tracks(precomputed_tracks) if precomputed_tracks else None
            if tracks is not None:
                report("Using approved tracking review")
            result = run_demo(
                input_path,
                output_dir,
                profiles,
                output_size=output_size,
                analysis_height=int(settings["analysis_height"]),
                analysis_fps=float(settings["analysis_fps"]),
                candidate_step=int(settings["candidate_step"]),
                enable_learned_tracking=False,
                adaptive_dense_flow=bool(settings.get("adaptive_dense_flow", True)),
                advanced_vision=False,
                enable_depth=False,
                create_comparison=create_comparison,
                target_specs=target_specs,
                precomputed_tracks=tracks,
                framing_corrections=framing_corrections,
                precomputed_bundle=precomputed_bundle,
                consistent_framing=bool(speed_options.get("consistent_framing", True)),
                active_speaker=bool(speed_options.get("active_speaker", True)),
                use_analysis_proxy=bool(speed_options.get("analysis_proxy", True)),
                coarse_to_fine=bool(speed_options.get("coarse_to_fine", True)),
                detector_cadence=bool(speed_options.get("detector_cadence", True)),
                direct_ffmpeg=bool(speed_options.get("direct_ffmpeg", True)),
                static_scene_reuse=bool(speed_options.get("static_scene_reuse", True)),
                persistent_models=bool(speed_options.get("persistent_models", True)),
                dnn_optimization=bool(speed_options.get("dnn_optimization", True)),
                dnn_batch_size=max(1, int(speed_options.get("dnn_batch_size", 4))),
                pipeline_queues=bool(speed_options.get("pipeline_queues", True)),
                progress=report,
            )

            archive_path = output_dir / "results.zip"
            report("Packing results")
            _archive_results(output_dir, archive_path)

            base_url = f"/runs/{job_id}"
            outputs = [
                {
                    "profile": name,
                    "label": name.replace("_", " "),
                    "name": variant_name(name)[0],
                    "hint": variant_name(name)[1],
                    "recommended": name in RECOMMENDED_PROFILES,
                    "description": get_profile(name).description,
                    "url": f"{base_url}/renders/{path.name}",
                }
                for name, path in result.rendered_paths.items()
            ]
            recommended = next(
                (item for item in outputs if item["recommended"]),
                outputs[-1],
            )
            comparison_url = (
                f"{base_url}/{result.comparison_path.name}"
                if result.comparison_path is not None
                else None
            )
            tracking_preview_url = (
                f"{base_url}/{result.tracking_preview_path.name}"
                if result.tracking_preview_path is not None
                else None
            )
            tracks_url = (
                f"{base_url}/{result.tracks_path.name}"
                if result.tracks_path is not None
                else None
            )
            tracking_review_url = (
                f"{base_url}/{result.tracking_review_path.name}"
                if result.tracking_review_path is not None
                else None
            )
            agent_trace_url = (
                f"{base_url}/{result.agent_trace_path.name}"
                if result.agent_trace_path is not None
                else None
            )
            performance_url = (
                f"{base_url}/{result.performance_path.name}"
                if result.performance_path is not None
                else None
            )
            corrections_url = (
                f"{base_url}/{result.corrections_path.name}"
                if result.corrections_path is not None
                else None
            )
            results = {
                "report_url": f"{base_url}/{result.report_path.name}",
                "performance_url": performance_url,
                "corrections_url": corrections_url,
                "agent_trace_url": agent_trace_url,
                "comparison_url": comparison_url,
                "tracking_preview_url": tracking_preview_url,
                "tracks_url": tracks_url,
                "tracking_review_url": tracking_review_url,
                "source_video_url": f"/source/{job_id}",
                "primary_video_url": comparison_url or recommended["url"],
                "download_url": f"{base_url}/{archive_path.name}",
                "outputs": outputs,
            }
            self.store.update(
                job_id,
                status="completed",
                progress=1.0,
                message="Done",
                results=results,
                error=None,
            )
        except Exception as exc:
            self.store.update(
                job_id,
                status="failed",
                message="Failed",
                error=f"{type(exc).__name__}: {exc}",
            )


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reframe Lab</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgdmlld0JveD0iMCAwIDY0IDY0Ij48cmVjdCB3aWR0aD0iNjQiIGhlaWdodD0iNjQiIHJ4PSIxNCIgZmlsbD0iIzZmM2E4ZiIvPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDYuNCA2LjQpIHNjYWxlKDAuOCkiPjxyZWN0IHg9IjYuNSIgeT0iMTUuNSIgd2lkdGg9IjUxIiBoZWlnaHQ9IjMzIiByeD0iNiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjgiLz48cmVjdCB4PSIxNSIgeT0iMTMiIHdpZHRoPSIyMSIgaGVpZ2h0PSIzOCIgcng9IjUiIGZpbGw9IiNmZmZmZmYiLz48L2c+PC9zdmc+">
<style>
:root{--bg:#f4f5f7;--surface:#fff;--ink:#1b1d21;--muted:#6a707a;--line:#e1e4e8;--accent:#6f3a8f;--accent-ink:#fff;--accent-soft:#f1e9f6;--stage:#0d0e10;--error:#a3242c;color-scheme:light;font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
@media(prefers-color-scheme:dark){:root{--bg:#131418;--surface:#1b1c21;--ink:#ebedf0;--muted:#9198a3;--line:#2c2f36;--accent:#c08ddc;--accent-ink:#1d1022;--accent-soft:#2a2131;--error:#ff8a8f;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg)}
main{max-width:980px;margin:0 auto;padding:0 20px 64px}
header{max-width:980px;margin:0 auto;padding:22px 20px 18px;display:flex;align-items:center;justify-content:space-between}
h1{font-size:1.05rem;font-weight:650;letter-spacing:-.01em;margin:0;display:flex;align-items:center;gap:9px}
h1 svg{color:var(--accent);flex:none}
button,select,input{font:inherit;color:inherit}
button{cursor:pointer;border:1px solid var(--line);background:var(--surface);border-radius:8px;min-height:36px;padding:0 13px}
button:hover{border-color:color-mix(in srgb,var(--ink) 30%,var(--line))}
button:disabled{opacity:.4;cursor:default}
button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible,.drop:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600;padding:0 20px}
.primary:hover{border-color:var(--accent);filter:brightness(1.08)}
.ghost{border-color:transparent;background:transparent;color:var(--muted)}.ghost:hover{color:var(--ink);border-color:transparent}
.icon{width:36px;padding:0;display:inline-grid;place-items:center}
.icon svg{width:18px;height:18px}
select{border:1px solid var(--line);background:var(--surface);border-radius:7px;min-height:30px;padding:0 6px;font-size:.85rem}
[hidden]{display:none!important}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px}

/* drop zone */
.drop{position:relative;display:grid;place-items:center;gap:6px;min-height:min(56vh,460px);border-radius:12px;background:var(--surface);cursor:pointer;text-align:center;border:1px solid var(--line);transition:background .15s}
.sampleRow{text-align:center;margin:10px 0 0}#drop[hidden]+.sampleRow{display:none}
.drop.over{background:var(--accent-soft)}
.drop b{font-size:1.1rem;font-weight:600}.drop span{color:var(--muted);font-size:.9rem}
.marks4{position:absolute;inset:22px;pointer-events:none}
.marks4 i{position:absolute;width:22px;height:22px;border-color:var(--accent);border-style:solid;border-width:0}
.marks4 i:nth-child(1){left:0;top:0;border-left-width:2px;border-top-width:2px}.marks4 i:nth-child(2){right:0;top:0;border-right-width:2px;border-top-width:2px}
.marks4 i:nth-child(3){left:0;bottom:0;border-left-width:2px;border-bottom-width:2px}.marks4 i:nth-child(4){right:0;bottom:0;border-right-width:2px;border-bottom-width:2px}
#video{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}

/* editor */
.editor{padding:12px}
#previewWrap{position:relative;background:var(--stage);border-radius:8px;overflow:hidden}
#preview{display:block;width:100%;max-height:60vh;object-fit:contain;background:var(--stage)}
#overlay{position:absolute;inset:0;width:100%;height:100%;cursor:pointer}
#overlay.selecting{cursor:crosshair}
.stageHint{position:absolute;left:50%;top:12px;transform:translateX(-50%);background:rgba(0,0,0,.66);color:#fff;font-size:.8rem;padding:5px 11px;border-radius:999px;pointer-events:none;white-space:nowrap}
.transport{display:flex;align-items:center;gap:10px;margin-top:10px}
.time{font-variant-numeric:tabular-nums;font-size:.82rem;color:var(--muted);min-width:4.4em}
.scrub{position:relative;flex:1;height:30px;cursor:pointer;touch-action:none}
.scrub .track{position:absolute;left:0;right:0;top:13px;height:4px;border-radius:2px;background:var(--line)}
.scrub .fill{position:absolute;left:0;top:13px;height:4px;border-radius:2px;background:var(--ink)}
.scrub .thumb{position:absolute;top:8px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--ink);border:2px solid var(--surface)}
.scrub .pin{position:absolute;top:3px;width:8px;height:8px;margin-left:-4px;border-radius:50%;cursor:pointer}
.scrub .pin.frame{border-radius:1px;transform:rotate(45deg);background:var(--accent)}
.scrub .pin.flag{top:22px;background:var(--error);border-radius:1px;width:3px;margin-left:-1.5px;height:8px}
.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-top:10px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;padding:2px;gap:2px;background:var(--bg)}
.seg button,.seg label{border:0;border-radius:7px;min-height:30px;padding:0 12px;background:transparent;display:inline-flex;align-items:center;gap:7px;cursor:pointer;font-size:.88rem;color:var(--muted)}
.seg button.active,.seg input:checked+span{color:var(--ink)}
.seg button.active,.seg label:has(input:checked){background:var(--surface);box-shadow:0 0 0 1px var(--line);color:var(--ink)}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg label:has(input:focus-visible){outline:2px solid var(--accent);outline-offset:1px}
.seg kbd{font:inherit;font-size:.72rem;color:var(--muted);opacity:.7}
.toolHint{color:var(--muted);font-size:.84rem}
.key{min-height:36px;width:140px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:0 10px}
.variant.source{cursor:default;grid-column:1/-1;justify-self:start;max-width:min(100%,460px)}
.variant.source:hover{background:transparent}
.variant.source figcaption b::before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--muted);margin-right:7px;vertical-align:1px}

.marks{list-style:none;margin:10px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:6px}
.marks:empty{display:none}
.mark{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:2px 4px 2px 10px;font-size:.84rem;background:var(--surface)}
.mark.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.mark .name{border:0;background:transparent;min-height:26px;padding:0;display:inline-flex;align-items:center;gap:7px}
.mark .dot{width:9px;height:9px;border-radius:50%}
.mark .dia{width:8px;height:8px;background:var(--accent);transform:rotate(45deg);border-radius:1px}
.mark .count{color:var(--muted);font-variant-numeric:tabular-nums}
.mark select{min-height:26px;border-radius:999px;padding:0 4px;border-color:transparent;background:var(--bg)}
.mark .x{border:0;background:transparent;min-height:26px;width:26px;padding:0;color:var(--muted);border-radius:50%}
.mark .x:hover{background:var(--bg);color:var(--ink)}

/* settings bar */
.settings{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-top:12px;padding:12px;border-top:1px solid var(--line)}
.shapes{display:flex;flex-wrap:wrap;gap:4px;border:0;margin:0;padding:0}
.shapes label{position:relative;display:grid;justify-items:center;align-content:end;gap:4px;width:52px;height:56px;padding-bottom:5px;border-radius:8px;cursor:pointer;color:var(--muted);font-size:.72rem;font-variant-numeric:tabular-nums}
.shapes label:hover{background:var(--bg)}
.shapes input{position:absolute;opacity:0;pointer-events:none}
.shapes .box{border:1.5px solid currentColor;border-radius:2px}
.shapes label:has(input:checked){color:var(--accent);background:var(--accent-soft)}
.shapes label:has(input:focus-visible){outline:2px solid var(--accent)}
.shapes .plus{font-size:1.1rem;line-height:22px}
.customSize{display:flex;align-items:center;gap:6px;font-size:.85rem;color:var(--muted)}
.customSize input{width:76px;min-height:32px;border:1px solid var(--line);border-radius:7px;background:var(--surface);padding:0 8px;font-variant-numeric:tabular-nums}
.spacer{flex:1}
.advanced{padding:4px 12px 14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}
.advanced h3{font-size:.78rem;font-weight:600;color:var(--muted);margin:0 0 6px}
.advanced label{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:5px 0;font-size:.88rem}
.switch{appearance:none;width:30px;height:18px;border-radius:9px;background:var(--line);position:relative;cursor:pointer;flex:none;margin:0;transition:background .15s}
.switch::after{content:"";position:absolute;left:2px;top:2px;width:14px;height:14px;border-radius:50%;background:var(--surface);transition:left .15s}
.switch:checked{background:var(--accent)}.switch:checked::after{left:14px}

/* progress, review, results */
.state{margin-top:14px;padding:14px 16px}
.bar{height:4px;border-radius:2px;background:var(--line);overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .4s}
.stateRow{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:9px;font-size:.88rem}
#message.error{color:var(--error)}
details summary{cursor:pointer;color:var(--muted);font-size:.84rem}
pre{white-space:pre-wrap;color:var(--muted);font-size:.78rem;max-height:220px;overflow:auto;margin:8px 0 0}
.review{margin-top:14px;padding:14px}
.review video{display:block;width:100%;max-height:52vh;background:var(--stage);border-radius:8px}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:10px}
.flagBtn{min-height:30px;font-size:.84rem;font-variant-numeric:tabular-nums}
.result{margin-top:14px;padding:14px}
.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,var(--card,220px)),1fr));gap:12px;justify-content:center}
.variant{margin:0;border-radius:10px;padding:6px;border:1px solid transparent;cursor:pointer;max-width:var(--cardMax,300px);justify-self:center;width:100%}
.variant:hover{background:var(--bg)}
.variant.selected{border-color:var(--accent);background:var(--accent-soft)}
.variant video{display:block;width:100%;background:var(--stage);border-radius:6px}
.variant figcaption{display:grid;gap:1px;padding:8px 4px 2px}
.variant figcaption b{font-weight:600;font-size:.92rem}
.variant figcaption span{color:var(--muted);font-size:.78rem}
.resultBar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}
.resultBar .scrub{min-width:120px;margin-right:8px}
a.button{display:inline-flex;align-items:center;min-height:36px;border-radius:8px;text-decoration:none}
.more{position:relative}
.more summary{list-style:none;min-height:36px;display:inline-grid;place-items:center;width:36px;border-radius:8px;border:1px solid var(--line);color:var(--ink)}
.more summary::-webkit-details-marker{display:none}
.more .menu{position:absolute;right:0;bottom:44px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px;min-width:190px;box-shadow:0 8px 24px rgba(0,0,0,.12);z-index:5}
.more .menu a{display:block;padding:7px 10px;border-radius:6px;color:var(--ink);text-decoration:none;font-size:.88rem}
.more .menu a:hover{background:var(--bg)}
@media(max-width:640px){main,header{padding-left:16px;padding-right:16px}.variants.tall{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.variants.tall .variant{padding:4px}.resultBar .scrub{flex:1 1 calc(100% - 56px);margin-right:0}.settings{padding:12px 0}.primary{flex:1}.toolHint{display:none}.seg kbd{display:none}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<header><h1><svg viewBox="0 0 64 64" width="22" height="22" aria-hidden="true"><rect x="6.5" y="15.5" width="51" height="33" rx="6" fill="none" stroke="currentColor" stroke-width="6"/><rect x="15" y="13" width="21" height="38" rx="5" fill="currentColor"/></svg>Reframe Lab</h1><button id="newRun" class="ghost" type="button" hidden>New video</button></header>
<main>
<label id="drop" class="drop" tabindex="0">
<div class="marks4"><i></i><i></i><i></i><i></i></div>
<div><b>Drop a video</b><br><span>or click to choose</span></div>
</label>
<p id="sampleRow" class="sampleRow" hidden><button id="sample" class="ghost" type="button">Try a sample clip</button></p>
<form id="form" class="panel" hidden>
<input id="video" name="video" type="file" accept="video/*,.mkv,.m4v,.ogv">
<input id="targets" name="targets" type="hidden" value="[]">
<input id="corrections" name="corrections" type="hidden" value="[]">
<input name="preset" type="hidden" value="all">
<input id="outputSize" name="output_size" type="hidden" value="540x960">
<section id="previewPanel" class="editor">
<div id="previewWrap"><video id="preview" playsinline preload="metadata"></video><canvas id="overlay"></canvas><div id="stageHint" class="stageHint" hidden></div></div>
<div class="transport">
<button id="playButton" class="icon ghost" type="button" aria-label="Play"><svg viewBox="0 0 20 20" fill="currentColor"><path id="playIcon" d="M6 4l10 6-10 6z"/></svg></button>
<span id="timeLabel" class="time">0:00.0</span>
<div id="scrub" class="scrub" role="slider" tabindex="0" aria-label="Position"><div class="track"></div><div class="fill"></div><div class="pins"></div><div class="thumb"></div></div>
<span id="durationLabel" class="time" style="text-align:right">0:00.0</span>
</div>
<div class="toolbar">
<div class="seg" role="group" aria-label="Edit tools">
<button type="button" data-mode="follow" title="Follow: click a point or drag a box (1)">Follow <kbd>1</kbd></button>
<button type="button" data-mode="keep" title="Keep in view: drag a box (2)">Keep in view <kbd>2</kbd></button>
<button type="button" data-mode="frame" title="Frame: click where the crop should sit (3)">Frame <kbd>3</kbd></button>
</div>
<span id="toolHint" class="toolHint"></span>
</div>
<ul id="marks" class="marks"></ul>
</section>
<div class="settings">
<fieldset class="shapes" id="shapes" aria-label="Output shape">
<label title="Vertical 540×960"><input type="radio" name="shape" value="540x960" checked><span class="box" style="width:13px;height:23px"></span>9:16</label>
<label title="Portrait 864×1080"><input type="radio" name="shape" value="864x1080"><span class="box" style="width:18px;height:22px"></span>4:5</label>
<label title="Square 960×960"><input type="radio" name="shape" value="960x960"><span class="box" style="width:20px;height:20px"></span>1:1</label>
<label title="Landscape 1280×720"><input type="radio" name="shape" value="1280x720"><span class="box" style="width:28px;height:16px"></span>16:9</label>
<label title="Cinema 1280×544"><input type="radio" name="shape" value="1280x544"><span class="box" style="width:30px;height:13px"></span>2.35</label>
<label title="Custom size"><input type="radio" name="shape" value="custom"><span class="plus">+</span>Custom</label>
</fieldset>
<div id="customSize" class="customSize" hidden>
<input id="customWidth" type="number" min="64" max="4096" step="2" value="1080" inputmode="numeric" aria-label="Width">×
<input id="customHeight" type="number" min="64" max="4096" step="2" value="1920" inputmode="numeric" aria-label="Height">
</div>
<span class="spacer"></span>
<div class="seg" role="radiogroup" aria-label="Analysis quality">
<label title="Faster analysis"><input type="radio" name="quality" value="quick">Fast</label>
<label><input type="radio" name="quality" value="balanced" checked>Balanced</label>
<label title="Slower, more detailed analysis"><input type="radio" name="quality" value="careful">Precise</label>
</div>
<button id="advancedToggle" class="icon" type="button" aria-label="Advanced settings" aria-expanded="false" title="Advanced settings"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M3 6h8M15 6h2M3 14h2M9 14h8"/><circle cx="13" cy="6" r="2"/><circle cx="7" cy="14" r="2"/></svg></button>
<button id="run" class="primary" type="submit">Reframe</button>
</div>
<div id="advanced" class="advanced" hidden>
<div>
<h3>Framing</h3>
<label>Same framing for repeated shots<input id="consistentFraming" class="switch" type="checkbox" checked></label>
<label>Favour the person speaking<input id="activeSpeaker" class="switch" type="checkbox" checked></label>
</div>
<div>
<h3>Speed</h3>
<label>Analyse a small copy<input id="analysisProxy" class="switch" type="checkbox" checked></label>
<label>Refine hard sections only<input id="coarseToFine" class="switch" type="checkbox" checked></label>
<label>Detect text and people less often<input id="detectorCadence" class="switch" type="checkbox" checked></label>
<label>Reuse unchanged scenes<input id="staticSceneReuse" class="switch" type="checkbox" checked></label>
</div>
<div>
<h3>Engine</h3>
<label>Keep models loaded<input id="persistentModels" class="switch" type="checkbox" checked></label>
<label>Tune OpenCV DNN<input id="dnnOptimization" class="switch" type="checkbox" checked></label>
<label>Batch size<select id="dnnBatchSize"><option value="1">1</option><option value="2">2</option><option value="4" selected>4</option><option value="8">8</option></select></label>
<label>Read and encode in parallel<input id="pipelineQueues" class="switch" type="checkbox" checked></label>
<label>Encode straight to FFmpeg<input id="directFfmpeg" class="switch" type="checkbox" checked></label>
</div>
</div>
</form>
<section id="state" class="panel state" hidden>
<div class="bar"><i id="progress"></i></div>
<div class="stateRow"><span id="message"></span><details><summary>Log</summary></details></div>
<pre id="log" hidden></pre>
</section>
<section id="review" class="panel review" hidden>
<video id="reviewPlayer" controls playsinline preload="metadata"></video>
<div class="row"><span id="reviewSummary" class="toolHint"></span><span id="reviewPoints" class="row" style="margin:0"></span><span class="spacer"></span><button id="renderReviewed" class="primary" type="button">Reframe</button></div>
<div id="reviewLinks" hidden></div>
</section>
<section id="result" class="panel result" hidden>
<div id="variants" class="variants"></div>
<div class="resultBar">
<button id="playAll" class="icon ghost" type="button" aria-label="Play all"><svg viewBox="0 0 20 20" fill="currentColor"><path id="playAllIcon" d="M6 4l10 6-10 6z"/></svg></button>
<div id="resultScrub" class="scrub" role="slider" tabindex="0" aria-label="Position"><div class="track"></div><div class="fill"></div><div class="thumb"></div></div>
<button id="adjustFraming" type="button">Edit</button>
<button id="applyCorrections" class="primary" type="button" hidden>Update videos</button>
<a id="downloadSelected" class="button primary" href="#" download>Download</a>
<details class="more"><summary aria-label="More files" title="More files"><svg viewBox="0 0 20 20" width="18" height="18" fill="currentColor"><circle cx="4" cy="10" r="1.6"/><circle cx="10" cy="10" r="1.6"/><circle cx="16" cy="10" r="1.6"/></svg></summary><div id="links" class="menu"></div></details>
</div>
</section>
</main>
<script>
const CLOUD=false;
const $=id=>document.getElementById(id);
const form=$('form'),drop=$('drop'),fileInput=$('video'),targetsInput=$('targets'),correctionsInput=$('corrections');
const previewPanel=$('previewPanel'),preview=$('preview'),wrap=$('previewWrap'),canvas=$('overlay'),stageHint=$('stageHint');
const playButton=$('playButton'),timeLabel=$('timeLabel'),durationLabel=$('durationLabel'),scrub=$('scrub'),toolHint=$('toolHint'),marksList=$('marks');
const runButton=$('run'),outputSize=$('outputSize'),customSize=$('customSize'),customWidth=$('customWidth'),customHeight=$('customHeight');
const analysisProxy=$('analysisProxy'),coarseToFine=$('coarseToFine'),detectorCadence=$('detectorCadence'),directFfmpeg=$('directFfmpeg'),staticSceneReuse=$('staticSceneReuse'),persistentModels=$('persistentModels'),dnnOptimization=$('dnnOptimization'),dnnBatchSize=$('dnnBatchSize'),pipelineQueues=$('pipelineQueues'),consistentFraming=$('consistentFraming'),activeSpeaker=$('activeSpeaker');
const state=$('state'),progress=$('progress'),message=$('message'),log=$('log');
const review=$('review'),reviewPlayer=$('reviewPlayer'),reviewSummary=$('reviewSummary'),reviewPoints=$('reviewPoints'),renderReviewed=$('renderReviewed'),reviewLinks=$('reviewLinks');
const result=$('result'),variantsBox=$('variants'),links=$('links'),adjustFraming=$('adjustFraming'),applyCorrections=$('applyCorrections'),downloadSelected=$('downloadSelected'),playAll=$('playAll'),resultScrub=$('resultScrub'),newRun=$('newRun');
const toolButtons=[...document.querySelectorAll('[data-mode]')];
const palette=['#2f9e5b','#e08a12','#3b7be8','#a24fd6','#e0484e','#1aa3b8','#5b5fe0','#d0438c'];
const IDLE_HINT='Optional: follow something, keep an area in view, or set the frame for a shot';
const HINTS={follow:'Click a point or drag a box',keep:'Drag a box around what must stay in view',frame:'Click where the crop should sit',anchor:'Mark the same thing at this time'};
let timer=null,objectUrl=null,mode=null,draft=null,hover=null,selectedId=null,currentJobId=null,lastJob=null,submittedTargets='[]';
let targetItems=[],correctionItems=[],flags=[],resultVideos=[],selectedVariant=null,editingResult=false;

function uid(){return (crypto.randomUUID?crypto.randomUUID():Date.now().toString(36)+Math.random().toString(36).slice(2));}
function clamp(value,min=0,max=1){return Math.max(min,Math.min(max,value));}
function shapeValue(){return (form.querySelector('input[name=shape]:checked')||{}).value||'540x960';}
function syncOutputSize(){
  const value=shapeValue();
  if(value!=='custom'){outputSize.value=value;customSize.hidden=true;return true;}
  customSize.hidden=false;
  let width=Number.parseInt(customWidth.value,10),height=Number.parseInt(customHeight.value,10);
  if(!Number.isFinite(width)||!Number.isFinite(height)||width<64||height<64||width>4096||height>4096){
    customWidth.reportValidity();customHeight.reportValidity();return false;
  }
  width-=width%2;height-=height%2;outputSize.value=width+'x'+height;return true;
}
function formatTime(value){
  const seconds=Number.isFinite(value)&&value>=0?value:0;
  const minutes=Math.floor(seconds/60);
  return minutes+':'+(seconds-minutes*60).toFixed(1).padStart(4,'0');
}
function duration(){return Number.isFinite(preview.duration)&&preview.duration>0?preview.duration:0;}
function seek(value){preview.pause();preview.currentTime=clamp(value,0,duration()||value);}
function updateTransport(){
  timeLabel.textContent=formatTime(preview.currentTime);durationLabel.textContent=formatTime(duration());
  const ratio=duration()?preview.currentTime/duration():0;
  scrub.querySelector('.fill').style.width=(ratio*100)+'%';scrub.querySelector('.thumb').style.left=(ratio*100)+'%';
  $('playIcon').setAttribute('d',preview.paused?'M6 4l10 6-10 6z':'M5 4h3.5v12H5zM11.5 4H15v12h-3.5z');
  playButton.setAttribute('aria-label',preview.paused?'Play':'Pause');
}
function renderPins(){
  const box=scrub.querySelector('.pins');box.replaceChildren();const total=duration();if(!total)return;
  const add=(time,cls,color,title)=>{const pin=document.createElement('span');pin.className='pin '+cls;pin.style.left=(clamp(time/total)*100)+'%';if(color)pin.style.background=color;pin.title=title;pin.addEventListener('pointerdown',event=>{event.stopPropagation();seek(time);});box.appendChild(pin);};
  targetItems.forEach((target,index)=>target.anchors.forEach(anchor=>add(anchor.time,'',palette[index%palette.length],target.label+' '+formatTime(anchor.time))));
  correctionItems.forEach(item=>add(item.time,'frame','',formatTime(item.time)));
  flags.forEach(point=>add(point.time,'flag','',point.reason||formatTime(point.time)));
}
function bindScrub(element,getDuration,onSeek){
  const at=event=>{const rect=element.getBoundingClientRect();return clamp((event.clientX-rect.left)/Math.max(rect.width,1))*getDuration();};
  element.addEventListener('pointerdown',event=>{if(!getDuration())return;element.setPointerCapture(event.pointerId);onSeek(at(event));const move=e=>onSeek(at(e));const up=()=>{element.removeEventListener('pointermove',move);element.removeEventListener('pointerup',up);};element.addEventListener('pointermove',move);element.addEventListener('pointerup',up);});
  element.addEventListener('keydown',event=>{const total=getDuration();if(!total)return;const step=event.shiftKey?1:.1;if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();event.stopPropagation();onSeek(null,event.key==='ArrowRight'?step:-step);}});
}
function setBusy(busy){runButton.disabled=busy;runButton.dataset.busy=busy?'1':'';for(const el of form.querySelectorAll('input,select,button'))if(el!==runButton)el.disabled=busy||el.dataset.locked==='true';renderReviewed.disabled=busy;applyCorrections.disabled=busy;adjustFraming.disabled=busy;if(busy)setMode(null);}
function showState(text,value){state.hidden=false;message.textContent=text;message.className='';progress.style.width=(clamp(value||0)*100)+'%';}
function formatApiError(body,fallback){
  const detail=body&&body.detail;
  if(typeof detail==='string'&&detail.trim())return detail;
  if(Array.isArray(detail)){
    const items=detail.map(item=>{
      if(typeof item==='string')return item;
      if(!item||typeof item!=='object')return String(item||'');
      const location=Array.isArray(item.loc)?item.loc.filter(part=>part!=='body').join('.'):'';
      if(location==='video'&&item.type==='missing')return 'Choose a video file before starting.';
      const text=typeof item.msg==='string'?item.msg:'Invalid request';
      return location?location+': '+text:text;
    }).filter(Boolean);
    if(items.length)return items.join('\n');
  }
  if(detail&&typeof detail==='object'){
    if(typeof detail.msg==='string')return detail.msg;
    try{return JSON.stringify(detail);}catch(error){}
  }
  if(body&&typeof body.message==='string')return body.message;
  return fallback;
}
function fail(text){const value=typeof text==='string'?text:String(text||'Request failed');setBusy(false);state.hidden=false;message.textContent=value;message.className='error';if(!log.textContent.trim())log.textContent=value;}
function addLink(label,url,download=false){if(!url)return;const a=document.createElement('a');a.textContent=label;a.href=url;if(download)a.download='';else a.target='_blank';links.appendChild(a);}
function showEditor(){drop.hidden=true;form.hidden=false;requestAnimationFrame(resizeCanvas);}
function loadSource(url){if(preview.src)return;preview.src=url;preview.load();showEditor();}

function showReview(job){
  setBusy(false);currentJobId=job.id;lastJob=job;review.hidden=false;result.hidden=true;state.hidden=true;reviewPoints.replaceChildren();reviewLinks.replaceChildren();
  if((!targetItems.length)&&job.targets){targetItems=JSON.parse(JSON.stringify(job.targets));selectedId=targetItems[0]?.id||null;syncTargets();}
  if((!correctionItems.length)&&job.corrections){correctionItems=JSON.parse(JSON.stringify(job.corrections));syncCorrections();}
  if(job.results.source_video_url)loadSource(job.results.source_video_url);
  reviewPlayer.src=job.results.tracking_preview_url;reviewPlayer.load();
  const points=(job.results.review&&job.results.review.points)||[];flags=points;renderPins();
  reviewSummary.textContent=points.length?'Check these moments':'Tracking looks steady';
  for(const point of points){const button=document.createElement('button');button.type='button';button.className='flagBtn';button.textContent=formatTime(point.time);button.title=point.reason;button.onclick=()=>{seek(point.time);selectedId=point.target_id;renderMarks();draw();wrap.scrollIntoView({behavior:'smooth',block:'center'});};reviewPoints.appendChild(button);}
  submittedTargets=JSON.stringify(targetItems);
}
function variantCard(item){
  const figure=document.createElement('figure');figure.className='variant';figure.tabIndex=0;
  const video=document.createElement('video');if(!item.signed)item.url=item.url.split('?')[0]+'?v='+encodeURIComponent(lastJob&&lastJob.updated_at||Date.now());video.src=item.url;video.preload='metadata';video.playsInline=true;video.muted=true;video.loop=true;
  const size=item.source?[preview.videoWidth||(lastJob&&lastJob.source&&lastJob.source.width)||16,preview.videoHeight||(lastJob&&lastJob.source&&lastJob.source.height)||9]:(lastJob&&lastJob.output_size||outputSize.value).split('x').map(Number);video.style.aspectRatio=(size[0]||9)+'/'+(size[1]||16);
  if(item.source){figure.classList.add('source');figure.removeAttribute('tabindex');}
  const caption=document.createElement('figcaption');const name=document.createElement('b');name.textContent=item.name||item.label;const hint=document.createElement('span');hint.textContent=item.hint||'';caption.append(name,hint);
  figure.append(video,caption);figure.title=item.description||'';
  const select=()=>{if(!item.source)selectVariant(item,figure);};figure.addEventListener('click',select);figure.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select();}});
  return {figure,video,item};
}
function selectVariant(item,figure){
  selectedVariant=item;for(const entry of resultVideos){entry.figure.classList.toggle('selected',entry.item===item);entry.video.muted=entry.item!==item;}
  if(item.signed){downloadSelected.href=item.url;downloadSelected.removeAttribute('download');downloadSelected.target='_blank';return;}
  downloadSelected.href=item.url;downloadSelected.setAttribute('download',(lastJob?.original_name||'video').replace(/\.[^.]+$/,'')+'-'+(item.name||item.profile).toLowerCase().replace(/\s+/g,'-')+'.mp4');
}
function leadVideo(){return resultVideos[0]?.video;}
function updateResultTransport(){
  const lead=leadVideo();if(!lead)return;const total=Number.isFinite(lead.duration)?lead.duration:0;const ratio=total?lead.currentTime/total:0;
  resultScrub.querySelector('.fill').style.width=(ratio*100)+'%';resultScrub.querySelector('.thumb').style.left=(ratio*100)+'%';
  $('playAllIcon').setAttribute('d',lead.paused?'M6 4l10 6-10 6z':'M5 4h3.5v12H5zM11.5 4H15v12h-3.5z');
}
function syncAll(time){for(const entry of resultVideos)entry.video.currentTime=time;}
function showResult(job){
  setBusy(false);review.hidden=true;result.hidden=false;state.hidden=true;links.replaceChildren();variantsBox.replaceChildren();currentJobId=job.id;lastJob=job;flags=[];
  if(job.corrections){correctionItems=JSON.parse(JSON.stringify(job.corrections));syncCorrections();}
  if(job.targets&&!targetItems.length){targetItems=JSON.parse(JSON.stringify(job.targets));syncTargets();}
  submittedTargets=JSON.stringify(targetItems);
  if(job.results.source_video_url&&!preview.src){preview.src=job.results.source_video_url;preview.load();}
  editingResult=false;runButton.textContent='Reframe';form.hidden=true;drop.hidden=true;newRun.hidden=false;applyCorrections.hidden=true;adjustFraming.hidden=false;downloadSelected.hidden=false;
  const size=(job.output_size||'540x960').split('x').map(Number),aspect=(size[0]||9)/(size[1]||16);
  variantsBox.classList.toggle('tall',aspect<.8);variantsBox.style.setProperty('--card',aspect<.8?'190px':aspect<1.2?'240px':'320px');variantsBox.style.setProperty('--cardMax',aspect<.8?'270px':aspect<1.2?'340px':'460px');
  const outputs=(job.results.outputs||[]).slice();
  if(job.results.source_video_url)outputs.unshift({source:true,signed:true,name:'Original',hint:'Your source clip',url:job.results.source_video_url});
  resultVideos=outputs.map(variantCard);
  for(const entry of resultVideos)variantsBox.appendChild(entry.figure);
  const lead=leadVideo();if(lead){lead.addEventListener('timeupdate',updateResultTransport);lead.addEventListener('play',updateResultTransport);lead.addEventListener('pause',updateResultTransport);lead.addEventListener('loadedmetadata',updateResultTransport);}
  const first=resultVideos.find(entry=>entry.item.recommended)||resultVideos.find(entry=>!entry.item.source);if(first)selectVariant(first.item,first.figure);
  addLink('Report',job.results.report_url);
  addLink('All files (.zip)',job.results.download_url,true);
  if(job.results.agent_trace_url)addLink('Auto-fix trace',job.results.agent_trace_url,true);
  if(job.results.performance_url)addLink('Performance',job.results.performance_url,true);
  if(job.results.corrections_url)addLink('Framing fixes',job.results.corrections_url,true);
  if(job.results.tracking_preview_url)addLink('Tracking preview',job.results.tracking_preview_url,true);
  if(job.results.tracks_url)addLink('Track data',job.results.tracks_url,true);
  if(job.results.comparison_url)addLink('Comparison video',job.results.comparison_url,true);
  result.scrollIntoView({behavior:'smooth',block:'start'});
}
async function poll(id){
  clearTimeout(timer);
  try{
    const response=await fetch('/api/jobs/'+id,{cache:'no-store'});
    if(!response.ok)throw new Error('Run not found');
    const job=await response.json();
    showState(job.message,job.progress);log.textContent=(job.logs||[]).join('\n');
    if(job.status==='completed'){showResult(job);return;}
    if(job.status==='review'){showReview(job);return;}
    if(job.status==='failed'){fail(job.error||'Run failed');return;}
    timer=setTimeout(()=>poll(id),900);
  }catch(error){fail(error.message||String(error));}
}

function syncTargets(){targetsInput.value=JSON.stringify(targetItems);renderMarks();draw();}
function syncCorrections(){correctionsInput.value=JSON.stringify(correctionItems);renderMarks();draw();}
function setMode(next){
  mode=next;draft=null;canvas.classList.toggle('selecting',Boolean(mode));
  for(const button of toolButtons)button.classList.toggle('active',button.dataset.mode===mode);
  toolHint.textContent=mode?HINTS[mode]:(targetItems.length||correctionItems.length?'':IDLE_HINT);
  const target=mode==='anchor'?targetItems.find(item=>item.id===selectedId):null;
  stageHint.hidden=!target;if(target)stageHint.textContent=target.label+' at '+formatTime(preview.currentTime)+' · Esc to cancel';
  draw();
}
function videoContentRect(){
  const width=canvas.clientWidth,height=canvas.clientHeight,vw=preview.videoWidth,vh=preview.videoHeight;
  if(!vw||!vh||!width||!height)return{x:0,y:0,width,height};
  const scale=Math.min(width/vw,height/vh),contentWidth=vw*scale,contentHeight=vh*scale;
  return{x:(width-contentWidth)/2,y:(height-contentHeight)/2,width:contentWidth,height:contentHeight};
}
function videoPoint(event,clampOutside=false){
  const rect=canvas.getBoundingClientRect(),area=videoContentRect();
  let x=event.clientX-rect.left,y=event.clientY-rect.top;
  if(!clampOutside&&(x<area.x||x>area.x+area.width||y<area.y||y>area.y+area.height))return null;
  x=clamp(x,area.x,area.x+area.width);y=clamp(y,area.y,area.y+area.height);
  return{x:clamp((x-area.x)/Math.max(area.width,1)),y:clamp((y-area.y)/Math.max(area.height,1))};
}
function addSelection(anchor){
  if(mode==='anchor'){
    const target=targetItems.find(item=>item.id===selectedId);if(!target)return;
    target.anchors=target.anchors.filter(item=>Math.abs(item.time-anchor.time)>.05);target.anchors.push(anchor);target.anchors.sort((a,b)=>a.time-b.time);
  }else{
    const role=mode==='keep'?'keep_visible':'follow',kind=anchor.width>0?'box':'point';
    const count=targetItems.filter(item=>item.role===role).length+1;
    targetItems.push({id:uid(),label:(role==='keep_visible'?'Keep ':'Follow ')+count,kind,role,anchors:[anchor]});selectedId=targetItems.at(-1).id;
  }
  setMode(null);syncTargets();
}
function removeTarget(id){targetItems=targetItems.filter(item=>item.id!==id);if(selectedId===id)selectedId=null;if(mode==='anchor')setMode(null);syncTargets();}
function markButton(text,label,onclick){const button=document.createElement('button');button.type='button';button.className='x';button.textContent=text;button.setAttribute('aria-label',label);button.title=label;button.onclick=event=>{event.stopPropagation();onclick();};return button;}
function smallSelect(options,value,label,onchange){const select=document.createElement('select');select.setAttribute('aria-label',label);for(const [key,text] of options){const option=document.createElement('option');option.value=key;option.textContent=text;option.selected=key===value;select.appendChild(option);}select.onchange=()=>onchange(select.value);select.onclick=event=>event.stopPropagation();return select;}
function renderMarks(){
  marksList.replaceChildren();
  targetItems.forEach((target,index)=>{
    const li=document.createElement('li');li.className='mark'+(target.id===selectedId?' selected':'');
    const name=document.createElement('button');name.type='button';name.className='name';
    const dot=document.createElement('span');dot.className='dot';dot.style.background=palette[index%palette.length];
    const count=document.createElement('span');count.className='count';count.textContent=target.anchors.length>1?String(target.anchors.length):'';
    name.append(dot,document.createTextNode(target.label),count);
    name.onclick=()=>{selectedId=target.id;const nearest=target.anchors.reduce((a,b)=>Math.abs(b.time-preview.currentTime)<Math.abs(a.time-preview.currentTime)?b:a,target.anchors[0]);if(nearest)seek(nearest.time);renderMarks();draw();};
    li.append(name,markButton('+','Mark '+target.label+' again at this time',()=>{selectedId=target.id;renderMarks();setMode(mode==='anchor'?null:'anchor');}),markButton('×','Remove '+target.label,()=>removeTarget(target.id)));
    marksList.appendChild(li);
  });
  correctionItems.forEach((item,index)=>{
    const li=document.createElement('li');li.className='mark';
    const name=document.createElement('button');name.type='button';name.className='name';
    const dia=document.createElement('span');dia.className='dia';name.append(dia,document.createTextNode(formatTime(item.time)));
    name.onclick=()=>seek(item.time);
    li.append(name,
      smallSelect([['shot','This shot'],['matching','Matching shots'],['all','Whole video']],item.scope,'Applies to',value=>{item.scope=value;syncCorrections();}),
      smallSelect([['auto','Auto'],['crop','Crop'],['split','Split'],['context_detail','Context + detail'],['fallback','Full frame']],item.mode,'Layout',value=>{item.mode=value;syncCorrections();}),
      markButton('×','Remove framing',()=>{correctionItems.splice(index,1);syncCorrections();}));
    marksList.appendChild(li);
  });
  renderPins();refreshApply();if(!mode)toolHint.textContent=targetItems.length||correctionItems.length?'':IDLE_HINT;
}
function outputDimensions(){const parts=outputSize.value.split('x').map(Number);return{width:parts[0]||540,height:parts[1]||960};}
function correctionRect(item){
  const sourceAspect=(preview.videoWidth||16)/(preview.videoHeight||9),size=outputDimensions(),targetAspect=size.width/size.height;
  let width=1,height=1;if(sourceAspect>targetAspect)width=targetAspect/sourceAspect;else if(sourceAspect<targetAspect)height=sourceAspect/targetAspect;
  const cx=clamp(item.center_x,width/2,1-width/2),cy=clamp(item.center_y,height/2,1-height/2);
  return{x:cx-width/2,y:cy-height/2,width,height};
}
function addFramingCorrection(point){
  correctionItems.push({time:preview.currentTime,center_x:point.x,center_y:point.y,scope:'shot',mode:'auto',label:'Framing '+(correctionItems.length+1)});
  correctionItems.sort((a,b)=>a.time-b.time);setMode(null);syncCorrections();
}
function resizeCanvas(){const rect=wrap.getBoundingClientRect();const ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));canvas.style.width=rect.width+'px';canvas.style.height=rect.height+'px';draw();}
function drawShape(ctx,anchor,color,label,alpha=1){
  const area=videoContentRect(),w=area.width,h=area.height;ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
  const left=area.x+anchor.x*w,top=area.y+anchor.y*h;
  if(anchor.width>0&&anchor.height>0){ctx.strokeRect(left,top,anchor.width*w,anchor.height*h);if(label)ctx.fillText(label,left+4,Math.max(area.y+14,top-6));}
  else{ctx.beginPath();ctx.arc(left,top,7,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(left,top,2,0,Math.PI*2);ctx.fill();if(label)ctx.fillText(label,left+11,Math.max(area.y+14,top-8));}
  ctx.restore();
}
function drawCrop(ctx,item,dashed,alpha){
  const area=videoContentRect(),rect=correctionRect(item),x=area.x+rect.x*area.width,y=area.y+rect.y*area.height,w=rect.width*area.width,h=rect.height*area.height;
  ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle='rgba(0,0,0,.45)';
  ctx.fillRect(area.x,area.y,area.width,y-area.y);ctx.fillRect(area.x,y+h,area.width,area.y+area.height-y-h);ctx.fillRect(area.x,y,x-area.x,h);ctx.fillRect(x+w,y,area.x+area.width-x-w,h);
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5;if(dashed)ctx.setLineDash([6,5]);ctx.strokeRect(x,y,w,h);ctx.restore();
}
function draw(){
  const ratio=window.devicePixelRatio||1,ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);ctx.font='600 12px system-ui';
  const now=preview.currentTime;
  correctionItems.forEach(item=>{const distance=Math.abs(item.time-now);if(distance<.45||item.scope==='all')drawCrop(ctx,item,false,mode==='frame'?.35:.9);});
  if(mode==='frame'&&hover)drawCrop(ctx,{center_x:hover.x,center_y:hover.y},true,1);
  targetItems.forEach((target,index)=>{let nearest=null;for(const anchor of target.anchors){const distance=Math.abs(anchor.time-now);if(!nearest||distance<nearest.distance)nearest={anchor,distance};}if(nearest&&(nearest.distance<.45||target.id===selectedId))drawShape(ctx,nearest.anchor,palette[index%palette.length],target.label,nearest.distance<.45?1:.35);});
  if(draft){const x=Math.min(draft.start.x,draft.end.x),y=Math.min(draft.start.y,draft.end.y),width=Math.abs(draft.end.x-draft.start.x),height=Math.abs(draft.end.y-draft.start.y);drawShape(ctx,{x,y,width,height},'#fff','',1);}
}
function needsNewRun(){
  if(!lastJob||CLOUD)return true;
  return JSON.stringify(targetItems)!==submittedTargets||outputSize.value!==lastJob.output_size||(form.querySelector('input[name=quality]:checked')||{}).value!==lastJob.quality;
}
function refreshApply(){
  applyCorrections.hidden=true;
  const update=editingResult&&!needsNewRun();
  runButton.textContent=update?'Update':'Reframe';
  if(editingResult&&!runButton.dataset.busy)runButton.disabled=update&&JSON.stringify(correctionItems.map(stripCorrection))===JSON.stringify((lastJob.corrections||[]).map(stripCorrection));
}
function stripCorrection(item){return [Math.round(item.time*100),Math.round(item.center_x*1000),Math.round(item.center_y*1000),item.scope,item.mode];}

/* file selection */
function chooseFile(file){
  if(!file)return;
  if(objectUrl)URL.revokeObjectURL(objectUrl);targetItems=[];correctionItems=[];flags=[];selectedId=null;currentJobId=null;lastJob=null;
  history.replaceState(null,'','/');editingResult=false;result.hidden=true;review.hidden=true;state.hidden=true;newRun.hidden=false;for(const button of toolButtons)button.hidden=false;for(const input of form.querySelectorAll('[data-locked]'))input.dataset.locked='';setBusy(false);refreshApply();
  objectUrl=URL.createObjectURL(file);preview.removeAttribute('src');preview.src=objectUrl;preview.load();syncTargets();syncCorrections();showEditor();
}
fileInput.addEventListener('change',()=>chooseFile(fileInput.files&&fileInput.files[0]));
drop.addEventListener('click',event=>{event.preventDefault();fileInput.click();});
drop.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();fileInput.click();}});
for(const name of ['dragenter','dragover'])document.addEventListener(name,event=>{event.preventDefault();drop.classList.add('over');});
for(const name of ['dragleave','drop'])document.addEventListener(name,event=>{event.preventDefault();if(name==='dragleave'&&event.relatedTarget)return;drop.classList.remove('over');});
document.addEventListener('drop',event=>{const file=event.dataTransfer&&event.dataTransfer.files[0];if(!file||!file.type.startsWith('video')&&!/\.(mkv|m4v|mov|mp4|webm|avi)$/i.test(file.name))return;const transfer=new DataTransfer();transfer.items.add(file);fileInput.files=transfer.files;chooseFile(file);});
newRun.addEventListener('click',()=>{clearTimeout(timer);fileInput.value='';preview.pause();preview.removeAttribute('src');preview.load();targetItems=[];correctionItems=[];flags=[];currentJobId=null;lastJob=null;history.replaceState(null,'','/');form.hidden=true;result.hidden=true;review.hidden=true;state.hidden=true;drop.hidden=false;newRun.hidden=true;setBusy(false);syncTargets();syncCorrections();});

/* settings */
form.querySelectorAll('input[name=shape],input[name=quality]').forEach(input=>input.addEventListener('change',()=>{syncOutputSize();draw();refreshApply();}));
customWidth.addEventListener('input',()=>{syncOutputSize();draw();});customHeight.addEventListener('input',()=>{syncOutputSize();draw();});syncOutputSize();
if(!CLOUD)$('advancedToggle').addEventListener('click',event=>{const panel=$('advanced');panel.hidden=!panel.hidden;event.currentTarget.setAttribute('aria-expanded',String(!panel.hidden));});
if(!CLOUD)state.querySelector('summary').addEventListener('click',event=>{event.preventDefault();log.hidden=!log.hidden;});

/* player */
preview.addEventListener('loadedmetadata',()=>{resizeCanvas();updateTransport();renderPins();});
for(const name of ['timeupdate','seeked','play','pause'])preview.addEventListener(name,()=>{updateTransport();draw();if(mode==='anchor')setMode('anchor');});
window.addEventListener('resize',resizeCanvas);
playButton.addEventListener('click',()=>{if(preview.paused)preview.play();else preview.pause();});
bindScrub(scrub,duration,(time,delta)=>seek(time===null?preview.currentTime+delta:time));
bindScrub(resultScrub,()=>{const lead=leadVideo();return lead&&Number.isFinite(lead.duration)?lead.duration:0;},(time,delta)=>{const lead=leadVideo();if(!lead)return;syncAll(time===null?clamp(lead.currentTime+delta,0,lead.duration):time);updateResultTransport();});
playAll.addEventListener('click',()=>{const lead=leadVideo();if(!lead)return;if(lead.paused){syncAll(lead.currentTime);for(const entry of resultVideos)entry.video.play();}else for(const entry of resultVideos)entry.video.pause();});
for(const button of toolButtons)button.addEventListener('click',()=>setMode(mode===button.dataset.mode?null:button.dataset.mode));
document.addEventListener('keydown',event=>{
  if(form.hidden||event.target.closest('input,select,textarea')||event.metaKey||event.ctrlKey||event.altKey)return;
  if(event.target.closest('.scrub'))return;
  const keys={'1':'follow','2':'keep','3':'frame'};
  if(keys[event.key]){setMode(mode===keys[event.key]?null:keys[event.key]);event.preventDefault();}
  else if(event.key==='Escape'){setMode(null);}
  else if(event.key===' '&&!event.target.closest('button,summary,.variant')){event.preventDefault();preview.paused?preview.play():preview.pause();}
  else if(event.key==='ArrowRight'||event.key==='ArrowLeft'){if(event.target.closest('button'))return;event.preventDefault();seek(preview.currentTime+(event.key==='ArrowRight'?1:-1)*(event.shiftKey?1:1/30));}
  else if((event.key==='Delete'||event.key==='Backspace')&&selectedId){event.preventDefault();removeTarget(selectedId);}
});
canvas.addEventListener('pointerdown',event=>{
  if(!mode){preview.paused?preview.play():preview.pause();return;}
  const point=videoPoint(event);if(!point)return;preview.pause();
  if(mode==='frame'){addFramingCorrection(point);return;}
  const target=mode==='anchor'?targetItems.find(item=>item.id===selectedId):null;
  if(target&&target.kind==='point'){addSelection({time:preview.currentTime,x:point.x,y:point.y,width:0,height:0});return;}
  draft={start:point,end:point};canvas.setPointerCapture(event.pointerId);draw();
});
canvas.addEventListener('pointermove',event=>{
  if(mode==='frame'){hover=videoPoint(event);draw();return;}
  if(!draft)return;const point=videoPoint(event,true);if(!point)return;draft.end=point;draw();
});
canvas.addEventListener('pointerleave',()=>{if(hover){hover=null;draw();}});
canvas.addEventListener('pointerup',event=>{
  if(!draft)return;const point=videoPoint(event,true);if(!point)return;draft.end=point;
  const x=Math.min(draft.start.x,draft.end.x),y=Math.min(draft.start.y,draft.end.y),width=Math.abs(draft.end.x-draft.start.x),height=Math.abs(draft.end.y-draft.start.y),start=draft.start;draft=null;
  const target=mode==='anchor'?targetItems.find(item=>item.id===selectedId):null;
  if(width>.008&&height>.008)addSelection({time:preview.currentTime,x,y,width,height});
  else if(mode==='follow')addSelection({time:preview.currentTime,x:start.x,y:start.y,width:0,height:0});
  else if(target&&target.kind==='point')addSelection({time:preview.currentTime,x:start.x,y:start.y,width:0,height:0});
  else draw();
});

/* run */
function speedFields(data){
  data.set('analysis_proxy',String(analysisProxy.checked));
  data.set('coarse_to_fine',String(coarseToFine.checked));
  data.set('detector_cadence',String(detectorCadence.checked));
  data.set('direct_ffmpeg',String(directFfmpeg.checked));
  data.set('static_scene_reuse',String(staticSceneReuse.checked));
  data.set('persistent_models',String(persistentModels.checked));
  data.set('dnn_optimization',String(dnnOptimization.checked));
  data.set('dnn_batch_size',String(dnnBatchSize.value));
  data.set('pipeline_queues',String(pipelineQueues.checked));
  data.set('consistent_framing',String(consistentFraming.checked));
  data.set('active_speaker',String(activeSpeaker.checked));
}
form.addEventListener('submit',event=>{
  event.preventDefault();if(!syncOutputSize())return;
  if(CLOUD){cloudSubmit();return;}
  if(editingResult&&!needsNewRun()){applyFixes();return;}
  const file=fileInput.files&&fileInput.files[0];
  if(!file){fail('Choose a video file before starting.');drop.hidden=false;return;}
  syncTargets();syncCorrections();
  const data=new FormData(form);
  data.set('video',file,file.name);
  data.delete('shape');
  speedFields(data);
  editingResult=false;review.hidden=true;result.hidden=true;flags=[];renderPins();setBusy(true);showState('Uploading',0);log.textContent='';
  const request=new XMLHttpRequest();request.open('POST','/api/jobs');
  request.upload.onprogress=event=>{if(event.lengthComputable)showState('Uploading',Math.min(.02,event.loaded/event.total*.02));};
  request.onload=()=>{let body={};try{body=JSON.parse(request.responseText||'{}');}catch(error){}if(request.status<200||request.status>=300){fail(formatApiError(body,'Upload failed'));return;}currentJobId=body.id;history.replaceState(null,'','/?job='+body.id);poll(body.id);};
  request.onerror=()=>fail('Upload failed');request.send(data);
});
renderReviewed.addEventListener('click',async()=>{if(!currentJobId)return;syncTargets();setBusy(true);review.hidden=true;showState('Rendering',0.02);const data=new FormData();data.set('targets',JSON.stringify(targetItems));data.set('corrections',JSON.stringify(correctionItems));try{const response=await fetch('/api/jobs/'+currentJobId+'/render',{method:'POST',body:data});const body=await response.json();if(!response.ok)throw new Error(formatApiError(body,'Could not start render'));poll(currentJobId);}catch(error){fail(error.message||String(error));}});
adjustFraming.addEventListener('click',()=>{
  if(!currentJobId&&!(CLOUD&&lastJob))return;
  if(CLOUD&&!(fileInput.files&&fileInput.files[0])&&lastJob&&lastJob.sampleSource){loadSampleFile(lastJob.sampleSource).then(()=>adjustFraming.click());return;}
  const canRetarget=Boolean(fileInput.files&&fileInput.files[0]);
  for(const button of toolButtons)if(button.dataset.mode!=='frame')button.hidden=!canRetarget;
  if(!preview.src&&!CLOUD){preview.src='/source/'+currentJobId;preview.load();}
  const quality=form.querySelector('input[name=quality][value="'+lastJob.quality+'"]');if(quality)quality.checked=true;
  const shape=form.querySelector('input[name=shape][value="'+lastJob.output_size+'"]');
  if(shape)shape.checked=true;else{form.querySelector('input[name=shape][value=custom]').checked=true;const parts=lastJob.output_size.split('x');customWidth.value=parts[0];customHeight.value=parts[1];}
  syncOutputSize();
  for(const input of form.querySelectorAll('input[name=shape],input[name=quality],#customWidth,#customHeight'))input.dataset.locked=canRetarget?'':'true';
  editingResult=true;form.hidden=false;drop.hidden=true;adjustFraming.hidden=true;requestAnimationFrame(resizeCanvas);setBusy(false);refreshApply();
  wrap.scrollIntoView({behavior:'smooth',block:'start'});
});
applyCorrections.addEventListener('click',()=>applyFixes());
async function applyFixes(){
  if(!currentJobId)return;syncCorrections();editingResult=false;
  setBusy(true);result.hidden=true;showState('Replanning from cached analysis',0.05);const data=new FormData();data.set('corrections',JSON.stringify(correctionItems));
  try{const response=await fetch('/api/jobs/'+currentJobId+'/correct',{method:'POST',body:data});const body=await response.json();if(!response.ok)throw new Error(formatApiError(body,'Could not apply framing fixes'));poll(currentJobId);}catch(error){fail(error.message||String(error));}
}
syncCorrections();toolHint.textContent=IDLE_HINT;
const sampleButton=$('sample');
async function loadSampleFile(url){
  const response=await fetch(url);if(!response.ok)throw new Error('The sample clip is not available');
  const file=new File([await response.blob()],'wild-animals.mp4',{type:'video/mp4'});
  const transfer=new DataTransfer();transfer.items.add(file);fileInput.files=transfer.files;
  const keep=lastJob;chooseFile(file);
  if(CLOUD&&keep){lastJob=keep;result.hidden=false;newRun.hidden=false;}
}
fetch('/sample.mp4',{method:'HEAD'}).then(r=>{if(r.ok)$('sampleRow').hidden=false;}).catch(()=>{});
sampleButton.addEventListener('click',async()=>{
  sampleButton.disabled=true;
  try{await loadSampleFile('/sample.mp4');}catch(error){fail(error.message||String(error));}
  finally{sampleButton.disabled=false;}
});
const existing=new URLSearchParams(location.search).get('job');if(existing){currentJobId=existing;drop.hidden=true;newRun.hidden=false;setBusy(true);poll(existing);}
</script>
</body>
</html>"""


def create_app(
    *,
    work_root: str | Path = "web-runs",
    max_workers: int = 1,
    max_upload_bytes: int = 8 * 1024**3,
    keep_days: int = 7,
) -> FastAPI:
    root = Path(work_root).expanduser().resolve()
    store = JobStore(root)
    store.clean_old(keep_days)
    runner = JobRunner(store, max_workers=max_workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner.shutdown()

    app = FastAPI(title="Reframe Lab", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        messages: list[str] = []
        for error in exc.errors():
            location = [str(part) for part in error.get("loc", ()) if part != "body"]
            if location == ["video"] and error.get("type") == "missing":
                messages.append("Choose a video file before starting.")
                continue
            message = str(error.get("msg", "Invalid request"))
            messages.append(f"{'.'.join(location)}: {message}" if location else message)
        detail = "; ".join(dict.fromkeys(messages)) or "The request could not be read."
        return JSONResponse(status_code=422, content={"detail": detail})

    app.state.job_store = store
    app.state.job_runner = runner
    app.state.work_root = root

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(
            _INDEX_HTML,
            headers={"Cache-Control": "no-store"},
        )

    sample_path = (
        Path(__file__).resolve().parents[2]
        / "deploy"
        / "aws"
        / "sample"
        / "wild-animals"
        / "source.mp4"
    )

    @app.api_route("/sample.mp4", methods=["GET", "HEAD"])
    async def sample_video() -> FileResponse:
        # Present in a source checkout. An installed wheel has no sample and the page hides the button.
        if not sample_path.is_file():
            raise HTTPException(status_code=404, detail="No sample clip in this install")
        return FileResponse(sample_path, media_type="video/mp4")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        video: UploadFile = File(...),
        preset: str = Form("all"),
        quality: str = Form("balanced"),
        output_size: str = Form("540x960"),
        targets: str = Form("[]"),
        corrections: str = Form("[]"),
        consistent_framing: bool = Form(True),
        active_speaker: bool = Form(True),
        analysis_proxy: bool = Form(True),
        coarse_to_fine: bool = Form(True),
        detector_cadence: bool = Form(True),
        direct_ffmpeg: bool = Form(True),
        static_scene_reuse: bool = Form(True),
        persistent_models: bool = Form(True),
        dnn_optimization: bool = Form(True),
        dnn_batch_size: int = Form(4),
        pipeline_queues: bool = Form(True),
    ) -> JSONResponse:
        if preset not in RUN_PRESETS:
            raise HTTPException(status_code=400, detail="Unknown run type")
        if quality not in QUALITY_PRESETS:
            raise HTTPException(status_code=400, detail="Unknown analysis quality")
        try:
            normalised_size = _validate_output_size(output_size)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        original_name = video.filename or "video.mp4"
        stored_name = _safe_filename(original_name)
        if Path(stored_name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Choose a supported video file")

        try:
            target_specs = parse_target_specs(targets)
            framing_corrections = parse_framing_corrections(corrections)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_preset = RUN_PRESETS[preset]
        speed_options = {
            "analysis_proxy": bool(analysis_proxy),
            "coarse_to_fine": bool(coarse_to_fine),
            "detector_cadence": bool(detector_cadence),
            "direct_ffmpeg": bool(direct_ffmpeg),
            "static_scene_reuse": bool(static_scene_reuse),
            "persistent_models": bool(persistent_models),
            "dnn_optimization": bool(dnn_optimization),
            "dnn_batch_size": max(1, min(16, int(dnn_batch_size))),
            "pipeline_queues": bool(pipeline_queues),
            "consistent_framing": bool(consistent_framing),
            "active_speaker": bool(active_speaker),
        }
        guided = bool(target_specs)
        profiles = _profiles_for_run(preset, guided)
        record = store.create(
            original_name=original_name,
            stored_name=stored_name,
            preset=preset,
            quality=quality,
            output_size=normalised_size,
            profiles=profiles,
            speed_options=speed_options,
            targets=[target.to_dict() for target in target_specs],
            corrections=[item.to_dict() for item in framing_corrections],
        )
        input_path = store.source_dir(record.id) / stored_name
        size = 0
        try:
            try:
                with input_path.open("wb") as output:
                    while True:
                        chunk = await video.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > max_upload_bytes:
                            raise HTTPException(status_code=413, detail="The upload is too large")
                        output.write(chunk)
            finally:
                await video.close()
            if size == 0:
                raise HTTPException(status_code=400, detail="The uploaded file is empty")

            info = probe_video(input_path)
            timeline_tolerance = max(0.15, 1.5 / max(info.fps, 1.0))
            for target in target_specs:
                for anchor in target.anchors:
                    if anchor.time > info.duration + timeline_tolerance:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Target {target.label!r} has a keyframe at "
                                f"{anchor.time:.3f}s, after the video ends at "
                                f"{info.duration:.3f}s"
                            ),
                        )
            for correction in framing_corrections:
                if correction.time > info.duration + timeline_tolerance:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"A framing correction at {correction.time:.3f}s is after "
                            f"the video ends at {info.duration:.3f}s"
                        ),
                    )
            store.update(
                record.id,
                source={
                    "width": info.width,
                    "height": info.height,
                    "fps": round(info.fps, 4),
                    "duration": round(info.duration, 3),
                    "bytes": size,
                },
            )
            runner.submit_initial(
                record.id,
                input_path,
                profiles=profiles,
                output_size=normalised_size,
                quality=quality,
                create_comparison=bool(run_preset["comparison"]),
                target_specs=target_specs,
                corrections=framing_corrections,
                speed_options=speed_options,
            )
        except HTTPException:
            store.delete(record.id)
            raise
        except Exception as exc:
            store.delete(record.id)
            raise HTTPException(status_code=400, detail=f"Could not read the video: {exc}") from exc

        return JSONResponse(store.public(record.id), status_code=202)

    @app.post("/api/jobs/{job_id}/render", status_code=202)
    async def render_reviewed_job(
        job_id: str,
        targets: str = Form("[]"),
        corrections: str = Form("[]"),
    ) -> JSONResponse:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Run not found")
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if record.status != "review":
            raise HTTPException(status_code=409, detail="This run is not awaiting review")
        try:
            target_specs = parse_target_specs(targets)
            framing_corrections = parse_framing_corrections(corrections)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_info = record.source
        tolerance = max(0.15, 1.5 / max(float(source_info.get("fps", 1.0)), 1.0))
        duration = float(source_info.get("duration", 0.0))
        for target in target_specs:
            for anchor in target.anchors:
                if anchor.time > duration + tolerance:
                    raise HTTPException(
                        status_code=400,
                        detail=f"A correction keyframe at {anchor.time:.3f}s is after the video ends",
                    )
        for correction in framing_corrections:
            if correction.time > duration + tolerance:
                raise HTTPException(
                    status_code=400,
                    detail=f"A framing correction at {correction.time:.3f}s is after the video ends",
                )

        input_path = store.source_dir(job_id) / record.stored_name
        same_targets = [target.to_dict() for target in target_specs] == record.targets
        tracks_path = store.output_dir(job_id) / "tracks.json"
        precomputed = tracks_path if same_targets and tracks_path.is_file() else None
        guided = bool(target_specs)
        profiles = _profiles_for_run(record.preset, guided)
        runner.submit_render(
            job_id,
            input_path,
            profiles=profiles,
            output_size=record.output_size,
            quality=record.quality,
            create_comparison=bool(RUN_PRESETS[record.preset]["comparison"]),
            target_specs=target_specs,
            precomputed_tracks=precomputed,
            framing_corrections=framing_corrections,
            precomputed_bundle=None,
            speed_options=record.speed_options,
        )
        return JSONResponse(store.public(job_id), status_code=202)

    @app.post("/api/jobs/{job_id}/correct", status_code=202)
    async def correct_completed_job(
        job_id: str,
        corrections: str = Form("[]"),
    ) -> JSONResponse:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Run not found")
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if record.status != "completed":
            raise HTTPException(status_code=409, detail="Only a completed run can be corrected")
        try:
            framing_corrections = parse_framing_corrections(corrections)
            target_specs = parse_target_specs(json.dumps(record.targets))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_info = record.source
        tolerance = max(0.15, 1.5 / max(float(source_info.get("fps", 1.0)), 1.0))
        duration = float(source_info.get("duration", 0.0))
        for correction in framing_corrections:
            if correction.time > duration + tolerance:
                raise HTTPException(
                    status_code=400,
                    detail=f"A framing correction at {correction.time:.3f}s is after the video ends",
                )

        output_dir = store.output_dir(job_id)
        bundle_path = output_dir / "analysis_bundle.pkl.gz"
        if not bundle_path.is_file():
            raise HTTPException(
                status_code=409,
                detail="The cached analysis is unavailable. Start a new run before correcting framing.",
            )
        tracks_path = output_dir / "tracks.json"
        precomputed_tracks = tracks_path if tracks_path.is_file() else None
        input_path = store.source_dir(job_id) / record.stored_name
        profiles = _profiles_for_run(record.preset, bool(target_specs))
        runner.submit_render(
            job_id,
            input_path,
            profiles=profiles,
            output_size=record.output_size,
            quality=record.quality,
            create_comparison=bool(RUN_PRESETS[record.preset]["comparison"]),
            target_specs=target_specs,
            precomputed_tracks=precomputed_tracks,
            speed_options=record.speed_options,
            framing_corrections=framing_corrections,
            precomputed_bundle=bundle_path,
        )
        return JSONResponse(store.public(job_id), status_code=202)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Run not found")
        record = store.public(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    @app.get("/source/{job_id}")
    async def source_file(job_id: str) -> FileResponse:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="File not found")
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="File not found")
        target = store.source_dir(job_id) / record.stored_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target, headers={"Accept-Ranges": "bytes"})

    @app.get("/runs/{job_id}/{relative_path:path}")
    async def run_file(job_id: str, relative_path: str) -> FileResponse:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="File not found")
        output_root = store.output_dir(job_id).resolve()
        target = (output_root / relative_path).resolve()
        try:
            target.relative_to(output_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target, headers={"Accept-Ranges": "bytes"})

    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    work_root: str | Path = "web-runs",
    max_workers: int = 1,
    max_upload_gb: float = 8.0,
    keep_days: int = 7,
    open_browser: bool = True,
    log_level: str = "warning",
) -> None:
    import uvicorn

    app = create_app(
        work_root=work_root,
        max_workers=max_workers,
        max_upload_bytes=max(1, int(max_upload_gb * 1024**3)),
        keep_days=keep_days,
    )
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}"
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    print(f"Reframe Lab: {url}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
