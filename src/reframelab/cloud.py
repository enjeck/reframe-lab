from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZipFile

from .corrections import parse_framing_corrections
from .pipeline import run_demo
from .profiles import TARGET_VARIANT_PROFILES, VARIANT_PROFILES
from .runtime import opencv_runtime_info
from .targets import parse_target_specs

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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Expected an S3 URI such as s3://bucket/key, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _archive_directory(root: Path, target: Path) -> None:
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.resolve() != target.resolve():
                archive.write(path, path.relative_to(root))


def run_s3_job(
    *,
    input_uri: str,
    output_uri: str,
    output_size: str = "540x960",
    quality: str = "balanced",
    compare: bool = False,
    auto_fix_rounds: int = 2,
    marks_uri: str | None = None,
) -> dict[str, object]:
    """Run one batch reframing job using S3 for input and output.

    The worker writes either ``cloud_job.json`` or ``error.json`` before it exits.
    This makes a stopped ECS task diagnosable without scraping human-readable logs.
    """

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on optional cloud install
        raise RuntimeError("Install the cloud extra: pip install 'reframe-lab[cloud]'") from exc

    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality {quality!r}; choose from {sorted(QUALITY_PRESETS)}")
    input_bucket, input_key = parse_s3_uri(input_uri)
    output_bucket, output_prefix = parse_s3_uri(output_uri)
    output_prefix = output_prefix.rstrip("/")
    job_id = os.environ.get("REFRAME_JOB_ID", uuid.uuid4().hex)
    settings = QUALITY_PRESETS[quality]
    s3 = boto3.client("s3")
    started = time.time()
    runtime = opencv_runtime_info()
    _log(
        "job_started",
        job_id=job_id,
        input=input_uri,
        output=output_uri,
        output_size=output_size,
        quality=quality,
        runtime=runtime,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="reframe-cloud-") as temporary:
            root = Path(temporary)
            source_name = Path(input_key).name or "input.mp4"
            source_path = root / source_name
            result_dir = root / "result"
            _log("download_started", job_id=job_id)
            s3.download_file(input_bucket, input_key, str(source_path))
            _log("download_completed", job_id=job_id, bytes=source_path.stat().st_size)

            def progress(message: str) -> None:
                _log("pipeline_progress", job_id=job_id, message=message)

            # Optional marks from the web editor: things to follow or keep in view,
            # and framing fixes. They are validated by the same parsers as the local app.
            targets = []
            corrections = []
            if marks_uri:
                marks_bucket, marks_key = parse_s3_uri(marks_uri)
                marks_path = root / "marks.json"
                s3.download_file(marks_bucket, marks_key, str(marks_path))
                marks = json.loads(marks_path.read_text(encoding="utf-8"))
                targets = parse_target_specs(json.dumps(marks.get("targets") or []))
                corrections = parse_framing_corrections(json.dumps(marks.get("corrections") or []))
                _log("marks_loaded", job_id=job_id, targets=len(targets), corrections=len(corrections))

            profiles = list(TARGET_VARIANT_PROFILES if targets else VARIANT_PROFILES)
            result = run_demo(
                source_path,
                result_dir,
                profiles,
                output_size=output_size,
                analysis_height=int(settings["analysis_height"]),
                analysis_fps=float(settings["analysis_fps"]),
                candidate_step=int(settings["candidate_step"]),
                adaptive_dense_flow=bool(settings["adaptive_dense_flow"]),
                create_comparison=compare,
                auto_fix=True,
                max_auto_fix_rounds=max(0, auto_fix_rounds),
                target_specs=targets,
                framing_corrections=corrections,
                progress=progress,
            )
            archive_path = result_dir / "results.zip"
            _archive_directory(result_dir, archive_path)

            uploaded: dict[str, str] = {}
            for path in sorted(result_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(result_dir).as_posix()
                key = f"{output_prefix}/{relative}"
                s3.upload_file(str(path), output_bucket, key)
                uploaded[relative] = f"s3://{output_bucket}/{key}"

            elapsed = time.time() - started
            manifest = {
                "schema_version": 1,
                "job_id": job_id,
                "status": "completed",
                "input": input_uri,
                "output_prefix": output_uri,
                "output_size": output_size,
                "quality": quality,
                "profiles": profiles,
                "auto_fix_rounds": max(0, auto_fix_rounds),
                "elapsed_seconds": elapsed,
                "runtime": runtime,
                "files": uploaded,
                "report": uploaded.get(result.report_path.relative_to(result_dir).as_posix()),
                "agent_trace": (
                    uploaded.get(result.agent_trace_path.relative_to(result_dir).as_posix())
                    if result.agent_trace_path is not None
                    else None
                ),
            }
            manifest_path = root / "cloud_job.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifest_key = f"{output_prefix}/cloud_job.json"
            s3.upload_file(str(manifest_path), output_bucket, manifest_key)
            manifest["manifest"] = f"s3://{output_bucket}/{manifest_key}"
            _log("job_completed", **manifest)
            return manifest
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "job_id": job_id,
            "status": "failed",
            "input": input_uri,
            "output_prefix": output_uri,
            "failed_at_epoch": time.time(),
            "elapsed_seconds": time.time() - started,
            "error_type": type(exc).__name__,
            "message": str(exc) or repr(exc),
            "runtime": runtime,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="reframe-cloud-error-") as temporary:
                failure_path = Path(temporary) / "error.json"
                failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
                s3.upload_file(
                    str(failure_path),
                    output_bucket,
                    f"{output_prefix}/error.json",
                )
        except Exception as upload_exc:  # pragma: no cover - only when AWS itself is unavailable
            _log(
                "failure_manifest_upload_failed",
                job_id=job_id,
                error=repr(upload_exc),
            )
        _log("job_failed", **failure)
        raise


def run_s3_benchmark(
    *,
    output_uri: str,
    input_uri: str | None = None,
    frame_count: int = 12,
    iterations: int = 8,
) -> dict[str, object]:
    """Run the reproducible OpenCV kernel benchmark and upload its JSON result."""

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the cloud extra: pip install 'reframe-lab[cloud]'") from exc
    from .benchmark import benchmark_opencv

    output_bucket, output_key = parse_s3_uri(output_uri)
    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory(prefix="reframe-benchmark-") as temporary:
        root = Path(temporary)
        input_path: Path | None = None
        if input_uri:
            input_bucket, input_key = parse_s3_uri(input_uri)
            input_path = root / (Path(input_key).name or "input.mp4")
            s3.download_file(input_bucket, input_key, str(input_path))
        result_path = root / "benchmark.json"
        payload = benchmark_opencv(
            input_path=input_path,
            output_path=result_path,
            frame_count=max(3, frame_count),
            iterations=max(1, iterations),
        )
        s3.upload_file(str(result_path), output_bucket, output_key)
        _log("benchmark_completed", output=output_uri, runtime=payload["runtime"])
        return {**payload, "output": output_uri}
