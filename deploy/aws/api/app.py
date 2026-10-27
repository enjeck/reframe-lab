from __future__ import annotations

import json
import os
import re
import secrets
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = os.environ["JOB_BUCKET"]
CLUSTER = os.environ["ECS_CLUSTER"]
TASK_DEFINITION = os.environ["TASK_DEFINITION"]
SUBNETS = [value for value in os.environ["SUBNET_IDS"].split(",") if value]
SECURITY_GROUP = os.environ["SECURITY_GROUP_ID"]
ACCESS_CODE = os.environ.get("ACCESS_CODE", "")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(4 * 1024**3)))
LOG_GROUP = os.environ.get("LOG_GROUP", "")
RESULT_URL_SECONDS = int(os.environ.get("RESULT_URL_SECONDS", "3600"))
UPLOAD_URL_SECONDS = int(os.environ.get("UPLOAD_URL_SECONDS", "900"))
RETENTION_HOURS = float(os.environ.get("RETENTION_HOURS", "12"))
# Only visitor data expires. samples/ holds the bundled example clip and is never deleted.
_EXPIRING_PREFIXES = ("uploads/", "jobs/")

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

# Presigned URLs must name the bucket's regional endpoint. The global
# s3.amazonaws.com name answers a new bucket outside us-east-1 with a 307
# redirect, and browsers do not follow redirects on cross-origin uploads.
s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=f"https://s3.{REGION}.amazonaws.com",
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
)
ecs = boto3.client("ecs")

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ogv"}
_QUALITY = {"quick", "balanced", "careful"}
_JOB_RE = re.compile(r"^[a-f0-9]{32}$")
_SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json; charset=utf-8",
            "cache-control": "no-store",
        },
        "body": json.dumps(payload),
    }


def _html_response(body: str) -> dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {
            "content-type": "text/html; charset=utf-8",
            "cache-control": "no-store",
        },
        "body": body,
    }


def _body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body") or "{}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object")
    return value


def _headers(event: dict[str, Any]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}


def _authorized(event: dict[str, Any]) -> bool:
    if not ACCESS_CODE:
        return True
    supplied = _headers(event).get("x-reframe-key", "")
    return secrets.compare_digest(supplied, ACCESS_CODE)


def _safe_filename(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned:
        cleaned = "video.mp4"
    suffix = PurePosixPath(cleaned).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise ValueError("Choose an MP4, MOV, MKV, WebM, AVI, M4V, or OGV video")
    stem = PurePosixPath(cleaned).stem[:96] or "video"
    return f"{stem}{suffix}"


def _validate_size(value: str) -> str:
    match = _SIZE_RE.fullmatch(value.strip().lower())
    if not match:
        raise ValueError("Output size must look like 540x960")
    width, height = int(match.group(1)), int(match.group(2))
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise ValueError("Output width and height must be between 64 and 4096")
    if width * height > 8_388_608:
        raise ValueError("Output may not exceed 8.3 megapixels")
    ratio = width / height
    if not (0.125 <= ratio <= 8.0):
        raise ValueError("Output aspect ratio must be between 1:8 and 8:1")
    return f"{width}x{height}"


def _put_json(key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def _get_json(key: str) -> dict[str, Any] | None:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    value = json.loads(response["Body"].read())
    return value if isinstance(value, dict) else None


def _s3_key(uri: str) -> str | None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != BUCKET:
        return None
    return parsed.path.lstrip("/")


def _signed_get(key: str) -> str:
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=RESULT_URL_SECONDS,
    )


def _result_links(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        return {}
    relative_keys: dict[str, str] = {}
    for relative, uri in files.items():
        if isinstance(relative, str) and isinstance(uri, str):
            key = _s3_key(uri)
            if key:
                relative_keys[relative] = key

    preferred = {
        "video": "renders/hybrid_safe.mp4",
        "comparison": "comparison.mp4",
        "results_zip": "results.zip",
        "performance": "performance.json",
        "auto_fix_trace": "auto_fix_trace.json",
    }
    links: dict[str, Any] = {}
    for label, relative in preferred.items():
        key = relative_keys.get(relative)
        if key:
            links[label] = _signed_get(key)
    variants = []
    for relative in sorted(relative_keys, key=_variant_order):
        if relative.startswith("renders/") and relative.endswith(".mp4"):
            profile = PurePosixPath(relative).stem
            variants.append({"profile": profile, "url": _signed_get(relative_keys[relative])})
    links["videos"] = variants
    source_key = _s3_key(str(manifest.get("input") or ""))
    if source_key:
        links["source"] = _signed_get(source_key)
    return links


_VARIANT_ORDER = ["target_guided_safe", "target_guided", "hybrid_safe", "hybrid_temporal", "center"]
_MAX_MARKS_BYTES = 64 * 1024


def _marks(payload: dict[str, Any]) -> dict[str, list[Any]] | None:
    """Light shape check. The worker validates every field with the local app's parsers."""
    marks = payload.get("marks")
    if marks in (None, {}):
        return None
    if not isinstance(marks, dict):
        raise ValueError("Marks must be an object with targets and corrections")
    targets = marks.get("targets") or []
    corrections = marks.get("corrections") or []
    if not isinstance(targets, list) or not isinstance(corrections, list):
        raise ValueError("Targets and corrections must be lists")
    if len(targets) > 16 or len(corrections) > 64:
        raise ValueError("Use at most 16 targets and 64 framing marks")
    if not targets and not corrections:
        return None
    clean = {"targets": targets, "corrections": corrections}
    if len(json.dumps(clean)) > _MAX_MARKS_BYTES:
        raise ValueError("The marks are too large")
    return clean


def _variant_order(relative: str) -> tuple[int, str]:
    stem = PurePosixPath(relative).stem
    rank = _VARIANT_ORDER.index(stem) if stem in _VARIANT_ORDER else len(_VARIANT_ORDER)
    return rank, relative


SAMPLE_PREFIX = "samples/wild-animals"
_SAMPLE_FILES = {
    "auto_fix_trace": "auto_fix_trace.json",
    "performance": "performance.json",
}


def _sample() -> dict[str, Any]:
    """Pre-rendered edits of the bundled sample clip, so a visitor sees results at once."""
    manifest = _get_json(f"{SAMPLE_PREFIX}/sample.json")
    if not manifest:
        raise LookupError("The sample clip has not been uploaded to this deployment")
    profiles = [str(name) for name in manifest.get("profiles") or [] if re.fullmatch(r"[a-z_]+", str(name))]
    links: dict[str, Any] = {
        "videos": [
            {"profile": name, "url": _signed_get(f"{SAMPLE_PREFIX}/{name}.mp4")} for name in profiles
        ],
        "source": _signed_get(f"{SAMPLE_PREFIX}/source.mp4"),
    }
    for label, name in _SAMPLE_FILES.items():
        links[label] = _signed_get(f"{SAMPLE_PREFIX}/{name}")
    return {
        "status": "completed",
        "sample": True,
        "title": str(manifest.get("title") or "Sample"),
        "output_size": _validate_size(str(manifest.get("output_size") or "540x960")),
        "links": links,
    }


def _upload_ticket(payload: dict[str, Any]) -> dict[str, Any]:
    filename = _safe_filename(str(payload.get("filename") or ""))
    declared_size = int(payload.get("size") or 0)
    if declared_size <= 0:
        raise ValueError("The selected video is empty")
    if declared_size > MAX_UPLOAD_BYTES:
        raise ValueError(f"The video exceeds the {MAX_UPLOAD_BYTES // 1024**3} GB upload limit")

    job_id = secrets.token_hex(16)
    key = f"uploads/{job_id}/{filename}"
    upload = s3.generate_presigned_post(
        Bucket=BUCKET,
        Key=key,
        Conditions=[["content-length-range", 1, MAX_UPLOAD_BYTES]],
        ExpiresIn=UPLOAD_URL_SECONDS,
    )
    return {
        "job_id": job_id,
        "input_key": key,
        "upload": upload,
        "expires_in": UPLOAD_URL_SECONDS,
    }


def _start_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(payload.get("job_id") or "")
    input_key = str(payload.get("input_key") or "")
    if not _JOB_RE.fullmatch(job_id):
        raise ValueError("Invalid job identifier")
    if not input_key.startswith(f"uploads/{job_id}/"):
        raise ValueError("The uploaded file does not belong to this job")

    try:
        source = s3.head_object(Bucket=BUCKET, Key=input_key)
    except ClientError as exc:
        raise ValueError("The video upload has not completed") from exc
    if int(source.get("ContentLength") or 0) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded video exceeds the configured size limit")

    output_size = _validate_size(str(payload.get("output_size") or "540x960"))
    quality = str(payload.get("quality") or "balanced")
    if quality not in _QUALITY:
        raise ValueError("Quality must be quick, balanced, or careful")
    compare = bool(payload.get("compare", False))
    marks = _marks(payload)

    input_uri = f"s3://{BUCKET}/{input_key}"
    output_prefix = f"jobs/{job_id}"
    output_uri = f"s3://{BUCKET}/{output_prefix}"
    command = [
        "cloud-job",
        "--input",
        input_uri,
        "--output",
        output_uri,
        "--output-size",
        output_size,
        "--quality",
        quality,
        "--auto-fix-rounds",
        "2",
    ]
    if compare:
        command.append("--compare")
    if marks:
        marks_key = f"uploads/{job_id}/marks.json"
        _put_json(marks_key, marks)
        command.extend(["--marks", f"s3://{BUCKET}/{marks_key}"])

    response = ecs.run_task(
        cluster=CLUSTER,
        taskDefinition=TASK_DEFINITION,
        launchType="FARGATE",
        platformVersion="LATEST",
        count=1,
        startedBy=f"reframe-{job_id[:20]}",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": [SECURITY_GROUP],
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": "worker",
                    "command": command,
                    "environment": [
                        {"name": "REFRAME_JOB_ID", "value": job_id},
                    ],
                }
            ]
        },
    )
    failures = response.get("failures") or []
    tasks = response.get("tasks") or []
    if failures or not tasks:
        reason = "; ".join(str(item.get("reason") or item) for item in failures)
        raise RuntimeError(reason or "ECS did not start the worker task")

    task_arn = str(tasks[0]["taskArn"])
    status = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "running",
        "submitted_at": _now(),
        "task_arn": task_arn,
        "input": input_uri,
        "output_prefix": output_uri,
        "output_size": output_size,
        "quality": quality,
        "compare": compare,
        "marks": {"targets": len(marks["targets"]), "corrections": len(marks["corrections"])}
        if marks
        else None,
    }
    _put_json(f"{output_prefix}/status.json", status)
    return status


def _job_status(job_id: str) -> dict[str, Any]:
    if not _JOB_RE.fullmatch(job_id):
        raise ValueError("Invalid job identifier")
    prefix = f"jobs/{job_id}"
    completed = _get_json(f"{prefix}/cloud_job.json")
    if completed:
        return {
            "job_id": job_id,
            "status": "completed",
            "elapsed_seconds": completed.get("elapsed_seconds"),
            "links": _result_links(completed),
            "runtime": completed.get("runtime"),
        }

    failed = _get_json(f"{prefix}/error.json")
    if failed:
        return {
            "job_id": job_id,
            "status": "failed",
            "message": failed.get("message") or "The worker failed",
            "error_type": failed.get("error_type"),
        }

    status = _get_json(f"{prefix}/status.json")
    if not status:
        # _start_job writes status.json before it returns, so a missing file
        # means the run was deleted by the retention cleanup (or never existed).
        return {
            "job_id": job_id,
            "status": "failed",
            "message": f"This run has expired. Uploads and results are deleted after {RETENTION_HOURS:g} hours.",
            "error_type": "Expired",
        }

    task_arn = status.get("task_arn")
    if task_arn:
        description = ecs.describe_tasks(cluster=CLUSTER, tasks=[task_arn])
        tasks = description.get("tasks") or []
        if tasks:
            task = tasks[0]
            last_status = str(task.get("lastStatus") or "UNKNOWN")
            if last_status == "STOPPED":
                containers = task.get("containers") or []
                details = "; ".join(
                    str(item.get("reason") or item.get("exitCode") or "container stopped")
                    for item in containers
                )
                message = str(task.get("stoppedReason") or details or "ECS worker stopped")
                payload = {
                    "schema_version": 1,
                    "job_id": job_id,
                    "status": "failed",
                    "message": message,
                    "error_type": "EcsTaskStopped",
                    "failed_at": _now(),
                }
                _put_json(f"{prefix}/error.json", payload)
                return payload
            status["ecs_status"] = last_status
    return status


def _cleanup(now: datetime | None = None) -> dict[str, Any]:
    """Delete uploads and results older than RETENTION_HOURS. Runs hourly from EventBridge."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=RETENTION_HOURS)
    deleted = 0
    for prefix in _EXPIRING_PREFIXES:
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            old = [
                {"Key": item["Key"]}
                for item in page.get("Contents") or []
                if item["Key"].startswith(_EXPIRING_PREFIXES) and item["LastModified"] < cutoff
            ]
            for start in range(0, len(old), 1000):
                s3.delete_objects(Bucket=BUCKET, Delete={"Objects": old[start : start + 1000], "Quiet": True})
            deleted += len(old)
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
    summary = {"event": "cleanup", "deleted": deleted, "cutoff": cutoff.isoformat()}
    print(json.dumps(summary))
    return summary


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context
    if event.get("source") == "aws.events" or event.get("detail-type") == "Scheduled Event":
        return _cleanup()
    method = str(event.get("requestContext", {}).get("http", {}).get("method") or "GET")
    path = str(event.get("rawPath") or "/")

    if path == "/" and method == "GET":
        html = (os.path.join(os.path.dirname(__file__), "index.html"))
        with open(html, encoding="utf-8") as handle:
            return _html_response(handle.read())

    if path == "/api/sample" and method == "GET":
        # Public on purpose: it only signs reads of the pre-rendered sample files.
        try:
            return _json_response(200, _sample())
        except LookupError as exc:
            return _json_response(404, {"error": str(exc)})

    if path.startswith("/api/") and not _authorized(event):
        return _json_response(401, {"error": "The access code is incorrect"})

    try:
        if path == "/api/upload" and method == "POST":
            return _json_response(200, _upload_ticket(_body(event)))
        if path == "/api/jobs" and method == "POST":
            return _json_response(202, _start_job(_body(event)))
        if path.startswith("/api/jobs/") and method == "GET":
            return _json_response(200, _job_status(path.rsplit("/", 1)[-1]))
        if path == "/api/health" and method == "GET":
            return _json_response(200, {"status": "ok", "log_group": LOG_GROUP})
        return _json_response(404, {"error": "Not found"})
    except ValueError as exc:
        return _json_response(400, {"error": str(exc)})
    except Exception as exc:
        print(json.dumps({"event": "api_error", "path": path, "error": repr(exc)}))
        traceback.print_exc()
        return _json_response(500, {"error": f"The AWS job could not be started: {_reason(exc)}"})


def _reason(exc: Exception) -> str:
    """A short, safe reason for an authorised user. Never includes credentials."""
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = str(error.get("Code") or type(exc).__name__)
        message = str(error.get("Message") or "")
        return f"{code}: {message}"[:300]
    return f"{type(exc).__name__}: {exc}"[:300]
