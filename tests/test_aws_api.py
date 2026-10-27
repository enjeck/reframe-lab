from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def generate_presigned_post(self, **kwargs):
        return {"url": "https://uploads.example", "fields": {"key": kwargs["Key"]}}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        assert operation == "get_object"
        assert ExpiresIn == 3600
        return f"https://download.example/{Params['Key']}"

    def put_object(self, Bucket, Key, Body, **kwargs):
        assert Bucket == "job-bucket"
        self.objects[Key] = bytes(Body)

    def get_object(self, Bucket, Key):
        assert Bucket == "job-bucket"
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        assert Bucket == "job-bucket"
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key])}


class FakeEcs:
    def __init__(self) -> None:
        self.last_run = None
        self.status = "RUNNING"

    def run_task(self, **kwargs):
        self.last_run = kwargs
        return {"tasks": [{"taskArn": "arn:aws:ecs:region:account:task/demo"}], "failures": []}

    def describe_tasks(self, **kwargs):
        task = {"taskArn": kwargs["tasks"][0], "lastStatus": self.status}
        if self.status == "STOPPED":
            task.update({"stoppedReason": "Essential container exited", "containers": [{"exitCode": 2}]})
        return {"tasks": [task]}


def load_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_s3 = FakeS3()
    fake_ecs = FakeEcs()
    monkeypatch.setenv("JOB_BUCKET", "job-bucket")
    monkeypatch.setenv("ECS_CLUSTER", "cluster")
    monkeypatch.setenv("TASK_DEFINITION", "task-def")
    monkeypatch.setenv("SUBNET_IDS", "subnet-a,subnet-b")
    monkeypatch.setenv("SECURITY_GROUP_ID", "sg-1")
    monkeypatch.setenv("ACCESS_CODE", "secret-code")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10485760")
    monkeypatch.setenv("LOG_GROUP", "/reframe/test")

    import boto3

    monkeypatch.setattr(boto3, "client", lambda name, **_: fake_s3 if name == "s3" else fake_ecs)
    path = (
        Path(__file__).parents[1]
        / "deploy"
        / "aws"
        / "api"
        / "app.py"
    )
    name = f"aws_api_{os.urandom(4).hex()}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, fake_s3, fake_ecs


def event(method: str, path: str, body: dict | None = None, code: str = "secret-code"):
    return {
        "rawPath": path,
        "headers": {"x-reframe-key": code},
        "requestContext": {"http": {"method": method}},
        "body": None if body is None else json.dumps(body),
    }


def response_json(response):
    return json.loads(response["body"])


def test_upload_and_start_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api, s3, ecs = load_api(monkeypatch, tmp_path)
    upload_response = api.handler(
        event("POST", "/api/upload", {"filename": "demo.mp4", "size": 1000}), None
    )
    assert upload_response["statusCode"] == 200
    ticket = response_json(upload_response)
    s3.objects[ticket["input_key"]] = b"video"

    start_response = api.handler(
        event(
            "POST",
            "/api/jobs",
            {
                "job_id": ticket["job_id"],
                "input_key": ticket["input_key"],
                "output_size": "540x960",
                "quality": "balanced",
                "compare": True,
            },
        ),
        None,
    )
    assert start_response["statusCode"] == 202
    assert ecs.last_run["launchType"] == "FARGATE"
    command = ecs.last_run["overrides"]["containerOverrides"][0]["command"]
    assert command[0] == "cloud-job"
    assert "--compare" in command
    assert f"jobs/{ticket['job_id']}/status.json" in s3.objects


def test_api_rejects_wrong_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api, _, _ = load_api(monkeypatch, tmp_path)
    result = api.handler(event("GET", "/api/health", code="wrong"), None)
    assert result["statusCode"] == 401
    assert "access code" in response_json(result)["error"].lower()


def test_completed_status_returns_signed_links(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api, s3, _ = load_api(monkeypatch, tmp_path)
    job_id = "a" * 32
    manifest = {
        "elapsed_seconds": 12.3,
        "runtime": {"opencv_version": "5.0.0"},
        "files": {
            "renders/center.mp4": f"s3://job-bucket/jobs/{job_id}/renders/center.mp4",
            "renders/hybrid_safe.mp4": f"s3://job-bucket/jobs/{job_id}/renders/hybrid_safe.mp4",
            "renders/hybrid_temporal.mp4": (
                f"s3://job-bucket/jobs/{job_id}/renders/hybrid_temporal.mp4"
            ),
            "comparison.mp4": f"s3://job-bucket/jobs/{job_id}/comparison.mp4",
            "results.zip": f"s3://job-bucket/jobs/{job_id}/results.zip",
        },
    }
    s3.objects[f"jobs/{job_id}/cloud_job.json"] = json.dumps(manifest).encode()
    result = api.handler(event("GET", f"/api/jobs/{job_id}"), None)
    payload = response_json(result)
    assert payload["status"] == "completed"
    assert payload["links"]["video"].endswith("renders/hybrid_safe.mp4")
    assert payload["links"]["results_zip"].endswith("results.zip")
    variants = payload["links"]["videos"]
    assert [item["profile"] for item in variants] == ["hybrid_safe", "hybrid_temporal", "center"]
    assert variants[2]["url"].endswith("renders/center.mp4")


def test_stopped_task_becomes_readable_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api, s3, ecs = load_api(monkeypatch, tmp_path)
    job_id = "b" * 32
    s3.objects[f"jobs/{job_id}/status.json"] = json.dumps(
        {"job_id": job_id, "status": "running", "task_arn": "task-arn"}
    ).encode()
    ecs.status = "STOPPED"
    result = api.handler(event("GET", f"/api/jobs/{job_id}"), None)
    payload = response_json(result)
    assert payload["status"] == "failed"
    assert "container" in payload["message"].lower()
    assert f"jobs/{job_id}/error.json" in s3.objects


def test_root_serves_cloud_ui_and_extreme_aspect_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, s3, _ = load_api(monkeypatch, tmp_path)
    root = api.handler(event("GET", "/", code="wrong"), None)
    assert root["statusCode"] == 200
    assert ">Reframe</button>" in root["body"]

    upload = response_json(
        api.handler(event("POST", "/api/upload", {"filename": "demo.mp4", "size": 1000}), None)
    )
    s3.objects[upload["input_key"]] = b"video"
    result = api.handler(
        event(
            "POST",
            "/api/jobs",
            {
                "job_id": upload["job_id"],
                "input_key": upload["input_key"],
                "output_size": "4096x64",
                "quality": "quick",
            },
        ),
        None,
    )
    assert result["statusCode"] == 400
    assert "aspect ratio" in response_json(result)["error"]


def test_failed_task_start_reports_the_aws_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from botocore.exceptions import ClientError

    api, s3, ecs = load_api(monkeypatch, tmp_path)

    def refuse(**kwargs):
        raise ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "not allowed to RunTask"}},
            "RunTask",
        )

    ecs.run_task = refuse
    ticket = response_json(
        api.handler(event("POST", "/api/upload", {"filename": "demo.mp4", "size": 1000}), None)
    )
    s3.objects[ticket["input_key"]] = b"video"
    result = api.handler(
        event("POST", "/api/jobs", {"job_id": ticket["job_id"], "input_key": ticket["input_key"]}),
        None,
    )
    assert result["statusCode"] == 500
    error = response_json(result)["error"]
    assert error.startswith("The AWS job could not be started: AccessDeniedException")
    assert "RunTask" in error


def test_sample_returns_prerendered_edits_without_access_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, s3, _ = load_api(monkeypatch, tmp_path)
    missing = api.handler(event("GET", "/api/sample", code="wrong"), None)
    assert missing["statusCode"] == 404

    s3.objects["samples/wild-animals/sample.json"] = json.dumps(
        {"title": "Wild animals", "output_size": "540x960", "profiles": ["hybrid_safe", "center", "../x"]}
    ).encode()
    result = api.handler(event("GET", "/api/sample", code="wrong"), None)
    assert result["statusCode"] == 200
    payload = response_json(result)
    assert payload["status"] == "completed"
    assert payload["output_size"] == "540x960"
    assert [item["profile"] for item in payload["links"]["videos"]] == ["hybrid_safe", "center"]
    assert payload["links"]["videos"][0]["url"].endswith("samples/wild-animals/hybrid_safe.mp4")
    assert payload["links"]["source"].endswith("samples/wild-animals/source.mp4")


def test_marks_are_stored_and_passed_to_the_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, s3, ecs = load_api(monkeypatch, tmp_path)
    ticket = response_json(
        api.handler(event("POST", "/api/upload", {"filename": "demo.mp4", "size": 1000}), None)
    )
    s3.objects[ticket["input_key"]] = b"video"
    marks = {
        "targets": [
            {
                "id": "t1",
                "label": "Follow 1",
                "kind": "box",
                "role": "follow",
                "anchors": [{"time": 1.0, "x": 0.2, "y": 0.2, "width": 0.3, "height": 0.5}],
            }
        ],
        "corrections": [],
    }
    result = api.handler(
        event(
            "POST",
            "/api/jobs",
            {"job_id": ticket["job_id"], "input_key": ticket["input_key"], "marks": marks},
        ),
        None,
    )
    assert result["statusCode"] == 202
    command = ecs.last_run["overrides"]["containerOverrides"][0]["command"]
    key = f"uploads/{ticket['job_id']}/marks.json"
    assert command[command.index("--marks") + 1] == f"s3://job-bucket/{key}"
    assert json.loads(s3.objects[key])["targets"][0]["label"] == "Follow 1"

    bad = api.handler(
        event(
            "POST",
            "/api/jobs",
            {"job_id": ticket["job_id"], "input_key": ticket["input_key"], "marks": {"targets": "x"}},
        ),
        None,
    )
    assert bad["statusCode"] == 400


def test_hourly_cleanup_deletes_old_visitor_files_but_never_the_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import datetime, timedelta, timezone

    api, s3, _ = load_api(monkeypatch, tmp_path)
    now = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    ages = {
        "uploads/a/old.mp4": 13,
        "uploads/b/new.mp4": 2,
        "jobs/a/renders/center.mp4": 12.5,
        "jobs/b/status.json": 1,
        "samples/wild-animals/source.mp4": 500,
    }
    deleted: list[str] = []

    def list_objects_v2(Bucket, Prefix, ContinuationToken=None):
        assert Bucket == "job-bucket"
        return {
            "Contents": [
                {"Key": key, "LastModified": now - timedelta(hours=hours)}
                for key, hours in ages.items()
                if key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def delete_objects(Bucket, Delete):
        deleted.extend(item["Key"] for item in Delete["Objects"])

    s3.list_objects_v2 = list_objects_v2
    s3.delete_objects = delete_objects
    summary = api._cleanup(now=now)
    assert sorted(deleted) == ["jobs/a/renders/center.mp4", "uploads/a/old.mp4"]
    assert summary["deleted"] == 2

    scheduled = api.handler({"source": "aws.events", "detail-type": "Scheduled Event"}, None)
    assert scheduled["event"] == "cleanup"


def test_deleted_run_reports_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api, _, _ = load_api(monkeypatch, tmp_path)
    result = response_json(api.handler(event("GET", f"/api/jobs/{'c' * 32}"), None))
    assert result["status"] == "failed"
    assert "expired" in result["message"]
