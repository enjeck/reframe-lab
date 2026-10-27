from __future__ import annotations

import io
import shutil
import time
from dataclasses import FrozenInstanceError
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from reframelab.synthetic import generate_sample
from reframelab.webapp import create_app


def test_web_index_and_health(tmp_path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Reframe Lab" in response.text
        assert 'type="file"' in response.text
        assert '<input name="preset" type="hidden" value="all">' in response.text
        assert 'name="quality" value="balanced" checked' in response.text
        assert response.text.count('type="file"') == 1
        assert 'id="analysisProxy"' in response.text
        assert 'id="coarseToFine"' in response.text
        assert 'id="detectorCadence"' in response.text
        assert 'id="directFfmpeg"' in response.text

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}


def test_web_missing_video_returns_readable_error(tmp_path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={
                "preset": "best",
                "quality": "balanced",
                "output_size": "540x960",
            },
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "Choose a video file before starting."}


def test_web_form_captures_video_before_disabling_controls(tmp_path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        html = client.get("/").text
    submit_start = html.index("form.addEventListener('submit'")
    submit_end = html.index("renderReviewed.addEventListener", submit_start)
    submit = html[submit_start:submit_end]
    assert submit.index("const data=new FormData(form)") < submit.index("setBusy(true)")
    assert "data.set('video',file,file.name)" in submit
    assert "formatApiError(body,'Upload failed')" in submit



@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_web_upload_runs_pipeline_and_serves_results(tmp_path) -> None:
    source, _ = generate_sample(
        tmp_path / "sample.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.4,
    )
    app = create_app(work_root=tmp_path / "runs", keep_days=0, max_workers=1)

    with TestClient(app) as client, source.open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("sample.mp4", video, "video/mp4")},
            data={
                "preset": "fast_compare",
                "quality": "quick",
                "output_size": "180x320",
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]

        deadline = time.monotonic() + 90.0
        job = response.json()
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job_response = client.get(f"/api/jobs/{job_id}")
            assert job_response.status_code == 200
            job = job_response.json()

        assert job["status"] == "completed", job
        assert job["progress"] == pytest.approx(1.0)
        assert job["results"]["performance_url"]
        assert len(job["results"]["outputs"]) == 2

        comparison = client.get(job["results"]["comparison_url"])
        assert comparison.status_code == 200
        assert len(comparison.content) > 1000

        performance = client.get(job["results"]["performance_url"])
        assert performance.status_code == 200
        assert performance.json()["settings"]["analysis_proxy"] is True

        report = client.get(job["results"]["report_url"])
        assert report.status_code == 200
        assert "Reframe Lab comparison" in report.text

        archive = client.get(job["results"]["download_url"])
        assert archive.status_code == 200
        with ZipFile(io.BytesIO(archive.content)) as bundle:
            names = set(bundle.namelist())
        assert "report.html" in names
        assert "comparison.mp4" in names
        assert "results.zip" not in names


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_web_upload_survives_face_detector_failure(tmp_path, monkeypatch) -> None:
    """A broken optional face path must not cancel the user's whole upload."""

    from reframelab.detector_pool import DetectorBundle

    source, _ = generate_sample(
        tmp_path / "face-failure.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.35,
    )

    class ExplodingPersonDetector:
        backend = "broken_face_test"

        def detect_batch(self, frames):
            del frames
            raise FrozenInstanceError("cannot assign to field 'face_x'")

        def runtime_metadata(self):
            return {"backend": self.backend, "face_failures": 1}

    import reframelab.analyzer as analyzer_module

    monkeypatch.setattr(
        analyzer_module,
        "get_detector_bundle",
        lambda **kwargs: DetectorBundle(
            person=ExplodingPersonDetector(),
            text=None,
            metadata={"cache_hit": False},
        ),
    )

    app = create_app(work_root=tmp_path / "runs", keep_days=0, max_workers=1)
    with TestClient(app) as client, source.open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("face-failure.mp4", video, "video/mp4")},
            data={
                "preset": "fast_compare",
                "quality": "quick",
                "output_size": "180x320",
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]

        deadline = time.monotonic() + 90.0
        job = response.json()
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job_id}").json()

        assert job["status"] == "completed", job
        assert len(job["results"]["outputs"]) == 2
        performance = client.get(job["results"]["performance_url"]).json()
        passes = performance["analysis"]["passes"]
        assert sum(item["counts"]["people_detector_errors"] for item in passes) >= 1
        assert any("Person and face detection failed" in line for line in job["logs"])


def test_web_rejects_non_video_extension(tmp_path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"video": ("notes.txt", b"not a video", "text/plain")},
            data={"preset": "compare", "quality": "quick", "output_size": "360x640"},
        )
    assert response.status_code == 400
    assert "supported video" in response.json()["detail"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_web_upload_with_selected_target_returns_track_outputs(tmp_path) -> None:
    import json

    source, _ = generate_sample(
        tmp_path / "target-sample.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.5,
    )
    targets = [
        {
            "id": "product",
            "label": "Product",
            "kind": "point",
            "role": "follow",
            "anchors": [{"time": 0.25, "x": 0.50, "y": 0.55}],
        }
    ]
    app = create_app(work_root=tmp_path / "runs", keep_days=0, max_workers=1)

    with TestClient(app) as client, source.open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("target-sample.mp4", video, "video/mp4")},
            data={
                "preset": "fast_compare",
                "quality": "quick",
                "output_size": "180x320",
                "targets": json.dumps(targets),
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]

        deadline = time.monotonic() + 120.0
        job = response.json()
        while job["status"] not in {"review", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job_id}").json()

        assert job["status"] == "review", job
        assert job["profiles"] == ["center", "hybrid_safe", "target_guided_safe"]
        assert job["targets"][0]["id"] == "product"
        assert job["results"]["tracking_preview_url"]
        assert job["results"]["tracks_url"]
        assert job["results"]["tracking_review_url"]

        preview = client.get(job["results"]["tracking_preview_url"])
        tracks = client.get(job["results"]["tracks_url"])
        assert preview.status_code == 200 and len(preview.content) > 1000
        assert tracks.status_code == 200
        payload = tracks.json()
        assert payload["tracks"][0]["target"]["id"] == "product"
        assert payload["tracks"][0]["metrics"]["visible_fraction"] > 0.0

        render_response = client.post(
            f"/api/jobs/{job_id}/render",
            data={"targets": json.dumps(targets)},
        )
        assert render_response.status_code == 202, render_response.text
        deadline = time.monotonic() + 120.0
        job = render_response.json()
        while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "completed", job
        assert job["results"]["report_url"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_web_rejects_target_keyframe_after_video_end(tmp_path) -> None:
    import json

    source, _ = generate_sample(
        tmp_path / "short.mp4",
        width=320,
        height=180,
        fps=8.0,
        shot_seconds=0.25,
    )
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    targets = [
        {
            "id": "late",
            "kind": "point",
            "role": "follow",
            "anchors": [{"time": 99.0, "x": 0.5, "y": 0.5}],
        }
    ]
    with TestClient(app) as client, source.open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("short.mp4", video, "video/mp4")},
            data={
                "preset": "fast_compare",
                "quality": "quick",
                "output_size": "180x320",
                "targets": json.dumps(targets),
            },
        )
    assert response.status_code == 400
    assert "after the video ends" in response.json()["detail"]


def test_sample_clip_is_served_from_a_source_checkout(tmp_path) -> None:
    app = create_app(work_root=tmp_path / "runs", keep_days=0)
    with TestClient(app) as client:
        assert 'id="sample"' in client.get("/").text
        head = client.head("/sample.mp4")
        response = client.get("/sample.mp4")
    assert head.status_code == 200
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 100_000
