from __future__ import annotations

import json
from pathlib import Path

import pytest

from reframelab import agent
from reframelab.agent import AutoFixPolicy, auto_fix_plan, inspect_plan, revise_profile
from reframelab.benchmark import benchmark_opencv, compare_benchmarks
from reframelab.cloud import parse_s3_uri
from reframelab.models import AnalysisBundle, CropPlan, PlanPoint, VideoInfo
from reframelab.profiles import get_profile


def _plan(metrics: dict[str, float], profile: str = "hybrid_safe") -> CropPlan:
    return CropPlan(
        profile=profile,
        description="test",
        source="source.mp4",
        source_width=1920,
        source_height=1080,
        source_fps=30.0,
        target_width=540,
        target_height=960,
        sample_fps=5.0,
        points=[
            PlanPoint(
                frame_index=0,
                time=0.0,
                shot_id=0,
                center_x_norm=0.5,
                center_y_norm=0.5,
                mode="hold",
                content_coverage=0.8,
                confidence=0.8,
            )
        ],
        metrics=metrics,
    )


def _good_metrics() -> dict[str, float]:
    return {
        "mean_shared_coverage": 0.88,
        "p10_shared_coverage": 0.75,
        "stability_score": 0.86,
        "mean_confidence": 0.8,
        "trusted_text_frame_fraction": 0.5,
        "mean_text_integrity_in_detail": 0.98,
        "text_visible_anywhere_fraction": 1.0,
        "text_region_cut_fraction": 0.01,
        "mean_jerk_viewports_per_second3": 0.2,
        "p95_jerk_viewports_per_second3": 0.6,
        "layout_switches_per_minute": 2.0,
    }


def test_inspect_plan_finds_specific_failures() -> None:
    metrics = _good_metrics()
    metrics.update(
        {
            "p10_shared_coverage": 0.2,
            "mean_text_integrity_in_detail": 0.55,
            "text_visible_anywhere_fraction": 0.7,
            "text_region_cut_fraction": 0.4,
            "stability_score": 0.2,
            "mean_jerk_viewports_per_second3": 4.0,
            "p95_jerk_viewports_per_second3": 12.0,
            "layout_switches_per_minute": 30.0,
        }
    )
    codes = {issue.code for issue in inspect_plan(_plan(metrics), AutoFixPolicy())}
    assert {"low_content_coverage", "text_clipped", "camera_too_abrupt", "layout_churn"} <= codes


def test_revise_profile_changes_only_relevant_settings() -> None:
    profile = get_profile("hybrid_safe")
    issues = inspect_plan(
        _plan(
            {
                **_good_metrics(),
                "mean_text_integrity_in_detail": 0.5,
                "text_region_cut_fraction": 0.5,
                "stability_score": 0.3,
                "p95_jerk_viewports_per_second3": 9.0,
            }
        )
    )
    revised = revise_profile(profile, issues, 0)
    assert revised.signal_weights["text_complete"] > profile.signal_weights["text_complete"]
    assert revised.text_integrity_threshold > profile.text_integrity_threshold
    assert revised.max_jerk_crop_widths_per_second3 < profile.max_jerk_crop_widths_per_second3
    assert revised.minimum_jerk_jerk_weight > profile.minimum_jerk_jerk_weight


def test_auto_fix_plan_selects_improved_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = get_profile("hybrid_safe")

    def fake_build_plan(bundle, candidate_profile, tracks):
        del bundle, tracks
        if candidate_profile.signal_weights.get("text_complete", 0.0) < 0.35:
            metrics = {
                **_good_metrics(),
                "mean_text_integrity_in_detail": 0.55,
                "text_visible_anywhere_fraction": 0.65,
                "text_region_cut_fraction": 0.4,
            }
        else:
            metrics = _good_metrics()
        return _plan(metrics)

    monkeypatch.setattr(agent, "build_plan", fake_build_plan)
    bundle = AnalysisBundle(
        video=VideoInfo("source.mp4", 1920, 1080, 30.0, 90, 3.0),
        analysis_width=320,
        analysis_height=180,
        target_width=540,
        target_height=960,
        sample_fps=5.0,
        frames=[],
    )
    result = auto_fix_plan(bundle, profile, max_rounds=2, revision_dir=tmp_path)
    assert result.trace["selected_round"] in {1, 2}
    assert result.trace["passed_all_gates"] is True
    assert result.plan.implementation["auto_fix"]["selected_round"] in {1, 2}
    assert (tmp_path / "round_0.json").is_file()
    assert (tmp_path / "round_1.json").is_file()


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/path/video.mp4") == ("bucket", "path/video.mp4")
    with pytest.raises(ValueError):
        parse_s3_uri("https://example.com/video.mp4")


def test_benchmark_and_compare(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    comparison_path = tmp_path / "comparison.json"
    baseline = benchmark_opencv(
        output_path=baseline_path,
        frame_count=3,
        iterations=1,
        synthetic_width=320,
        synthetic_height=180,
    )
    candidate = json.loads(json.dumps(baseline))
    for values in candidate["operations"].values():
        values["median_ms"] *= 0.5
    candidate["production_mix_frames_per_second"] *= 2.0
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    result = compare_benchmarks(baseline_path, candidate_path, comparison_path)
    assert result["production_mix_speedup_x"] == pytest.approx(2.0)
    assert result["operations"]["resize"]["speedup_x"] == pytest.approx(2.0)
    assert comparison_path.is_file()


def test_cloud_job_downloads_runs_and_uploads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil
    import sys
    from types import SimpleNamespace

    from reframelab import cloud
    from reframelab.pipeline import DemoResult

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video-bytes")
    uploads: dict[str, bytes] = {}

    class FakeS3:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            assert (bucket, key) == ("input-bucket", "input/source.mp4")
            shutil.copyfile(source, filename)

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            assert bucket == "output-bucket"
            uploads[key] = Path(filename).read_bytes()

    fake_s3 = FakeS3()
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: fake_s3))

    def fake_run_demo(input_path, output_dir, profiles, **kwargs):
        del input_path, profiles, kwargs
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        report = root / "report.html"
        report.write_text("<html>ok</html>", encoding="utf-8")
        render = root / "renders" / "hybrid_safe.mp4"
        render.parent.mkdir(parents=True, exist_ok=True)
        render.write_bytes(b"render")
        trace = root / "auto_fix_trace.json"
        trace.write_text('{"passed": true}', encoding="utf-8")
        return DemoResult(
            output_dir=root,
            report_path=report,
            comparison_path=None,
            rendered_paths={"hybrid_safe": render},
            plan_paths={},
            agent_trace_path=trace,
        )

    monkeypatch.setattr(cloud, "run_demo", fake_run_demo)
    manifest = cloud.run_s3_job(
        input_uri="s3://input-bucket/input/source.mp4",
        output_uri="s3://output-bucket/jobs/demo",
        quality="quick",
        auto_fix_rounds=1,
    )
    assert manifest["status"] == "completed"
    assert manifest["report"] == "s3://output-bucket/jobs/demo/report.html"
    assert manifest["agent_trace"] == "s3://output-bucket/jobs/demo/auto_fix_trace.json"
    assert "jobs/demo/results.zip" in uploads
    assert "jobs/demo/cloud_job.json" in uploads


def test_cloud_job_uploads_failure_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil
    import sys
    from types import SimpleNamespace

    from reframelab import cloud

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video-bytes")
    uploads: dict[str, bytes] = {}

    class FakeS3:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            shutil.copyfile(source, filename)

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            uploads[key] = Path(filename).read_bytes()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: FakeS3()))
    monkeypatch.setattr(cloud, "run_demo", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        cloud.run_s3_job(
            input_uri="s3://input-bucket/input/source.mp4",
            output_uri="s3://output-bucket/jobs/failure",
            quality="quick",
        )
    payload = json.loads(uploads["jobs/failure/error.json"])
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"] == "boom"


def test_cloud_job_reads_marks_and_renders_target_edits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    import shutil
    import sys
    from types import SimpleNamespace

    from reframelab import cloud
    from reframelab.pipeline import DemoResult

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video-bytes")
    marks = tmp_path / "marks.json"
    marks.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "id": "t1",
                        "label": "Follow 1",
                        "kind": "point",
                        "role": "follow",
                        "anchors": [{"time": 0.5, "x": 0.4, "y": 0.5, "width": 0, "height": 0}],
                    }
                ],
                "corrections": [
                    {"time": 1.0, "center_x": 0.3, "center_y": 0.5, "scope": "shot", "mode": "auto"}
                ],
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    class FakeS3:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            shutil.copyfile(marks if key.endswith("marks.json") else source, filename)

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            pass

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: FakeS3()))

    def fake_run_demo(input_path, output_dir, profiles, **kwargs):
        seen["profiles"] = profiles
        seen["targets"] = kwargs["target_specs"]
        seen["corrections"] = kwargs["framing_corrections"]
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        report = root / "report.html"
        report.write_text("ok", encoding="utf-8")
        return DemoResult(
            output_dir=root, report_path=report, comparison_path=None, rendered_paths={}, plan_paths={}
        )

    monkeypatch.setattr(cloud, "run_demo", fake_run_demo)
    cloud.run_s3_job(
        input_uri="s3://bucket/uploads/j/source.mp4",
        output_uri="s3://bucket/jobs/j",
        marks_uri="s3://bucket/uploads/j/marks.json",
    )
    assert seen["profiles"][0] == "target_guided_safe"
    assert len(seen["targets"]) == 1 and seen["targets"][0].label == "Follow 1"
    assert len(seen["corrections"]) == 1
