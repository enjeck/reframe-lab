from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .agent import auto_fix_plan, save_auto_fix_trace
from .analysis_cache import load_analysis_bundle, save_analysis_bundle
from .analyzer import analyze_video
from .comparison import build_comparison_video
from .corrections import FramingCorrection, save_framing_corrections
from .ground_truth import evaluate_ground_truth, find_ground_truth
from .models import AnalysisBundle, CropPlan
from .planner import build_plan
from .profiles import get_profile
from .renderer import render_plans
from .report import write_report
from .review import build_tracking_review, save_tracking_review
from .targets import TargetSpec, TargetTrack, load_tracks, save_tracks
from .tracking import render_tracking_preview, track_targets
from .video import even, parse_size, prepare_working_video

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrackingReviewResult:
    tracks: list[TargetTrack]
    tracks_path: Path
    review: dict[str, object]
    review_path: Path
    tracking_preview_path: Path


@dataclass(frozen=True)
class DemoResult:
    output_dir: Path
    report_path: Path
    comparison_path: Path | None
    rendered_paths: dict[str, Path]
    plan_paths: dict[str, Path]
    tracks_path: Path | None = None
    tracking_preview_path: Path | None = None
    tracking_review_path: Path | None = None
    agent_trace_path: Path | None = None
    performance_path: Path | None = None
    analysis_cache_path: Path | None = None
    corrections_path: Path | None = None


def _merge_tracks(*groups: list[TargetTrack]) -> list[TargetTrack]:
    merged: dict[str, TargetTrack] = {}
    for group in groups:
        for track in group:
            merged[track.target.id] = track
    return list(merged.values())


def prepare_tracking_review(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    target_specs: list[TargetSpec] | None = None,
    analysis_height: int = 360,
    tracking_fps: float = 8.0,
    enable_learned_tracking: bool = False,
    advanced_vision: bool = False,
    enable_depth: bool = False,
    model_dir: str | Path | None = None,
    allow_model_download: bool = True,
    progress: ProgressCallback | None = None,
) -> TrackingReviewResult:
    """Track optional user annotations and emit a low-cost pre-render review."""

    del enable_depth  # Kept for API compatibility with v0.8.x.

    targets = target_specs or []
    if not targets:
        raise ValueError("Tracking review requires at least one selection")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    visual_tracks: list[TargetTrack] = []

    with prepare_working_video(input_path, progress=progress) as working:
        if targets:
            if progress:
                progress("Tracking selected targets for review")
            visual_tracks = track_targets(
                working.path,
                targets,
                analysis_height=max(300, analysis_height),
                sample_fps=max(5.0, tracking_fps),
                enable_learned=enable_learned_tracking,
                enable_masks=advanced_vision,
                enable_dense_flow=advanced_vision,
                model_dir=model_dir,
                allow_model_download=allow_model_download,
                progress=progress,
            )
        tracks = visual_tracks
        tracks_path = save_tracks(root / "tracks.json", tracks)
        review = build_tracking_review(tracks)
        review_path = save_tracking_review(root / "tracking_review.json", review)
        if progress:
            progress("Building the pre-render tracking review")
        preview_path = render_tracking_preview(
            working.path,
            tracks,
            root / "tracking_preview.mp4",
            sample_fps=min(5.0, max(2.0, tracking_fps)),
            progress=progress,
        )

    return TrackingReviewResult(
        tracks=tracks,
        tracks_path=tracks_path,
        review=review,
        review_path=review_path,
        tracking_preview_path=preview_path,
    )


def run_demo(
    input_path: str | Path,
    output_dir: str | Path,
    profile_names: list[str],
    *,
    output_size: str = "540x960",
    analysis_height: int = 240,
    analysis_fps: float = 5.0,
    candidate_step: int = 8,
    shot_threshold: float = 0.43,
    create_comparison: bool = True,
    target_specs: list[TargetSpec] | None = None,
    precomputed_tracks: list[TargetTrack] | str | Path | None = None,
    framing_corrections: list[FramingCorrection] | None = None,
    precomputed_bundle: AnalysisBundle | str | Path | None = None,
    enable_learned_tracking: bool = False,
    adaptive_dense_flow: bool = True,
    advanced_vision: bool = False,
    enable_depth: bool = False,
    model_dir: str | Path | None = None,
    allow_model_download: bool = True,
    auto_fix: bool = True,
    max_auto_fix_rounds: int = 2,
    use_analysis_proxy: bool = True,
    coarse_to_fine: bool = True,
    detector_cadence: bool = True,
    direct_ffmpeg: bool = True,
    static_scene_reuse: bool = True,
    persistent_models: bool = True,
    dnn_optimization: bool = True,
    dnn_batch_size: int = 4,
    pipeline_queues: bool = True,
    consistent_framing: bool = True,
    active_speaker: bool = True,
    progress: ProgressCallback | None = None,
) -> DemoResult:
    if not profile_names:
        raise ValueError("At least one profile is required")
    profiles = [get_profile(name) for name in profile_names]
    targets = target_specs or []
    corrections = framing_corrections or []
    if isinstance(precomputed_tracks, (str, Path)):
        supplied_tracks = load_tracks(precomputed_tracks)
    else:
        supplied_tracks = precomputed_tracks or []
    has_guidance = bool(targets or supplied_tracks)
    if any(profile.target_weight > 0.0 for profile in profiles) and not has_guidance:
        raise ValueError("Target-guided profiles require a selected target")

    requested_width, requested_height = parse_size(output_size)
    target_width, target_height = even(requested_width), even(requested_height)
    original_input = Path(input_path).resolve()
    root = Path(output_dir)
    plans_dir = root / "plans"
    renders_dir = root / "renders"
    plans_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = find_ground_truth(original_input)
    plans: dict[str, CropPlan] = {}
    plan_paths: dict[str, Path] = {}
    comparison_path: Path | None = None
    tracks_path: Path | None = None
    tracking_preview_path: Path | None = None
    tracking_review_path: Path | None = None
    agent_trace_path: Path | None = None
    performance_path: Path | None = None
    analysis_cache_path: Path | None = None
    corrections_path: Path | None = None
    agent_traces: dict[str, object] = {}
    target_tracks: list[TargetTrack] = list(supplied_tracks)
    performance: dict[str, object] = {
        "version": 2,
        "settings": {
            "analysis_proxy": use_analysis_proxy,
            "coarse_to_fine": coarse_to_fine,
            "detector_cadence": detector_cadence,
            "direct_ffmpeg": direct_ffmpeg,
            "static_scene_reuse": static_scene_reuse,
            "persistent_models": persistent_models,
            "dnn_optimization": dnn_optimization,
            "dnn_batch_size": int(max(1, dnn_batch_size)),
            "pipeline_queues": pipeline_queues,
        },
        "features": {
            "consistent_framing": consistent_framing,
            "active_speaker": active_speaker,
            "framing_corrections": len(corrections),
        },
        "stages": {},
    }
    pipeline_started = time.perf_counter()
    normalization_started = time.perf_counter()

    with prepare_working_video(original_input, progress=progress) as working:
        performance["stages"]["normalization_seconds"] = (
            time.perf_counter() - normalization_started
        )
        tracking_fps = min(10.0, max(5.0, analysis_fps))
        tracking_started = time.perf_counter()
        if targets and not supplied_tracks:
            if progress:
                progress("Tracking selected targets across the video")
            visual_tracks = track_targets(
                working.path,
                targets,
                analysis_height=max(analysis_height, 300),
                sample_fps=tracking_fps,
                enable_learned=enable_learned_tracking,
                enable_masks=advanced_vision,
                enable_dense_flow=advanced_vision,
                model_dir=model_dir,
                allow_model_download=allow_model_download,
                progress=progress,
            )
            target_tracks = _merge_tracks(target_tracks, visual_tracks)
        performance["stages"]["tracking_seconds"] = time.perf_counter() - tracking_started
        if target_tracks:
            tracks_path = save_tracks(root / "tracks.json", target_tracks)
            review = build_tracking_review(target_tracks)
            tracking_review_path = save_tracking_review(
                root / "tracking_review.json", review
            )

        if isinstance(precomputed_bundle, (str, Path)):
            if progress:
                progress("Loading cached OpenCV analysis for framing corrections")
            analysis_started = time.perf_counter()
            bundle = load_analysis_bundle(precomputed_bundle)
            performance["analysis_cache_reused"] = True
        elif isinstance(precomputed_bundle, AnalysisBundle):
            analysis_started = time.perf_counter()
            bundle = precomputed_bundle
            performance["analysis_cache_reused"] = True
        else:
            if progress:
                progress("Analysing the source once for all profiles")
            analysis_started = time.perf_counter()
            bundle = analyze_video(
                working.path,
                target_width=target_width,
                target_height=target_height,
                analysis_height=analysis_height,
                sample_fps=analysis_fps,
                candidate_step=candidate_step,
                shot_threshold=shot_threshold,
                adaptive_dense_flow=adaptive_dense_flow,
                advanced_vision=advanced_vision,
                enable_dense_flow=advanced_vision,
                enable_depth=advanced_vision and enable_depth,
                model_dir=model_dir,
                allow_model_download=allow_model_download,
                use_analysis_proxy=use_analysis_proxy,
                coarse_to_fine=coarse_to_fine,
                detector_cadence=detector_cadence,
                static_scene_reuse=static_scene_reuse,
                persistent_models=persistent_models,
                dnn_optimization=dnn_optimization,
                dnn_batch_size=dnn_batch_size,
                pipeline_queues=pipeline_queues,
                active_speaker=active_speaker,
                progress=progress,
            )
            performance["analysis_cache_reused"] = False
        performance["stages"]["analysis_seconds"] = time.perf_counter() - analysis_started
        performance["analysis"] = bundle.performance
        # Plans and reports should refer to the user's file, not the temporary working copy.
        bundle.video = replace(bundle.video, path=str(original_input))
        analysis_cache_path = save_analysis_bundle(root / "analysis_bundle.pkl.gz", bundle)
        if corrections:
            corrections_path = save_framing_corrections(root / "framing_corrections.json", corrections)

        planning_started = time.perf_counter()
        for profile in profiles:
            if progress:
                progress(f"Planning profile: {profile.name}")
            use_auto_fix = bool(
                auto_fix
                and profile.temporal
                and profile.allow_fallback
                and profile.name in {"hybrid_safe", "target_guided_safe"}
            )
            if use_auto_fix:
                revision_dir = plans_dir / "revisions" / profile.name
                fixed = auto_fix_plan(
                    bundle,
                    profile,
                    target_tracks,
                    corrections,
                    consistent_framing=consistent_framing,
                    max_rounds=max_auto_fix_rounds,
                    revision_dir=revision_dir,
                    progress=progress,
                )
                plan = fixed.plan
                agent_traces[profile.name] = fixed.trace
            else:
                plan = build_plan(
                    bundle, profile, target_tracks, corrections,
                    consistent_framing=consistent_framing,
                )
            if ground_truth_path is not None:
                plan.metrics.update(evaluate_ground_truth(plan, ground_truth_path))
            plan_path = plans_dir / f"{profile.name}.json"
            plan.save(plan_path)
            plans[profile.name] = plan
            plan_paths[profile.name] = plan_path

        performance["stages"]["planning_seconds"] = time.perf_counter() - planning_started
        if agent_traces:
            agent_trace_path = save_auto_fix_trace(
                root / "auto_fix_trace.json",
                {
                    "version": 1,
                    "enabled": True,
                    "max_rounds": max_auto_fix_rounds,
                    "profiles": agent_traces,
                },
            )

        output_paths = {name: renders_dir / f"{name}.mp4" for name in plans}
        if progress:
            progress("Rendering all profiles in a single source decode pass")
        render_started = time.perf_counter()
        rendered_paths = render_plans(
            working.path,
            plans,
            output_paths,
            direct_ffmpeg=direct_ffmpeg,
            pipeline_queues=pipeline_queues,
            progress=progress,
        )
        performance["stages"]["render_seconds"] = time.perf_counter() - render_started

        if target_tracks:
            if progress:
                progress("Building the selected-target tracking preview")
            tracking_preview_path = render_tracking_preview(
                working.path,
                target_tracks,
                root / "tracking_preview.mp4",
                sample_fps=min(5.0, tracking_fps),
                progress=progress,
            )

        comparison_started = time.perf_counter()
        if create_comparison:
            if progress:
                progress("Building the labelled side-by-side comparison")
            comparison_path = build_comparison_video(
                working.path,
                rendered_paths,
                root / "comparison.mp4",
            )
        performance["stages"]["comparison_seconds"] = time.perf_counter() - comparison_started

        source_normalized = working.normalized
        normalization_reason = working.reason

    if progress:
        progress("Writing the report")
    report_started = time.perf_counter()
    report_path = write_report(
        root,
        bundle=bundle,
        plans=list(plans.values()),
        rendered_paths=rendered_paths,
        comparison_path=comparison_path,
        ground_truth_path=ground_truth_path,
        target_tracks=target_tracks,
        tracking_preview_path=tracking_preview_path,
        agent_trace_path=agent_trace_path,
    )
    performance["stages"]["report_seconds"] = time.perf_counter() - report_started
    performance["stages"]["total_seconds"] = time.perf_counter() - pipeline_started
    performance_path = root / "performance.json"
    performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "input": str(original_input),
                "source_normalized": source_normalized,
                "normalization_reason": normalization_reason,
                "profiles": profile_names,
                "output_size": [target_width, target_height],
                "analysis_height": analysis_height,
                "analysis_fps": analysis_fps,
                "candidate_step": candidate_step,
                "shot_threshold": shot_threshold,
                "adaptive_dense_flow": adaptive_dense_flow,
                "experimental_vision": advanced_vision,
                "learned_tracking": enable_learned_tracking,
                "auto_fix": auto_fix,
                "max_auto_fix_rounds": max_auto_fix_rounds,
                "consistent_framing": consistent_framing,
                "active_speaker": active_speaker,
                "framing_corrections": [item.to_dict() for item in corrections],
                "analysis_cache": str(analysis_cache_path) if analysis_cache_path else None,
                "speed_optimizations": {
                    "analysis_proxy": use_analysis_proxy,
                    "coarse_to_fine": coarse_to_fine,
                    "detector_cadence": detector_cadence,
                    "direct_ffmpeg": direct_ffmpeg,
                    "static_scene_reuse": static_scene_reuse,
                    "persistent_models": persistent_models,
                    "dnn_optimization": dnn_optimization,
                    "dnn_batch_size": int(max(1, dnn_batch_size)),
                    "pipeline_queues": pipeline_queues,
                },
                "performance": str(performance_path),
                "auto_fix_trace": str(agent_trace_path) if agent_trace_path else None,
                "targets": [target.to_dict() for target in targets],
                "tracks": str(tracks_path) if tracks_path else None,
                "tracking_review": (
                    str(tracking_review_path) if tracking_review_path else None
                ),
                "tracking_preview": (
                    str(tracking_preview_path) if tracking_preview_path else None
                ),
                "plans": {name: str(path) for name, path in plan_paths.items()},
                "renders": {name: str(path) for name, path in rendered_paths.items()},
                "comparison": str(comparison_path) if comparison_path else None,
                "report": str(report_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return DemoResult(
        output_dir=root,
        report_path=report_path,
        comparison_path=comparison_path,
        rendered_paths=rendered_paths,
        plan_paths=plan_paths,
        tracks_path=tracks_path,
        tracking_preview_path=tracking_preview_path,
        tracking_review_path=tracking_review_path,
        agent_trace_path=agent_trace_path,
        performance_path=performance_path,
        analysis_cache_path=analysis_cache_path,
        corrections_path=corrections_path,
    )
