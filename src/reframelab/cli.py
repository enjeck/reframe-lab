from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import cv2 as cv

from . import __version__
from .agent import auto_fix_plan, save_auto_fix_trace
from .analyzer import analyze_video
from .benchmark import benchmark_opencv, compare_benchmarks
from .corrections import FramingCorrection, parse_framing_corrections
from .ground_truth import evaluate_ground_truth, find_ground_truth
from .model_assets import (
    CORE_MODEL_NAMES,
    EXPERIMENTAL_MODEL_NAMES,
    MODEL_ASSETS,
    ensure_model,
    model_status,
)
from .models import CropPlan
from .pipeline import prepare_tracking_review, run_demo
from .planner import build_plan
from .profiles import DEFAULT_DEMO_PROFILES, PROFILES, TARGET_DEMO_PROFILES, get_profile
from .renderer import render_plan
from .runtime import opencv_runtime_info
from .synthetic import generate_sample
from .targets import TargetSpec, parse_target_specs
from .tracking import track_targets
from .video import even, parse_size, prepare_working_video


def _progress(message: str) -> None:
    print(f"[reframe-lab] {message}", file=sys.stderr, flush=True)


def _profile_list(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    for name in names:
        get_profile(name)
    return names


def _load_corrections(path: Path | None) -> list[FramingCorrection]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read framing correction file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Framing correction file {path} is not valid JSON") from exc
    if isinstance(payload, dict) and "corrections" in payload:
        payload = payload["corrections"]
    return parse_framing_corrections(payload)


def _load_targets(path: Path | None) -> list[TargetSpec]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read target file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Target file {path} is not valid JSON") from exc
    if isinstance(payload, dict) and "targets" in payload:
        payload = payload["targets"]
    return parse_target_specs(payload)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--no-model-download",
        action="store_true",
        help="Use installed models only; do not download OpenCV Zoo weights",
    )


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-size",
        default="540x960",
        help=(
            "Output size as WIDTHxHEIGHT. Portrait, square, landscape, "
            "ultrawide, and custom ratios are supported."
        ),
    )
    parser.add_argument("--analysis-height", type=int, default=240)
    parser.add_argument("--analysis-fps", type=float, default=5.0)
    parser.add_argument("--candidate-step", type=int, default=8)
    parser.add_argument("--shot-threshold", type=float, default=0.43)
    parser.add_argument(
        "--experimental-vision",
        action="store_true",
        help=(
            "Enable research-only RAFT, depth, EfficientSAM and ViTTrack paths. "
            "The default pipeline stays on the smaller OpenCV core."
        ),
    )
    parser.add_argument(
        "--no-analysis-proxy",
        action="store_true",
        help="Decode and sample frames in Python instead of the FFmpeg analysis stream",
    )
    parser.add_argument(
        "--no-coarse-to-fine",
        action="store_true",
        help="Analyse the whole video at the requested quality instead of refining difficult intervals",
    )
    parser.add_argument(
        "--no-detector-cadence",
        action="store_true",
        help="Run text and person DNN detection on every sampled frame",
    )
    parser.add_argument(
        "--legacy-render",
        action="store_true",
        help="Use the old temporary-MP4 double-encode renderer for comparison",
    )
    parser.add_argument(
        "--no-static-reuse",
        action="store_true",
        help="Recompute visual evidence even when consecutive sampled frames are unchanged",
    )
    parser.add_argument(
        "--no-persistent-models",
        action="store_true",
        help="Reload text and person models for every run instead of keeping them warm",
    )
    parser.add_argument(
        "--no-dnn-optimization",
        action="store_true",
        help="Disable per-model OpenCV DNN engine tuning and batched inference",
    )
    parser.add_argument(
        "--dnn-batch-size",
        type=int,
        default=4,
        help="Maximum OpenCV DNN analysis batch size (default: 4)",
    )
    parser.add_argument(
        "--no-pipeline-queues",
        action="store_true",
        help="Disable bounded decode, analysis, and encoding queues",
    )
    parser.add_argument(
        "--no-framing-lock",
        action="store_true",
        help="Do not keep visually matching camera shots on a shared crop anchor",
    )
    parser.add_argument(
        "--no-active-speaker",
        action="store_true",
        help="Disable audio and mouth-motion active-speaker evidence",
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="Disable the deterministic crop quality review and replanning loop",
    )
    parser.add_argument(
        "--auto-fix-rounds",
        type=int,
        default=2,
        help="Maximum number of corrective replanning passes (default: 2)",
    )
    _add_model_arguments(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reframe-lab",
        description="Explainable OpenCV 5 video reframing for arbitrary output sizes.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles_parser = subparsers.add_parser("profiles", help="List built-in comparison profiles")
    profiles_parser.set_defaults(handler=_handle_profiles)

    models_parser = subparsers.add_parser(
        "models", help="Show or download the small OpenCV Zoo model files"
    )
    models_parser.add_argument(
        "--download",
        choices=["core", "experimental", "all", *sorted(MODEL_ASSETS)],
        help="Download core models, experimental models, all models, or one named model",
    )
    models_parser.add_argument("--model-dir", type=Path)
    models_parser.set_defaults(handler=_handle_models)

    doctor_parser = subparsers.add_parser("doctor", help="Check OpenCV and FFmpeg dependencies")
    doctor_parser.add_argument("--strict-opencv-5", action="store_true")
    doctor_parser.add_argument("--model-dir", type=Path)
    doctor_parser.set_defaults(handler=_handle_doctor)

    sample_parser = subparsers.add_parser("generate-sample", help="Generate the annotated synthetic test video")
    sample_parser.add_argument("output", type=Path)
    sample_parser.add_argument("--size", default="960x540")
    sample_parser.add_argument("--fps", type=float, default=24.0)
    sample_parser.set_defaults(handler=_handle_generate_sample)

    demo_parser = subparsers.add_parser("demo", help="Compare multiple reframing profiles on one video")
    demo_parser.add_argument("input", type=Path)
    demo_parser.add_argument("--output", type=Path, required=True)
    demo_parser.add_argument(
        "--profiles",
        type=_profile_list,
        default=None,
        help=(
            "Comma-separated profile names. Defaults to automatic profiles, or target-guided "
            "profiles when --targets is supplied."
        ),
    )
    demo_parser.add_argument(
        "--targets",
        type=Path,
        help="JSON file containing point/box selections and optional correction keyframes",
    )
    demo_parser.add_argument(
        "--corrections",
        type=Path,
        help="JSON file containing manual crop and layout corrections",
    )
    demo_parser.add_argument("--no-comparison", action="store_true")
    _add_analysis_arguments(demo_parser)
    demo_parser.set_defaults(handler=_handle_demo)

    sample_demo_parser = subparsers.add_parser(
        "sample-demo",
        help="Generate the synthetic video and run the full comparison in one command",
    )
    sample_demo_parser.add_argument("--output", type=Path, required=True)
    sample_demo_parser.add_argument(
        "--profiles",
        type=_profile_list,
        default=DEFAULT_DEMO_PROFILES,
    )
    _add_analysis_arguments(sample_demo_parser)
    sample_demo_parser.set_defaults(handler=_handle_sample_demo)

    analyze_parser = subparsers.add_parser("analyze", help="Create one JSON crop plan")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument("--profile", default="hybrid_safe", choices=sorted(PROFILES))
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument(
        "--targets",
        type=Path,
        help="JSON file containing point/box selections and optional correction keyframes",
    )
    analyze_parser.add_argument(
        "--corrections",
        type=Path,
        help="JSON file containing manual crop and layout corrections",
    )
    _add_analysis_arguments(analyze_parser)
    analyze_parser.set_defaults(handler=_handle_analyze)

    track_parser = subparsers.add_parser(
        "track",
        help="Track selected points or objects and render a diagnostic preview",
    )
    track_parser.add_argument("input", type=Path)
    track_parser.add_argument("--targets", type=Path)
    track_parser.add_argument("--output", type=Path, required=True)
    track_parser.add_argument("--analysis-height", type=int, default=360)
    track_parser.add_argument("--tracking-fps", type=float, default=8.0)
    track_parser.add_argument(
        "--experimental-vision",
        action="store_true",
        help="Enable ViTTrack, EfficientSAM and learned dense-flow mask propagation",
    )
    _add_model_arguments(track_parser)
    track_parser.set_defaults(handler=_handle_track)

    render_parser = subparsers.add_parser("render", help="Render an existing JSON crop plan")
    render_parser.add_argument("input", type=Path)
    render_parser.add_argument("plan", type=Path)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument(
        "--legacy-render",
        action="store_true",
        help="Use the temporary-MP4 double-encode renderer",
    )
    render_parser.add_argument(
        "--no-pipeline-queues",
        action="store_true",
        help="Disable asynchronous frame decode and FFmpeg writer queues",
    )
    render_parser.set_defaults(handler=_handle_render)

    web_parser = subparsers.add_parser("web", help="Run the local upload-and-compare web interface")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--work-dir", type=Path, default=Path("web-runs"))
    web_parser.add_argument("--max-workers", type=int, default=1)
    web_parser.add_argument("--max-upload-gb", type=float, default=8.0)
    web_parser.add_argument("--keep-days", type=int, default=7)
    web_parser.add_argument("--no-browser", action="store_true")
    web_parser.add_argument("--log-level", default="warning")
    web_parser.set_defaults(handler=_handle_web)

    benchmark_parser = subparsers.add_parser(
        "benchmark-opencv",
        help="Benchmark the OpenCV kernels used by the production pipeline",
    )
    benchmark_parser.add_argument("--input", type=Path)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--frames", type=int, default=12)
    benchmark_parser.add_argument("--iterations", type=int, default=8)
    benchmark_parser.set_defaults(handler=_handle_benchmark_opencv)

    compare_benchmark_parser = subparsers.add_parser(
        "compare-benchmarks",
        help="Compare a standard OpenCV benchmark with a COOL benchmark",
    )
    compare_benchmark_parser.add_argument("baseline", type=Path)
    compare_benchmark_parser.add_argument("candidate", type=Path)
    compare_benchmark_parser.add_argument("--output", type=Path, required=True)
    compare_benchmark_parser.set_defaults(handler=_handle_compare_benchmarks)

    cloud_parser = subparsers.add_parser(
        "cloud-job",
        help="Run one S3-to-S3 batch job for ECS or another AWS worker",
    )
    cloud_parser.add_argument("--input", required=True, help="Input S3 URI")
    cloud_parser.add_argument("--output", required=True, help="Output S3 prefix")
    cloud_parser.add_argument("--output-size", default="540x960")
    cloud_parser.add_argument("--quality", choices=["quick", "balanced", "careful"], default="balanced")
    cloud_parser.add_argument("--compare", action="store_true")
    cloud_parser.add_argument("--auto-fix-rounds", type=int, default=2)
    cloud_parser.add_argument("--marks", help="Optional S3 URI of a JSON file with targets and corrections")
    cloud_parser.set_defaults(handler=_handle_cloud_job)

    cloud_benchmark_parser = subparsers.add_parser(
        "cloud-benchmark",
        help="Run the OpenCV benchmark in AWS and upload the result to S3",
    )
    cloud_benchmark_parser.add_argument("--output", required=True, help="Output S3 JSON URI")
    cloud_benchmark_parser.add_argument("--input", help="Optional input video S3 URI")
    cloud_benchmark_parser.add_argument("--frames", type=int, default=12)
    cloud_benchmark_parser.add_argument("--iterations", type=int, default=8)
    cloud_benchmark_parser.set_defaults(handler=_handle_cloud_benchmark)

    return parser


def _handle_profiles(args: argparse.Namespace) -> int:
    del args
    for name, profile in PROFILES.items():
        print(f"{name:18} {profile.description}")
    return 0


def _handle_models(args: argparse.Namespace) -> int:
    if args.download:
        if args.download == "core":
            names = list(CORE_MODEL_NAMES)
        elif args.download == "experimental":
            names = list(EXPERIMENTAL_MODEL_NAMES)
        elif args.download == "all":
            names = sorted(MODEL_ASSETS)
        else:
            names = [args.download]
        for name in names:
            path = ensure_model(
                name,
                model_dir=args.model_dir,
                download=True,
                strict=True,
                progress=_progress,
            )
            print(path)
    for item in model_status(args.model_dir):
        state = "installed" if item["installed"] else "missing"
        print(
            f"{item['name']:24} {item['group']:12} {state:9} "
            f"{item['license']:10} {item['path']}"
        )
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    major = int(cv.__version__.split(".")[0])
    ffmpeg = shutil.which("ffmpeg")
    print(f"Python: {sys.version.split()[0]}")
    runtime = opencv_runtime_info()
    print(f"OpenCV: {cv.__version__}")
    print(f"OpenCV distribution: {runtime['opencv_distribution']}")
    print(f"OpenCV module: {runtime['opencv_module']}")
    print(f"Machine: {runtime['machine']}")
    print(f"FFmpeg: {ffmpeg or 'NOT FOUND'}")
    for item in model_status(args.model_dir):
        if item["installed"]:
            state = "installed"
        elif item["group"] == "core":
            state = "missing (downloads on first use)"
        else:
            state = "missing (experimental; explicit opt-in only)"
        print(f"Model {item['name']} [{item['group']}]: {state}")
    if ffmpeg is None:
        print("ERROR: FFmpeg is required.", file=sys.stderr)
        return 2
    if major < 5:
        message = "OpenCV 5 is recommended. This runtime is using the API-compatible OpenCV 4 path."
        if args.strict_opencv_5:
            print(f"ERROR: {message}", file=sys.stderr)
            return 2
        print(f"WARNING: {message}", file=sys.stderr)
    print("Dependency check passed.")
    return 0


def _handle_generate_sample(args: argparse.Namespace) -> int:
    width, height = parse_size(args.size)
    video, ground_truth = generate_sample(args.output, width=width, height=height, fps=args.fps)
    print(video)
    print(ground_truth)
    return 0


def _handle_demo(args: argparse.Namespace) -> int:
    targets = _load_targets(args.targets)
    corrections = _load_corrections(args.corrections)
    guided = bool(targets)
    profiles = args.profiles or (TARGET_DEMO_PROFILES if guided else DEFAULT_DEMO_PROFILES)
    result = run_demo(
        args.input,
        args.output,
        profiles,
        output_size=args.output_size,
        analysis_height=args.analysis_height,
        analysis_fps=args.analysis_fps,
        candidate_step=args.candidate_step,
        shot_threshold=args.shot_threshold,
        create_comparison=not args.no_comparison,
        target_specs=targets,
        framing_corrections=corrections,
        enable_learned_tracking=args.experimental_vision,
        advanced_vision=args.experimental_vision,
        enable_depth=args.experimental_vision,
        model_dir=args.model_dir,
        allow_model_download=not args.no_model_download,
        auto_fix=not args.no_auto_fix,
        max_auto_fix_rounds=max(0, args.auto_fix_rounds),
        use_analysis_proxy=not args.no_analysis_proxy,
        coarse_to_fine=not args.no_coarse_to_fine,
        detector_cadence=not args.no_detector_cadence,
        direct_ffmpeg=not args.legacy_render,
        static_scene_reuse=not args.no_static_reuse,
        persistent_models=not args.no_persistent_models,
        dnn_optimization=not args.no_dnn_optimization,
        dnn_batch_size=max(1, args.dnn_batch_size),
        pipeline_queues=not args.no_pipeline_queues,
        consistent_framing=not args.no_framing_lock,
        active_speaker=not args.no_active_speaker,
        progress=_progress,
    )
    print(result.report_path)
    return 0


def _handle_sample_demo(args: argparse.Namespace) -> int:
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    sample_path = root / "sample_input.mp4"
    generate_sample(sample_path)
    result = run_demo(
        sample_path,
        root,
        args.profiles,
        output_size=args.output_size,
        analysis_height=args.analysis_height,
        analysis_fps=args.analysis_fps,
        candidate_step=args.candidate_step,
        shot_threshold=args.shot_threshold,
        create_comparison=True,
        enable_learned_tracking=args.experimental_vision,
        advanced_vision=args.experimental_vision,
        enable_depth=args.experimental_vision,
        model_dir=args.model_dir,
        allow_model_download=not args.no_model_download,
        auto_fix=not args.no_auto_fix,
        max_auto_fix_rounds=max(0, args.auto_fix_rounds),
        use_analysis_proxy=not args.no_analysis_proxy,
        coarse_to_fine=not args.no_coarse_to_fine,
        detector_cadence=not args.no_detector_cadence,
        direct_ffmpeg=not args.legacy_render,
        static_scene_reuse=not args.no_static_reuse,
        persistent_models=not args.no_persistent_models,
        dnn_optimization=not args.no_dnn_optimization,
        dnn_batch_size=max(1, args.dnn_batch_size),
        pipeline_queues=not args.no_pipeline_queues,
        consistent_framing=not args.no_framing_lock,
        active_speaker=not args.no_active_speaker,
        progress=_progress,
    )
    print(result.report_path)
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    requested_width, requested_height = parse_size(args.output_size)
    target_width, target_height = even(requested_width), even(requested_height)
    original_input = args.input.resolve()
    targets = _load_targets(args.targets)
    corrections = _load_corrections(args.corrections)
    profile = get_profile(args.profile)
    if profile.target_weight > 0.0 and not targets:
        raise ValueError(f"Profile {profile.name!r} requires --targets")
    tracks = []
    with prepare_working_video(original_input, progress=_progress) as working:
        if targets:
            tracks.extend(
                track_targets(
                    working.path,
                    targets,
                    analysis_height=max(args.analysis_height, 300),
                    sample_fps=min(10.0, max(5.0, args.analysis_fps)),
                    enable_learned=args.experimental_vision,
                    enable_masks=args.experimental_vision,
                    enable_dense_flow=args.experimental_vision,
                    model_dir=args.model_dir,
                    allow_model_download=not args.no_model_download,
                    progress=_progress,
                )
            )
        bundle = analyze_video(
            working.path,
            target_width=target_width,
            target_height=target_height,
            analysis_height=args.analysis_height,
            sample_fps=args.analysis_fps,
            candidate_step=args.candidate_step,
            shot_threshold=args.shot_threshold,
            advanced_vision=args.experimental_vision,
            enable_dense_flow=args.experimental_vision,
            enable_depth=args.experimental_vision,
            model_dir=args.model_dir,
            allow_model_download=not args.no_model_download,
            use_analysis_proxy=not args.no_analysis_proxy,
            coarse_to_fine=not args.no_coarse_to_fine,
            detector_cadence=not args.no_detector_cadence,
            static_scene_reuse=not args.no_static_reuse,
            persistent_models=not args.no_persistent_models,
            dnn_optimization=not args.no_dnn_optimization,
            dnn_batch_size=max(1, args.dnn_batch_size),
            pipeline_queues=not args.no_pipeline_queues,
            active_speaker=not args.no_active_speaker,
            progress=_progress,
        )
        bundle.video = replace(bundle.video, path=str(original_input))
    if (
        not args.no_auto_fix
        and profile.temporal
        and profile.allow_fallback
        and profile.name in {"hybrid_safe", "target_guided_safe"}
    ):
        fixed = auto_fix_plan(
            bundle,
            profile,
            tracks,
            corrections,
            consistent_framing=not args.no_framing_lock,
            max_rounds=max(0, args.auto_fix_rounds),
            revision_dir=args.output.parent / f"{args.output.stem}_revisions",
            progress=_progress,
        )
        plan = fixed.plan
        save_auto_fix_trace(
            args.output.with_suffix(".auto_fix.json"),
            {"version": 1, "profiles": {profile.name: fixed.trace}},
        )
    else:
        plan = build_plan(
            bundle, profile, tracks, corrections,
            consistent_framing=not args.no_framing_lock,
        )
    ground_truth = find_ground_truth(original_input)
    if ground_truth:
        plan.metrics.update(evaluate_ground_truth(plan, ground_truth))
    plan.save(args.output)
    print(args.output)
    return 0

def _handle_track(args: argparse.Namespace) -> int:
    targets = _load_targets(args.targets)
    if not targets:
        raise ValueError("Supply --targets")
    result = prepare_tracking_review(
        args.input.resolve(),
        args.output.resolve(),
        target_specs=targets,
        analysis_height=args.analysis_height,
        tracking_fps=args.tracking_fps,
        enable_learned_tracking=args.experimental_vision,
        advanced_vision=args.experimental_vision,
        model_dir=args.model_dir,
        allow_model_download=not args.no_model_download,
        progress=_progress,
    )
    print(result.tracks_path)
    print(result.review_path)
    print(result.tracking_preview_path)
    return 0

def _handle_render(args: argparse.Namespace) -> int:
    plan = CropPlan.load(args.plan)
    with prepare_working_video(args.input, progress=_progress) as working:
        render_plan(
            working.path,
            plan,
            args.output,
            direct_ffmpeg=not args.legacy_render,
            pipeline_queues=not args.no_pipeline_queues,
            progress=_progress,
        )
    print(args.output)
    return 0


def _handle_web(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.max_workers < 1:
        raise ValueError("max-workers must be at least 1")
    if args.max_upload_gb <= 0:
        raise ValueError("max-upload-gb must be positive")
    from .webapp import run_server

    run_server(
        host=args.host,
        port=args.port,
        work_root=args.work_dir,
        max_workers=args.max_workers,
        max_upload_gb=args.max_upload_gb,
        keep_days=args.keep_days,
        open_browser=not args.no_browser,
        log_level=args.log_level,
    )
    return 0


def _handle_benchmark_opencv(args: argparse.Namespace) -> int:
    payload = benchmark_opencv(
        input_path=args.input,
        output_path=args.output,
        frame_count=max(3, args.frames),
        iterations=max(1, args.iterations),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _handle_compare_benchmarks(args: argparse.Namespace) -> int:
    payload = compare_benchmarks(args.baseline, args.candidate, args.output)
    print(json.dumps(payload, indent=2))
    return 0


def _handle_cloud_job(args: argparse.Namespace) -> int:
    from .cloud import run_s3_job

    payload = run_s3_job(
        input_uri=args.input,
        output_uri=args.output,
        output_size=args.output_size,
        quality=args.quality,
        compare=args.compare,
        auto_fix_rounds=max(0, args.auto_fix_rounds),
        marks_uri=args.marks,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _handle_cloud_benchmark(args: argparse.Namespace) -> int:
    from .cloud import run_s3_benchmark

    payload = run_s3_benchmark(
        output_uri=args.output,
        input_uri=args.input,
        frame_count=max(3, args.frames),
        iterations=max(1, args.iterations),
    )
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
