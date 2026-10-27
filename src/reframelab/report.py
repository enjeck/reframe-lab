from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable
from pathlib import Path

from .models import AnalysisBundle, CropPlan
from .targets import TargetTrack

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5"]


def _relative(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _format_metric(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _metric_columns(plans: Iterable[CropPlan]) -> list[str]:
    keys: set[str] = set()
    for plan in plans:
        keys.update(plan.metrics)
    preferred = [
        "proxy_quality_score_0_100",
        "mean_shared_coverage",
        "p10_shared_coverage",
        "stability_score",
        "path_length_viewports",
        "mean_velocity_viewports_per_second",
        "path_length_crop_widths",
        "mean_velocity_crop_widths_per_second",
        "alternate_layout_fraction",
        "split_layout_fraction",
        "context_detail_fraction",
        "full_context_fallback_fraction",
        "fallback_fraction",
        "mean_confidence",
        "repeated_shot_center_spread_viewports",
        "active_speaker_evidence_fraction",
        "mean_active_speaker_panel_fraction",
        "ground_truth_mean_center_error_crop_widths",
        "ground_truth_mode_accuracy",
        "ground_truth_fallback_recall",
    ]
    return [name for name in preferred if name in keys] + sorted(keys.difference(preferred))


def write_metrics_files(output_dir: Path, plans: list[CropPlan]) -> tuple[Path, Path]:
    columns = _metric_columns(plans)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile", *columns])
        for plan in plans:
            writer.writerow([plan.profile, *[plan.metrics.get(column, "") for column in columns]])

    json_path = output_dir / "metrics.json"
    json_path.write_text(
        json.dumps({plan.profile: plan.metrics for plan in plans}, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


def write_trajectory_svg(output_dir: Path, bundle: AnalysisBundle, plans: list[CropPlan]) -> Path:
    width, height = 1080, 360
    left, right, top, bottom = 65, 20, 40, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    duration = max(bundle.video.duration, 1e-6)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - fraction) * plot_height
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        elements.append(f'<text x="12" y="{y + 4:.1f}" font-size="12" fill="#475569">{fraction:.2f}</text>')

    seen_shots: set[int] = set()
    for frame in bundle.frames:
        if frame.shot_id in seen_shots:
            continue
        seen_shots.add(frame.shot_id)
        x = left + frame.time / duration * plot_width
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#94a3b8" stroke-dasharray="4 5"/>')

    crop_axis = getattr(bundle, "crop_axis", "x")
    for index, plan in enumerate(plans):
        color = _PALETTE[index % len(_PALETTE)]
        points = []
        for point in plan.points:
            x = left + point.time / duration * plot_width
            active_center = point.center_y_norm if crop_axis == "y" else point.center_x_norm
            y = top + (1.0 - active_center) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.90"/>'
        )
        legend_x = left + index * 150
        legend_y = height - 20
        elements.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 22}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 28}" y="{legend_y}" font-size="12" fill="#0f172a">{html.escape(plan.profile)}</text>')

    direction = "vertical" if crop_axis == "y" else "horizontal" if crop_axis == "x" else "fixed"
    elements.append(f'<text x="{left}" y="20" font-size="16" font-weight="600" fill="#0f172a">Normalised {direction} viewport centre over time</text>')
    elements.append(f'<text x="{left + plot_width / 2 - 25}" y="{height - 2}" font-size="12" fill="#475569">time</text>')
    elements.append('</svg>')

    path = output_dir / "trajectories.svg"
    path.write_text("\n".join(elements), encoding="utf-8")
    return path


def write_report(
    output_dir: str | Path,
    *,
    bundle: AnalysisBundle,
    plans: list[CropPlan],
    rendered_paths: dict[str, Path],
    comparison_path: Path | None,
    ground_truth_path: Path | None,
    target_tracks: list[TargetTrack] | None = None,
    tracking_preview_path: Path | None = None,
    agent_trace_path: Path | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = write_metrics_files(root, plans)
    trajectory_path = write_trajectory_svg(root, bundle, plans)
    columns = _metric_columns(plans)

    shot_boundaries = []
    previous_shot = None
    for frame in bundle.frames:
        if frame.shot_id != previous_shot:
            shot_boundaries.append({"shot": frame.shot_id, "time": frame.time})
            previous_shot = frame.shot_id
    (root / "analysis_summary.json").write_text(
        json.dumps(
            {
                "source": bundle.video.path,
                "source_resolution": [bundle.video.width, bundle.video.height],
                "fps": bundle.video.fps,
                "duration": bundle.video.duration,
                "analysis_resolution": [bundle.analysis_width, bundle.analysis_height],
                "sample_fps": bundle.sample_fps,
                "sample_count": len(bundle.frames),
                "target_resolution": [bundle.target_width, bundle.target_height],
                "target_aspect_ratio": bundle.target_width / max(bundle.target_height, 1),
                "crop_axis": getattr(bundle, "crop_axis", "x"),
                "shot_boundaries": shot_boundaries,
                "active_speaker": bundle.performance.get("active_speaker", {}),
                "metric_warning": "Automatic metrics are proxy signals, not editorial ground truth.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    header_cells = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    metric_rows = []
    for plan in plans:
        cells = "".join(
            f"<td>{_format_metric(float(plan.metrics[column])) if column in plan.metrics else '—'}</td>"
            for column in columns
        )
        metric_rows.append(f"<tr><th>{html.escape(plan.profile)}</th>{cells}</tr>")

    video_cards = []
    for plan in plans:
        video_path = rendered_paths.get(plan.profile)
        if video_path is None:
            continue
        metrics = plan.metrics
        video_cards.append(
            f"""
            <article class="card">
              <h3>{html.escape(plan.profile)}</h3>
              <p>{html.escape(plan.description)}</p>
              <video controls muted loop preload="metadata" src="{html.escape(_relative(video_path, root))}"></video>
              <p class="small">Coverage {_format_metric(metrics.get('mean_shared_coverage', 0.0))} · stability {_format_metric(metrics.get('stability_score', 0.0))} · split {_format_metric(metrics.get('split_layout_fraction', 0.0))} · detail {_format_metric(metrics.get('context_detail_fraction', 0.0))}</p>
            </article>
            """
        )

    feature_notes = []
    if plans:
        consistency = plans[-1].implementation.get("consistent_framing", {})
        if isinstance(consistency, dict) and consistency.get("enabled"):
            feature_notes.append(
                f"Repeated-shot framing lock adjusted {int(consistency.get('locked_shots', 0))} shots."
            )
        correction_count = len(plans[-1].implementation.get("framing_corrections", []))
        if correction_count:
            feature_notes.append(f"Applied {correction_count} manual framing decision(s) from cached analysis.")
    speaker = bundle.performance.get("active_speaker", {})
    if isinstance(speaker, dict) and speaker.get("backend") not in {None, "disabled", "not_needed", "no_audio"}:
        feature_notes.append(
            f"Active-speaker evidence used {html.escape(str(speaker.get('backend')))} across "
            f"{int(speaker.get('speaking_frames', 0))} sampled frames."
        )
    feature_html = ""
    if feature_notes:
        feature_html = "<section><h2>Framing decisions</h2><p>" + " ".join(feature_notes) + "</p></section>"

    comparison_html = ""
    if comparison_path is not None:
        comparison_html = f"""
        <section>
          <h2>Side-by-side comparison</h2>
          <p>Play this first. All variants use the same input timeline and audio.</p>
          <video class="comparison" controls preload="metadata" src="{html.escape(_relative(comparison_path, root))}"></video>
        </section>
        """

    ground_truth_note = (
        f"Synthetic or supplied ground truth was found at {html.escape(str(ground_truth_path))}."
        if ground_truth_path
        else "No ground-truth annotation was found. Judge the outputs visually and treat the automatic scores as diagnostics."
    )

    agent_html = ""
    if agent_trace_path is not None and agent_trace_path.is_file():
        try:
            agent_payload = json.loads(agent_trace_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            agent_payload = {}
        rows = []
        for profile_name, trace in dict(agent_payload.get("profiles", {})).items():
            steps = list(trace.get("steps", [])) if isinstance(trace, dict) else []
            selected = int(trace.get("selected_round", 0)) if isinstance(trace, dict) else 0
            score = float(trace.get("selected_quality_score_0_100", 0.0)) if isinstance(trace, dict) else 0.0
            remaining = list(trace.get("remaining_issues", [])) if isinstance(trace, dict) else []
            issue_names = ", ".join(
                str(item.get("code", item)) if isinstance(item, dict) else str(item)
                for item in remaining
            ) or "none"
            rows.append(
                "<tr>"
                f"<th>{html.escape(str(profile_name))}</th>"
                f"<td>{len(steps)}</td>"
                f"<td>{selected}</td>"
                f"<td>{score:.2f}</td>"
                f"<td>{html.escape(issue_names)}</td>"
                "</tr>"
            )
        if rows:
            agent_html = f"""
            <section>
              <h2>Automatic crop review</h2>
              <p>The controller inspected OpenCV-derived coverage, text integrity, target retention, layout changes, and camera jerk. It changed only the settings tied to a diagnosed problem, then kept the best plan.</p>
              <div class="table-wrap"><table>
                <thead><tr><th>profile</th><th>plans tried</th><th>selected pass</th><th>quality score</th><th>remaining issues</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table></div>
              <p class="small"><a href="{html.escape(_relative(agent_trace_path, root))}">auto_fix_trace.json</a> records every observation, rule, parameter change, and rejected revision.</p>
            </section>
            """

    tracks = target_tracks or []
    target_html = ""
    if tracks:
        rows = []
        for track in tracks:
            rows.append(
                "<tr>"
                f"<th>{html.escape(track.target.label)}</th>"
                f"<td>{html.escape(track.target.kind)}</td>"
                f"<td>{html.escape(track.target.role)}</td>"
                f"<td>{_format_metric(track.metrics.get('visible_fraction', 0.0))}</td>"
                f"<td>{_format_metric(track.metrics.get('mean_visible_confidence', 0.0))}</td>"
                f"<td>{int(track.metrics.get('global_match_observations', 0.0))}</td>"
                f"<td>{_format_metric(track.metrics.get('mean_intent_priority', 1.0))}</td>"
                "</tr>"
            )
        preview = ""
        if tracking_preview_path is not None:
            preview = (
                '<video class="comparison" controls muted preload="metadata" '
                f'src="{html.escape(_relative(tracking_preview_path, root))}"></video>'
            )
        point_warning = ""
        has_visual_point = any(track.target.kind == "point" for track in tracks)
        if has_visual_point:
            point_warning = (
                '<div class="notice"><strong>Review point tracks:</strong> confidence measures '
                'visual similarity, not semantic identity. Small cursors can resemble icons or text. '
                'Inspect the preview and add a correction keyframe where the overlay drifts.</div>'
            )
        target_html = f"""
        <section>
          <h2>Selected target tracking</h2>
          {point_warning}
          {preview}
          <div class="table-wrap"><table>
            <thead><tr><th>target</th><th>kind</th><th>role</th><th>visible fraction</th><th>confidence</th><th>global matches</th><th>intent priority</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table></div>
          <p class="small"><a href="tracks.json">tracks.json</a> contains every sampled observation, confidence value, source method and reacquisition.</p>
        </section>
        """

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reframe Lab comparison</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
body {{ margin: 0; background: #f1f5f9; color: #0f172a; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1, h2, h3 {{ line-height: 1.2; }}
p {{ line-height: 1.55; }}
section {{ margin: 28px 0; }}
.notice {{ background: #fff7ed; border: 1px solid #fdba74; border-radius: 10px; padding: 14px 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 18px; }}
.card {{ background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(15,23,42,.05); }}
.card video {{ width: 100%; max-height: 640px; background: #000; border-radius: 8px; }}
.comparison {{ width: 100%; background: #000; border-radius: 10px; }}
.small {{ color: #475569; font-size: 0.88rem; }}
.table-wrap {{ overflow-x: auto; background: white; border: 1px solid #cbd5e1; border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
a {{ color: #1d4ed8; }}
img {{ max-width: 100%; background: white; border: 1px solid #cbd5e1; border-radius: 10px; }}
</style>
</head>
<body><main>
<h1>Reframe Lab comparison</h1>
<p><strong>Source:</strong> {html.escape(Path(bundle.video.path).name)} · {bundle.video.width}×{bundle.video.height} · {bundle.video.duration:.2f}s · output {bundle.target_width}×{bundle.target_height} · analysis at {bundle.sample_fps:.2f} fps</p>
<div class="notice"><strong>Interpretation:</strong> {html.escape(ground_truth_note)} A high coverage score can still represent a poor editorial choice, and a stable centre crop can score well on motion while missing the subject.</div>
{comparison_html}
{feature_html}
{agent_html}
{target_html}
<section>
  <h2>Individual outputs</h2>
  <div class="grid">{''.join(video_cards)}</div>
</section>
<section>
  <h2>Camera paths</h2>
  <p>The vertical axis is the normalised centre on the active crop axis. Flat lines are holds; slopes are viewport moves. Dashed vertical lines are detected shot boundaries.</p>
  <img src="{html.escape(_relative(trajectory_path, root))}" alt="Camera trajectories">
</section>
<section>
  <h2>Metrics</h2>
  <div class="table-wrap"><table><thead><tr><th>profile</th>{header_cells}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
  <p class="small">Shared coverage evaluates every profile against the same saliency, residual-motion, structure, text, and semantic-person evidence. Stability penalises fast or abrupt camera movement. Split and context-detail layouts preserve distant regions without reducing the whole source to a small inset. Proxy quality remains a debugging diagnostic.</p>
  <p><a href="{html.escape(_relative(csv_path, root))}">metrics.csv</a> · <a href="{html.escape(_relative(json_path, root))}">metrics.json</a> · <a href="analysis_summary.json">analysis_summary.json</a></p>
</section>
<section>
  <h2>Human review checklist</h2>
  <ol>
    <li>Watch once without looking at profile names. Note every moment where the crop hides the information needed to understand the shot.</li>
    <li>Watch again for camera behaviour: unnecessary pans, late tracking, oscillation, and motion across shot cuts.</li>
    <li>Check text readability and whether whole objects are preserved rather than merely their most salient point.</li>
    <li>Compare split, context-detail, and full-context layouts against the cropped alternatives. Check whether each panel remains readable and whether repeated content is useful.</li>
    <li>Record preferences by shot, not only for the entire video. Different content classes usually need different policies.</li>
  </ol>
</section>
<section>
  <h2>Implementation notes</h2>
  <p>See <code>docs/TECHNICAL_REPORT.md</code> in the project root. Each JSON crop plan records the exact enabled signals and planner settings.</p>
</section>
</main></body></html>
"""

    report_path = root / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
