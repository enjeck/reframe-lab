from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .corrections import FramingCorrection
from .models import AnalysisBundle, CropPlan, Profile
from .planner import build_plan
from .targets import TargetTrack

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class AutoFixPolicy:
    """Quality gates for the deterministic crop-review loop.

    The values are deliberately conservative. The loop should change a plan only when
    the diagnostics point to a clear failure, not merely because another crop has a
    slightly higher proxy score.
    """

    minimum_shared_p10: float = 0.48
    minimum_text_integrity: float = 0.90
    minimum_text_visible_anywhere: float = 0.985
    maximum_text_cut_fraction: float = 0.12
    minimum_stability: float = 0.52
    maximum_mean_jerk: float = 1.75
    maximum_p95_jerk: float = 5.0
    minimum_target_in_crop: float = 0.95
    minimum_target_p10_context: float = 0.74
    maximum_layout_switches_per_minute: float = 10.0
    minimum_confidence_for_autonomy: float = 0.24


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: float
    message: str
    action: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AutoFixResult:
    plan: CropPlan
    profile: Profile
    trace: dict[str, object]


def _metric(metrics: dict[str, float], name: str, default: float) -> float:
    value = metrics.get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def inspect_plan(plan: CropPlan, policy: AutoFixPolicy | None = None) -> list[QualityIssue]:
    """Return actionable problems found in a crop plan.

    These checks use measurements generated from OpenCV evidence and the planned camera
    path. They do not use an LLM and they never change a plan without an explicit rule.
    """

    rules = policy or AutoFixPolicy()
    metrics = plan.metrics
    issues: list[QualityIssue] = []

    shared_p10 = _metric(metrics, "p10_shared_coverage", 1.0)
    if shared_p10 < rules.minimum_shared_p10:
        severity = np.clip(
            (rules.minimum_shared_p10 - shared_p10) / max(rules.minimum_shared_p10, 1e-6),
            0.0,
            1.0,
        )
        issues.append(
            QualityIssue(
                code="low_content_coverage",
                severity=float(severity),
                message=f"Important evidence falls outside the view on weak frames ({shared_p10:.3f}).",
                action="make unsafe crops switch to split or context-detail sooner",
            )
        )

    trusted_text = _metric(metrics, "trusted_text_frame_fraction", 0.0)
    if trusted_text >= 0.04:
        integrity = _metric(metrics, "mean_text_integrity_in_detail", 1.0)
        visible = _metric(metrics, "text_visible_anywhere_fraction", 1.0)
        cut_fraction = _metric(metrics, "text_region_cut_fraction", 0.0)
        if (
            integrity < rules.minimum_text_integrity
            or visible < rules.minimum_text_visible_anywhere
            or cut_fraction > rules.maximum_text_cut_fraction
        ):
            deficit = max(
                rules.minimum_text_integrity - integrity,
                rules.minimum_text_visible_anywhere - visible,
                cut_fraction - rules.maximum_text_cut_fraction,
                0.0,
            )
            issues.append(
                QualityIssue(
                    code="text_clipped",
                    severity=float(np.clip(deficit / 0.30, 0.0, 1.0)),
                    message=(
                        "Detected text is clipped or hidden "
                        f"(integrity {integrity:.3f}, visible {visible:.3f})."
                    ),
                    action="raise complete-text priority and allow text-driven layouts earlier",
                )
            )

    target_visible = _metric(metrics, "target_visible_observations", 0.0)
    if target_visible > 0.0:
        target_inside = _metric(metrics, "target_in_crop_fraction", 1.0)
        target_p10 = _metric(metrics, "p10_target_context_coverage", 1.0)
        if (
            target_inside < rules.minimum_target_in_crop
            or target_p10 < rules.minimum_target_p10_context
        ):
            deficit = max(
                rules.minimum_target_in_crop - target_inside,
                rules.minimum_target_p10_context - target_p10,
                0.0,
            )
            issues.append(
                QualityIssue(
                    code="target_lost",
                    severity=float(np.clip(deficit / 0.35, 0.0, 1.0)),
                    message=(
                        "The selected target or its context leaves the detail view "
                        f"(inside {target_inside:.3f}, p10 context {target_p10:.3f})."
                    ),
                    action="increase target priority, context margin, and safe-layout sensitivity",
                )
            )

    stability = _metric(metrics, "stability_score", 1.0)
    mean_jerk = _metric(metrics, "mean_jerk_viewports_per_second3", 0.0)
    p95_jerk = _metric(metrics, "p95_jerk_viewports_per_second3", 0.0)
    if (
        stability < rules.minimum_stability
        or mean_jerk > rules.maximum_mean_jerk
        or p95_jerk > rules.maximum_p95_jerk
    ):
        severity = max(
            (rules.minimum_stability - stability) / max(rules.minimum_stability, 1e-6),
            (mean_jerk - rules.maximum_mean_jerk) / max(rules.maximum_mean_jerk, 1e-6),
            (p95_jerk - rules.maximum_p95_jerk) / max(rules.maximum_p95_jerk, 1e-6),
            0.0,
        )
        issues.append(
            QualityIssue(
                code="camera_too_abrupt",
                severity=float(np.clip(severity, 0.0, 1.0)),
                message=(
                    "The virtual camera changes too abruptly "
                    f"(stability {stability:.3f}, p95 jerk {p95_jerk:.3f})."
                ),
                action="tighten speed and jerk limits and give the planner more lookahead pressure",
            )
        )

    switches_per_minute = _metric(metrics, "layout_switches_per_minute", 0.0)
    if switches_per_minute > rules.maximum_layout_switches_per_minute:
        issues.append(
            QualityIssue(
                code="layout_churn",
                severity=float(
                    np.clip(
                        (switches_per_minute - rules.maximum_layout_switches_per_minute)
                        / max(rules.maximum_layout_switches_per_minute, 1e-6),
                        0.0,
                        1.0,
                    )
                ),
                message=f"The composition changes too often ({switches_per_minute:.2f} per minute).",
                action="hold layouts longer and use slower layout transitions",
            )
        )

    confidence = _metric(metrics, "mean_confidence", 1.0)
    if confidence < rules.minimum_confidence_for_autonomy:
        issues.append(
            QualityIssue(
                code="low_visual_confidence",
                severity=float(
                    np.clip(
                        (rules.minimum_confidence_for_autonomy - confidence)
                        / max(rules.minimum_confidence_for_autonomy, 1e-6),
                        0.0,
                        1.0,
                    )
                ),
                message=f"The crop evidence is ambiguous (mean confidence {confidence:.3f}).",
                action="keep the safest plan and expose the low-confidence result for human review",
            )
        )

    return issues


def plan_quality_score(plan: CropPlan) -> float:
    """Comparable 0-100 score used only to choose between auto-fix revisions."""

    metrics = plan.metrics
    components: list[tuple[float, float]] = [
        (_metric(metrics, "mean_shared_coverage", 0.0), 0.21),
        (_metric(metrics, "p10_shared_coverage", 0.0), 0.19),
        (_metric(metrics, "stability_score", 0.0), 0.20),
        (_metric(metrics, "mean_confidence", 0.0), 0.07),
    ]

    trusted_text = _metric(metrics, "trusted_text_frame_fraction", 0.0)
    if trusted_text >= 0.04:
        components.extend(
            [
                (_metric(metrics, "mean_text_integrity_in_detail", 0.0), 0.17),
                (_metric(metrics, "text_visible_anywhere_fraction", 0.0), 0.08),
                (1.0 - _metric(metrics, "text_region_cut_fraction", 1.0), 0.08),
            ]
        )

    if _metric(metrics, "target_visible_observations", 0.0) > 0.0:
        components.extend(
            [
                (_metric(metrics, "target_in_crop_fraction", 0.0), 0.13),
                (_metric(metrics, "p10_target_context_coverage", 0.0), 0.10),
            ]
        )

    total_weight = sum(weight for _, weight in components)
    base = sum(np.clip(value, 0.0, 1.0) * weight for value, weight in components)
    base /= max(total_weight, 1e-8)

    mean_jerk = _metric(metrics, "mean_jerk_viewports_per_second3", 0.0)
    switches = _metric(metrics, "layout_switches_per_minute", 0.0)
    penalty = min(0.12, max(0.0, mean_jerk - 1.0) * 0.014)
    penalty += min(0.08, max(0.0, switches - 8.0) * 0.004)
    return float(np.clip(100.0 * (base - penalty), 0.0, 100.0))


def _normalised_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {name: max(0.0, float(value)) for name, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 1e-8:
        return cleaned
    return {name: value / total for name, value in cleaned.items()}


def revise_profile(profile: Profile, issues: list[QualityIssue], round_index: int) -> Profile:
    """Create one conservative profile revision from a set of diagnosed failures."""

    codes = {issue.code for issue in issues}
    weights = dict(profile.signal_weights)
    changes: dict[str, object] = {}

    if "text_clipped" in codes:
        weights["text_complete"] = max(weights.get("text_complete", 0.0) * 1.38, 0.38)
        weights["text"] = max(weights.get("text", 0.0) * 1.12, 0.08)
        changes.update(
            text_integrity_threshold=min(0.985, profile.text_integrity_threshold + 0.035),
            text_split_min_regions=max(2, profile.text_split_min_regions - 1),
            context_detail_fraction=min(0.76, profile.context_detail_fraction + 0.035),
            fallback_coverage_threshold=min(0.68, profile.fallback_coverage_threshold + 0.035),
            fallback_static_coverage_threshold=min(
                0.78, profile.fallback_static_coverage_threshold + 0.035
            ),
        )

    if "low_content_coverage" in codes:
        changes.update(
            fallback_coverage_threshold=min(
                0.72,
                max(
                    float(changes.get("fallback_coverage_threshold", 0.0)),
                    profile.fallback_coverage_threshold + 0.045,
                ),
            ),
            fallback_static_coverage_threshold=min(
                0.82,
                max(
                    float(changes.get("fallback_static_coverage_threshold", 0.0)),
                    profile.fallback_static_coverage_threshold + 0.045,
                ),
            ),
            context_detail_fraction=min(
                0.78,
                max(
                    float(changes.get("context_detail_fraction", 0.0)),
                    profile.context_detail_fraction + 0.04,
                ),
            ),
            semantic_layouts=True,
        )

    if "target_lost" in codes:
        changes.update(
            target_weight=min(0.94, max(profile.target_weight + 0.08, 0.82)),
            target_context_fraction=min(0.76, profile.target_context_fraction + 0.06),
            target_min_coverage=min(0.94, profile.target_min_coverage + 0.04),
            fallback_coverage_threshold=min(
                0.72,
                max(
                    float(changes.get("fallback_coverage_threshold", 0.0)),
                    profile.fallback_coverage_threshold + 0.04,
                ),
            ),
        )

    if "camera_too_abrupt" in codes:
        scale = 0.82 if round_index == 0 else 0.74
        changes.update(
            movement_penalty=profile.movement_penalty * 1.20,
            dead_zone_fraction=min(0.26, profile.dead_zone_fraction + 0.025),
            max_speed_crop_widths_per_second=max(
                0.55, profile.max_speed_crop_widths_per_second * scale
            ),
            max_acceleration_crop_widths_per_second2=max(
                1.4, profile.max_acceleration_crop_widths_per_second2 * scale
            ),
            max_jerk_crop_widths_per_second3=max(
                4.0, profile.max_jerk_crop_widths_per_second3 * scale
            ),
            minimum_jerk_jerk_weight=profile.minimum_jerk_jerk_weight * 1.55,
        )

    if "layout_churn" in codes:
        changes.update(
            minimum_layout_seconds=min(2.8, profile.minimum_layout_seconds + 0.45),
            layout_transition_seconds=min(1.15, profile.layout_transition_seconds + 0.18),
        )

    # Ambiguous evidence is deliberately not "fixed" by inventing confidence. The safest
    # action is to keep the best plan and make the trace visible to the user.
    if weights != profile.signal_weights:
        changes["signal_weights"] = _normalised_weights(weights)
    if not changes:
        return profile
    return replace(profile, **changes)


def _profile_snapshot(profile: Profile) -> dict[str, object]:
    return {
        "signal_weights": dict(profile.signal_weights),
        "movement_penalty": profile.movement_penalty,
        "dead_zone_fraction": profile.dead_zone_fraction,
        "max_speed_viewports_per_second": profile.max_speed_crop_widths_per_second,
        "max_acceleration_viewports_per_second2": (
            profile.max_acceleration_crop_widths_per_second2
        ),
        "max_jerk_viewports_per_second3": profile.max_jerk_crop_widths_per_second3,
        "minimum_layout_seconds": profile.minimum_layout_seconds,
        "layout_transition_seconds": profile.layout_transition_seconds,
        "fallback_coverage_threshold": profile.fallback_coverage_threshold,
        "fallback_static_coverage_threshold": profile.fallback_static_coverage_threshold,
        "text_integrity_threshold": profile.text_integrity_threshold,
        "text_split_min_regions": profile.text_split_min_regions,
        "target_weight": profile.target_weight,
        "target_context_fraction": profile.target_context_fraction,
        "target_min_coverage": profile.target_min_coverage,
    }


def auto_fix_plan(
    bundle: AnalysisBundle,
    profile: Profile,
    target_tracks: list[TargetTrack] | None = None,
    framing_corrections: list[FramingCorrection] | None = None,
    *,
    consistent_framing: bool = True,
    max_rounds: int = 2,
    policy: AutoFixPolicy | None = None,
    revision_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> AutoFixResult:
    """Build, inspect, and conservatively revise a plan.

    The loop is deterministic. OpenCV-derived evidence creates the plan, the quality
    gates diagnose specific failures, and a bounded rule set changes only the relevant
    planner settings. The best-scoring revision is retained even when no candidate
    clears every gate.
    """

    tracks = target_tracks or []
    rules = policy or AutoFixPolicy()
    rounds = max(0, int(max_rounds))
    current_profile = profile
    candidates: list[tuple[CropPlan, Profile, float, list[QualityIssue]]] = []
    steps: list[dict[str, object]] = []
    output_root = Path(revision_dir) if revision_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    for round_index in range(rounds + 1):
        if progress:
            label = "initial crop plan" if round_index == 0 else f"auto-fix pass {round_index}"
            progress(f"Reviewing {profile.name}: {label}")
        if framing_corrections or not consistent_framing:
            plan = build_plan(
                bundle,
                current_profile,
                tracks,
                framing_corrections,
                consistent_framing=consistent_framing,
            )
        else:
            # Preserve the original call shape for simple integrations and tests.
            plan = build_plan(bundle, current_profile, tracks)
        issues = inspect_plan(plan, rules)
        score = plan_quality_score(plan)
        candidates.append((plan, current_profile, score, issues))

        revision_path = None
        if output_root is not None:
            revision_path = output_root / f"round_{round_index}.json"
            plan.save(revision_path)

        step = {
            "round": round_index,
            "quality_score_0_100": score,
            "passed": not issues,
            "issues": [issue.to_dict() for issue in issues],
            "metrics": dict(plan.metrics),
            "profile": _profile_snapshot(current_profile),
            "plan": str(revision_path) if revision_path is not None else None,
        }
        steps.append(step)
        if not issues or round_index >= rounds:
            break
        next_profile = revise_profile(current_profile, issues, round_index)
        if next_profile == current_profile:
            break
        current_profile = next_profile

    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index][2],
            -len(candidates[index][3]),
            -index,
        ),
    )
    best_plan, best_profile, best_score, best_issues = candidates[best_index]
    best_plan.implementation["auto_fix"] = {
        "enabled": True,
        "policy": asdict(rules),
        "rounds_attempted": len(candidates),
        "selected_round": best_index,
        "quality_score_0_100": best_score,
        "remaining_issues": [issue.code for issue in best_issues],
    }
    trace = {
        "version": 1,
        "profile": profile.name,
        "controller": "deterministic_quality_gate",
        "max_rounds": rounds,
        "selected_round": best_index,
        "selected_quality_score_0_100": best_score,
        "passed_all_gates": not best_issues,
        "remaining_issues": [issue.to_dict() for issue in best_issues],
        "steps": steps,
    }
    return AutoFixResult(plan=best_plan, profile=best_profile, trace=trace)


def save_auto_fix_trace(path: str | Path, traces: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    return target
