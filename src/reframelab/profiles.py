from __future__ import annotations

from .models import Profile

_HYBRID_WEIGHTS = {
    "saliency": 0.12,
    "motion": 0.15,
    "structure": 0.08,
    "text": 0.08,
    "text_complete": 0.28,
    "people": 0.20,
    "speaker": 0.15,
}


PROFILES: dict[str, Profile] = {
    "center": Profile(
        name="center",
        description="Fixed centre crop. This is the simple baseline.",
    ),
    "saliency_raw": Profile(
        name="saliency_raw",
        description="Spectral-residual visual saliency, selected independently per frame.",
        signal_weights={"saliency": 1.0},
    ),
    "motion_raw": Profile(
        name="motion_raw",
        description="Camera-compensated residual motion, selected independently per frame.",
        signal_weights={"motion": 1.0},
    ),
    "structure_raw": Profile(
        name="structure_raw",
        description="Edge and structural detail, selected independently per frame.",
        signal_weights={"structure": 1.0},
    ),
    "hybrid_raw": Profile(
        name="hybrid_raw",
        description=(
            "Saliency, residual motion, structure and text-like regions without temporal planning."
        ),
        signal_weights=dict(_HYBRID_WEIGHTS),
    ),
    "hybrid_temporal": Profile(
        name="hybrid_temporal",
        description=(
            "The hybrid signal model plus shot-aware path optimisation and camera stabilisation."
        ),
        signal_weights=dict(_HYBRID_WEIGHTS),
        temporal=True,
        stabilize=True,
        movement_penalty=0.08,
        max_transition_crop_fraction=0.75,
        dead_zone_fraction=0.08,
        max_speed_crop_widths_per_second=1.90,
        max_acceleration_crop_widths_per_second2=5.5,
        max_jerk_crop_widths_per_second3=20.0,
        layout_transition_seconds=0.52,
        minimum_layout_seconds=1.00,
    ),
    "hybrid_safe": Profile(
        name="hybrid_safe",
        description="Hybrid temporal framing with a full-context fallback when a crop is unsafe.",
        signal_weights=dict(_HYBRID_WEIGHTS),
        temporal=True,
        stabilize=True,
        allow_fallback=True,
        movement_penalty=0.085,
        max_transition_crop_fraction=0.75,
        dead_zone_fraction=0.08,
        max_speed_crop_widths_per_second=1.85,
        max_acceleration_crop_widths_per_second2=5.2,
        max_jerk_crop_widths_per_second3=19.0,
        layout_transition_seconds=0.56,
        minimum_layout_seconds=1.10,
        fallback_coverage_threshold=0.46,
        fallback_static_coverage_threshold=0.62,
        fallback_motion_reliability_threshold=0.12,
        semantic_layouts=True,
        context_detail_fraction=0.68,
        text_integrity_threshold=0.90,
        text_split_min_regions=4,
    ),
    "target_naive": Profile(
        name="target_naive",
        description=(
            "Direct user-target framing without temporal planning; useful as a tracking baseline."
        ),
        signal_weights=dict(_HYBRID_WEIGHTS),
        target_weight=1.0,
        target_only=True,
        target_context_fraction=0.50,
    ),
    "target_guided": Profile(
        name="target_guided",
        description=(
            "Automatic framing guided by selected points or objects, with a stable virtual camera."
        ),
        signal_weights=dict(_HYBRID_WEIGHTS),
        temporal=True,
        stabilize=True,
        movement_penalty=0.075,
        max_transition_crop_fraction=0.80,
        dead_zone_fraction=0.16,
        max_speed_crop_widths_per_second=1.65,
        max_acceleration_crop_widths_per_second2=4.5,
        max_jerk_crop_widths_per_second3=16.0,
        layout_transition_seconds=0.52,
        minimum_layout_seconds=1.00,
        target_weight=0.78,
        target_context_fraction=0.58,
        target_min_coverage=0.82,
    ),
    "target_guided_safe": Profile(
        name="target_guided_safe",
        description=(
            "Target-guided stable framing with full-context fallback when selected content cannot fit."
        ),
        signal_weights=dict(_HYBRID_WEIGHTS),
        temporal=True,
        stabilize=True,
        allow_fallback=True,
        movement_penalty=0.08,
        max_transition_crop_fraction=0.80,
        dead_zone_fraction=0.16,
        max_speed_crop_widths_per_second=1.60,
        max_acceleration_crop_widths_per_second2=4.3,
        max_jerk_crop_widths_per_second3=15.0,
        layout_transition_seconds=0.56,
        minimum_layout_seconds=1.10,
        fallback_coverage_threshold=0.43,
        fallback_static_coverage_threshold=0.58,
        fallback_motion_reliability_threshold=0.12,
        target_weight=0.80,
        target_context_fraction=0.58,
        target_min_coverage=0.82,
        semantic_layouts=True,
        context_detail_fraction=0.68,
        text_integrity_threshold=0.90,
        text_split_min_regions=4,
    ),
}


EVALUATION_WEIGHTS: dict[str, float] = {
    "saliency": 0.10,
    "motion": 0.14,
    "structure": 0.08,
    "text": 0.10,
    "text_complete": 0.29,
    "people": 0.20,
    "speaker": 0.15,
}

DEFAULT_DEMO_PROFILES = [
    "center",
    "hybrid_safe",
]

TARGET_DEMO_PROFILES = [
    "center",
    "hybrid_safe",
    "target_guided_safe",
]

# Every user-facing variant. The web interfaces render all of them by default so a
# person can pick the edit they like instead of trusting one automatic choice.
VARIANT_PROFILES = [
    "hybrid_safe",
    "hybrid_temporal",
    "center",
]

TARGET_VARIANT_PROFILES = [
    "target_guided_safe",
    "target_guided",
    "hybrid_safe",
    "hybrid_temporal",
    "center",
]

RECOMMENDED_PROFILES = {"hybrid_safe", "target_guided_safe"}

VARIANT_NAMES: dict[str, tuple[str, str]] = {
    "hybrid_safe": ("Smart", "Crops, splits or widens"),
    "hybrid_temporal": ("Follow", "One moving crop"),
    "center": ("Centre", "Fixed middle crop"),
    "target_guided_safe": ("Smart target", "Your marks, with splits"),
    "target_guided": ("Target", "Your marks, one crop"),
}


def variant_name(name: str) -> tuple[str, str]:
    return VARIANT_NAMES.get(name, (name.replace("_", " ").capitalize(), ""))


# Useful for development reports, but deliberately kept out of the normal web UI.
DEVELOPMENT_DEMO_PROFILES = [
    "center",
    "saliency_raw",
    "motion_raw",
    "structure_raw",
    "hybrid_raw",
    "hybrid_temporal",
    "hybrid_safe",
]

DEVELOPMENT_TARGET_PROFILES = [
    "center",
    "hybrid_safe",
    "target_naive",
    "target_guided",
    "target_guided_safe",
]


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Available profiles: {choices}") from exc
