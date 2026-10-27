from __future__ import annotations

from reframelab.models import CropPlan, PlanPoint


def test_crop_plan_json_round_trip(tmp_path) -> None:
    plan = CropPlan(
        profile="test",
        description="round trip",
        source="input.mp4",
        source_width=1920,
        source_height=1080,
        source_fps=30.0,
        target_width=1080,
        target_height=1920,
        sample_fps=5.0,
        points=[PlanPoint(0, 0.0, 0, 0.5, "hold", 0.8, 0.7)],
        metrics={"x": 1.0},
    )
    path = tmp_path / "plan.json"
    plan.save(path)
    loaded = CropPlan.load(path)
    assert loaded.to_dict() == plan.to_dict()
