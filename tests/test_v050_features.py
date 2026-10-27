from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from reframelab.models import (
    AnalysisBundle,
    FrameEvidence,
    PlanPoint,
    Profile,
    SemanticRegion,
    VideoInfo,
)
from reframelab.planner import build_plan
from reframelab.renderer import render_frame
from reframelab.review import build_tracking_review
from reframelab.semantic import PersonDetector
from reframelab.targets import (
    TargetAnchor,
    TargetObservation,
    TargetSpec,
    TargetTrack,
)
from reframelab.tracking import _create_learned_tracker, _learned_tracker_candidate, _TrackerState


def _semantic_bundle() -> AnalysisBundle:
    positions = np.array([0, 50, 100], dtype=np.int32)
    coverage = np.array([0.24, 0.28, 0.24], dtype=np.float32)
    regions = [
        SemanticRegion("person", 0.04, 0.08, 0.24, 0.82, 0.95),
        SemanticRegion("person", 0.72, 0.08, 0.24, 0.82, 0.93),
    ]
    frames = [
        FrameEvidence(
            frame_index=index * 10,
            time=index * 0.4,
            shot_id=0,
            shot_score=0.0,
            x_positions=positions,
            crop_width=100,
            signal_coverage={"saliency": coverage, "people": coverage},
            signal_centering={
                "saliency": np.ones(3, dtype=np.float32),
                "people": np.ones(3, dtype=np.float32),
            },
            signal_boundary={
                "saliency": np.zeros(3, dtype=np.float32),
                "people": np.zeros(3, dtype=np.float32),
            },
            signal_reliability={"saliency": 1.0, "people": 1.0},
            evaluation_score=coverage.copy(),
            evaluation_coverage=coverage.copy(),
            semantic_regions=regions,
            semantic_backend="opencv_dnn_yunet",
        )
        for index in range(4)
    ]
    return AnalysisBundle(
        VideoInfo("people.mp4", 300, 200, 25.0, 100, 4.0),
        200,
        200,
        100,
        200,
        2.5,
        frames,
    )


def test_semantic_multi_person_scene_uses_stacked_split() -> None:
    profile = Profile(
        name="semantic_safe",
        description="test",
        signal_weights={"saliency": 0.5, "people": 0.5},
        temporal=True,
        stabilize=True,
        allow_fallback=True,
        semantic_layouts=True,
        fallback_coverage_threshold=0.5,
    )
    plan = build_plan(_semantic_bundle(), profile)
    assert {point.mode for point in plan.points} == {"split"}
    assert all(point.secondary_center_x_norm is not None for point in plan.points)
    assert plan.metrics["split_layout_fraction"] == pytest.approx(1.0)
    assert plan.implementation["semantic_backends"] == ["opencv_dnn_yunet"]


def test_context_detail_and_split_rendering_preserve_distinct_regions() -> None:
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frame[:, :120] = (20, 20, 220)
    frame[:, 120:] = (20, 220, 20)
    context = render_frame(
        frame,
        PlanPoint(0, 0.0, 0, 0.75, "context_detail", 1.0, 1.0),
        120,
        240,
    )
    assert context.shape == (240, 120, 3)
    # The large detail favours the right-hand region while the overview contains both.
    assert float(context[:140, :, 1].mean()) > float(context[:140, :, 2].mean())
    assert float(context[175:, :, 1].mean()) > 20.0
    assert float(context[175:, :, 2].mean()) > 20.0

    split = render_frame(
        frame,
        PlanPoint(
            0,
            0.0,
            0,
            0.25,
            "split",
            1.0,
            1.0,
            secondary_center_x_norm=0.75,
        ),
        120,
        240,
    )
    assert float(split[:110, :, 2].mean()) > float(split[:110, :, 1].mean())
    assert float(split[130:, :, 1].mean()) > float(split[130:, :, 2].mean())


def test_person_detector_exposes_multiple_semantic_regions(monkeypatch) -> None:
    detector = PersonDetector(allow_download=False)

    class FakeYuNet:
        def setInputSize(self, size) -> None:  # noqa: N802
            self.size = size

        def detect(self, frame):
            detections = np.array(
                [
                    [10, 10, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
                    [70, 10, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90],
                ],
                dtype=np.float32,
            )
            return 2, detections

    detector._yunet = FakeYuNet()
    detector._haar = None
    detector.backend = "opencv_dnn_yunet"
    regions = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(regions) == 2
    assert regions[0].center_x < regions[1].center_x


def test_person_detector_merges_face_metadata_without_mutating_frozen_region(monkeypatch) -> None:
    detector = PersonDetector.__new__(PersonDetector)
    body = SemanticRegion(
        kind="person",
        x=0.10,
        y=0.05,
        width=0.50,
        height=0.90,
        score=0.82,
    )
    face = SemanticRegion(
        kind="person",
        x=0.20,
        y=0.10,
        width=0.20,
        height=0.55,
        score=0.94,
        face_x=0.24,
        face_y=0.12,
        face_width=0.10,
        face_height=0.14,
    )

    class FakePersonDetector:
        def detect(self, frame):
            del frame
            return [body]

    detector._person = FakePersonDetector()
    monkeypatch.setattr(detector, "_face_regions", lambda frame: [face])

    regions = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(regions) == 1
    assert body.face_width == 0.0
    assert regions[0] is not body
    assert regions[0].face_x == pytest.approx(face.face_x)
    assert regions[0].face_width == pytest.approx(face.face_width)
    assert regions[0].score == pytest.approx(face.score)


def test_person_detector_falls_back_when_model_rejects_batches(monkeypatch) -> None:
    detector = PersonDetector.__new__(PersonDetector)
    detector.progress = None
    detector._person_batch_disabled = False
    detector._person_batch_failures = 0
    detector._person_frame_failures = 0
    detector._face_failures = 0
    detector._face_fallback_uses = 0
    detector._warnings_emitted = set()
    detector._runtime_warnings = []
    detector._last_person_error = None
    detector._last_face_error = None
    body = SemanticRegion("person", 0.1, 0.05, 0.5, 0.9, 0.82)
    face = SemanticRegion(
        "person", 0.2, 0.1, 0.2, 0.55, 0.94,
        face_x=0.24, face_y=0.12, face_width=0.10, face_height=0.14,
    )

    class FixedBatchPerson:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        def detect_batch(self, frames):
            del frames
            self.batch_calls += 1
            raise RuntimeError("model input has fixed batch size 1")

        def detect(self, frame):
            del frame
            self.single_calls += 1
            return [body]

    backend = FixedBatchPerson()
    detector._person = backend
    monkeypatch.setattr(detector, "_face_regions", lambda frame: [face])

    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(3)]
    first = detector.detect_batch(frames)
    second = detector.detect_batch(frames)

    assert backend.batch_calls == 1
    assert backend.single_calls == 6
    assert detector._person_batch_disabled is True
    assert detector._person_batch_failures == 1
    assert len(first) == len(second) == 3
    assert all(regions[0].face_width == pytest.approx(face.face_width) for regions in first)
    assert any("single-frame" in message for message in detector.drain_runtime_warnings())


def test_yunet_runtime_failure_keeps_person_regions() -> None:
    detector = PersonDetector.__new__(PersonDetector)
    detector.progress = None
    detector._person_batch_disabled = False
    detector._person_batch_failures = 0
    detector._person_frame_failures = 0
    detector._face_failures = 0
    detector._face_fallback_uses = 0
    detector._warnings_emitted = set()
    detector._runtime_warnings = []
    detector._last_person_error = None
    detector._last_face_error = None
    detector._haar = None
    detector._haar_fallback = None
    body = SemanticRegion("person", 0.1, 0.05, 0.5, 0.9, 0.82)

    class PersonBackend:
        def detect(self, frame):
            del frame
            return [body]

    class BrokenYuNet:
        def setInputSize(self, size):  # noqa: N802
            del size

        def detect(self, frame):
            del frame
            raise RuntimeError("face graph failed")

    detector._person = PersonBackend()
    detector._yunet = BrokenYuNet()
    detector.backend = "opencv_dnn_mediapipe_person+opencv_dnn_yunet"

    regions = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert regions == [body]
    assert detector._face_failures == 1
    assert "face graph failed" in detector._last_face_error


def test_yunet_discards_non_finite_and_invalid_face_boxes() -> None:
    detector = PersonDetector.__new__(PersonDetector)
    detector.progress = None
    detector._person = None
    detector._person_batch_disabled = False
    detector._person_batch_failures = 0
    detector._person_frame_failures = 0
    detector._face_failures = 0
    detector._face_fallback_uses = 0
    detector._warnings_emitted = set()
    detector._runtime_warnings = []
    detector._last_person_error = None
    detector._last_face_error = None
    detector._haar = None
    detector._haar_fallback = None

    class MixedYuNet:
        def setInputSize(self, size):  # noqa: N802
            del size

        def detect(self, frame):
            del frame
            return 3, np.asarray(
                [
                    [np.nan, 10, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
                    [20, 20, -4, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90],
                    [70, 10, 40, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.88],
                ],
                dtype=np.float32,
            )

    detector._yunet = MixedYuNet()
    detector.backend = "opencv_dnn_yunet"
    regions = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(regions) == 1
    assert 0.0 <= regions[0].face_x < 1.0
    assert 0.0 < regions[0].face_width <= 1.0


def test_person_merge_sanitizes_malformed_semantic_regions() -> None:
    valid = SemanticRegion("person", -0.1, 0.05, 0.7, 1.1, 1.4)
    invalid = SemanticRegion("person", np.nan, 0.1, 0.2, 0.4, 0.8)

    regions = PersonDetector._merge_regions([valid, invalid], [])

    assert len(regions) == 1
    assert regions[0].x == 0.0
    assert regions[0].height <= 1.0
    assert regions[0].score == 1.0


def test_person_runtime_diagnostics_reset_between_persistent_jobs() -> None:
    detector = PersonDetector.__new__(PersonDetector)
    detector._person_batch_disabled = True
    detector._person_batch_failures = 2
    detector._person_frame_failures = 3
    detector._face_failures = 4
    detector._face_fallback_uses = 5
    detector._warnings_emitted = {"warning"}
    detector._runtime_warnings = ["old warning"]
    detector._last_person_error = "old person error"
    detector._last_face_error = "old face error"

    detector.reset_runtime_state()

    assert detector._person_batch_disabled is True
    assert detector._person_batch_failures == 0
    assert detector._person_frame_failures == 0
    assert detector._face_failures == 0
    assert detector._face_fallback_uses == 0
    assert detector._runtime_warnings == []
    assert detector._last_person_error is None
    assert detector._last_face_error is None


def test_review_flags_ambiguous_learned_observation() -> None:
    target = TargetSpec(
        "cursor",
        "Cursor",
        "point",
        "follow",
        (TargetAnchor(0.0, 0.2, 0.2),),
    )
    track = TargetTrack(
        target,
        [
            TargetObservation(0, 0.0, True, 1.0, 0.2, 0.2, 0.01, 0.01),
            TargetObservation(
                1,
                0.5,
                True,
                0.62,
                0.75,
                0.2,
                0.01,
                0.01,
                source="vittrack",
                needs_review=True,
                uncertainty=0.7,
            ),
        ],
    )
    review = build_tracking_review([track])
    assert review["review_required"] is True
    assert review["points"][0]["target_id"] == "cursor"


def test_opencv_vittrack_backend_reports_model_score(monkeypatch, tmp_path: Path) -> None:
    class FakeTracker:
        def init(self, frame, bbox):
            self.bbox = bbox
            return True

        def update(self, frame):
            return True, self.bbox

        def getTrackingScore(self):  # noqa: N802
            return 0.81

    monkeypatch.setattr(cv, "TrackerVit_create", lambda params: FakeTracker())
    model = tmp_path / "vittrack.onnx"
    model.write_bytes(b"fixture")
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    tracker = _create_learned_tracker(frame, (10, 10, 20, 20), model)
    assert tracker is not None
    target = TargetSpec(
        "object",
        "Object",
        "box",
        "follow",
        (TargetAnchor(0.0, 0.1, 0.1, 0.2, 0.2),),
    )
    state = _TrackerState(
        target=target,
        templates=[],
        anchor_boxes=[],
        kalman=cv.KalmanFilter(4, 2),
        learned_tracker=tracker,
    )
    candidate = _learned_tracker_candidate(state, frame)
    assert candidate is not None
    assert candidate.source == "vittrack"
    assert candidate.confidence == pytest.approx(0.81)
    assert candidate.uncertainty == pytest.approx(0.19)


def test_blazepose_anchor_generator_matches_model_contract() -> None:
    from reframelab.semantic import _blazepose_anchors

    anchors = _blazepose_anchors()
    assert anchors.shape == (2254, 2)
    assert anchors[0].tolist() == pytest.approx([1 / 56, 1 / 56])
    assert anchors[0].tolist() == pytest.approx(anchors[1].tolist())
    assert np.all((anchors > 0.0) & (anchors < 1.0))


def test_mediapipe_person_dnn_postprocessing_without_model_file() -> None:
    from reframelab.semantic import _blazepose_anchors, _MediaPipePersonDetector

    class FakeNet:
        def setInput(self, blob) -> None:  # noqa: N802
            assert blob.shape == (1, 3, 224, 224)

        def getUnconnectedOutLayersNames(self):  # noqa: N802
            return ["boxes", "scores"]

        def forward(self, names):
            regression = np.zeros((1, 2254, 12), dtype=np.float32)
            scores = np.full((1, 2254, 1), -20.0, dtype=np.float32)
            index = 28 * 28  # a stable interior anchor
            regression[0, index, :4] = [0.0, 0.0, 38.0, 60.0]
            regression[0, index, 4:12] = [
                0.0,
                8.0,
                0.0,
                70.0,
                0.0,
                -16.0,
                0.0,
                22.0,
            ]
            scores[0, index, 0] = 10.0
            return [regression, scores]

    detector = _MediaPipePersonDetector.__new__(_MediaPipePersonDetector)
    detector.net = FakeNet()
    detector.score_threshold = 0.48
    detector.nms_threshold = 0.30
    detector.anchors = _blazepose_anchors()
    regions = detector.detect(np.zeros((200, 320, 3), dtype=np.uint8))
    assert len(regions) == 1
    assert regions[0].kind == "person"
    assert 0.0 <= regions[0].x < 1.0
    assert 0.0 < regions[0].width <= 1.0
