from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import cv2 as cv
import numpy as np

from .dnn_runtime import load_onnx_net
from .model_assets import ensure_model
from .models import SemanticRegion
from .signals import robust_normalize

ProgressCallback = Callable[[str], None]


def _blazepose_anchors() -> np.ndarray:
    """Generate the 2,254 fixed SSD anchor centres used by BlazePose detection.

    The official OpenCV Zoo wrapper stores the same values as a large literal table.
    Generating the regular grids is smaller and makes the model contract testable.
    """

    anchors: list[tuple[float, float]] = []
    for feature_size, repeats in ((28, 2), (14, 2), (7, 6)):
        for y_index in range(feature_size):
            center_y = (y_index + 0.5) / feature_size
            for x_index in range(feature_size):
                center_x = (x_index + 0.5) / feature_size
                anchors.extend([(center_x, center_y)] * repeats)
    result = np.asarray(anchors, dtype=np.float32)
    if result.shape != (2254, 2):
        raise RuntimeError(f"Unexpected BlazePose anchor shape: {result.shape}")
    return result


def _intersection_over_union(first: SemanticRegion, second: SemanticRegion) -> float:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x + first.width, second.x + second.width)
    y1 = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / max(union, 1e-8)


class _MediaPipePersonDetector:
    """OpenCV DNN wrapper for the OpenCV Zoo MediaPipe person detector."""

    input_size = 224

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.48,
        *,
        dnn_engine: str = "auto",
        batch_size: int = 1,
    ) -> None:
        sample = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        loaded = load_onnx_net(
            model_path,
            model_name="person_mediapipe",
            sample_blob=sample,
            mode=dnn_engine,
        )
        self.net = loaded.net
        self.engine = loaded.engine
        self.engine_benchmark_ms = loaded.benchmark_ms
        self.load_seconds = loaded.load_seconds
        self.batch_size = max(1, int(batch_size))
        self.score_threshold = float(score_threshold)
        self.nms_threshold = 0.30
        self.anchors = _blazepose_anchors()

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        height, width = frame_bgr.shape[:2]
        ratio = min(self.input_size / max(height, 1), self.input_size / max(width, 1))
        resized_width = max(1, int(round(width * ratio)))
        resized_height = max(1, int(round(height * ratio)))
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        rgb = cv.resize(rgb, (resized_width, resized_height), interpolation=cv.INTER_AREA)
        rgb = rgb.astype(np.float32) / 127.5 - 1.0
        pad_left = (self.input_size - resized_width) // 2
        pad_top = (self.input_size - resized_height) // 2
        pad_right = self.input_size - resized_width - pad_left
        pad_bottom = self.input_size - resized_height - pad_top
        padded = cv.copyMakeBorder(
            rgb, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT,
            value=(0.0, 0.0, 0.0),
        )
        blob = np.transpose(padded, (2, 0, 1))[None].astype(np.float32)
        pad_source = np.asarray([pad_left / ratio, pad_top / ratio], dtype=np.float32)
        return blob, ratio, pad_source

    @staticmethod
    def _select_outputs_batched(
        outputs: list[np.ndarray], batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        regression: np.ndarray | None = None
        classification: np.ndarray | None = None
        for output in outputs:
            value = np.asarray(output)
            if value.ndim < 2:
                continue
            last = value.shape[-1]
            if last >= 12:
                regression = value.reshape(batch_size, -1, last)
            elif last == 1:
                classification = value.reshape(batch_size, -1)
        if regression is None or classification is None:
            shapes = [tuple(np.asarray(output).shape) for output in outputs]
            raise RuntimeError(f"Unexpected MediaPipe person detector outputs: {shapes}")
        return regression, classification

    def _postprocess(
        self, regression: np.ndarray, logits: np.ndarray, *, width: int, height: int,
        pad_source: np.ndarray,
    ) -> list[SemanticRegion]:
        count = min(len(regression), len(logits), len(self.anchors))
        regression = regression[:count]
        logits = np.clip(logits[:count].astype(np.float64), -100.0, 100.0)
        scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        anchors = self.anchors[:count]
        box_delta = regression[:, :4].astype(np.float32)
        center_delta = box_delta[:, :2] / float(self.input_size)
        size_delta = np.maximum(0.0, box_delta[:, 2:] / float(self.input_size))
        scale = float(max(width, height))
        top_left = (center_delta - size_delta / 2.0 + anchors) * scale
        bottom_right = (center_delta + size_delta / 2.0 + anchors) * scale
        top_left -= pad_source
        bottom_right -= pad_source
        xywh = np.concatenate((top_left, np.maximum(1.0, bottom_right - top_left)), axis=1)
        candidates = np.flatnonzero(scores >= self.score_threshold)
        if candidates.size == 0:
            return []
        kept = cv.dnn.NMSBoxes(
            xywh[candidates].tolist(), scores[candidates].tolist(),
            self.score_threshold, self.nms_threshold, top_k=64,
        )
        if len(kept) == 0:
            return []
        selected = candidates[np.asarray(kept).reshape(-1)]
        landmarks = regression[:, 4:12].reshape(-1, 4, 2).astype(np.float32)
        landmarks = (landmarks / float(self.input_size) + anchors[:, None, :]) * scale
        landmarks -= pad_source[None, None, :]
        regions: list[SemanticRegion] = []
        for index in selected:
            score = float(scores[index])
            hip_center, full_body_point, shoulder_center, upper_body_point = landmarks[index]
            full_radius = float(np.linalg.norm(full_body_point - hip_center))
            upper_radius = float(np.linalg.norm(upper_body_point - shoulder_center))
            if not np.isfinite(full_radius) or full_radius < 4.0:
                x, y, box_width, box_height = xywh[index]
                center_x = float(x + box_width / 2.0)
                center_y = float(y + box_height / 2.0)
                full_radius = max(float(box_width), float(box_height)) * 1.2
            else:
                center_x = float(hip_center[0])
                center_y = float(hip_center[1])
            body_width = float(np.clip(full_radius * 2.25, 24.0, width))
            body_height = float(np.clip(full_radius * 2.45, 32.0, height))
            body_x = float(np.clip(center_x - body_width / 2.0, 0.0, width - body_width))
            body_y = float(np.clip(center_y - body_height * 0.56, 0.0, height - body_height))
            if not np.isfinite(upper_radius) or upper_radius < 2.0:
                upper_radius = max(5.0, full_radius * 0.22)
            face_width = float(np.clip(upper_radius * 1.10, 8.0, width))
            face_height = float(np.clip(upper_radius * 1.10, 8.0, height))
            face_x = float(np.clip(shoulder_center[0] - face_width / 2.0, 0.0, width - face_width))
            face_y = float(np.clip(shoulder_center[1] - upper_radius * 1.25, 0.0, height - face_height))
            regions.append(SemanticRegion(
                kind="person", x=body_x / width, y=body_y / height,
                width=body_width / width, height=body_height / height, score=score,
                face_x=face_x / width, face_y=face_y / height,
                face_width=face_width / width, face_height=face_height / height,
            ))
        return regions

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[SemanticRegion]]:
        if not frames:
            return []
        blobs: list[np.ndarray] = []
        metadata: list[tuple[int, int, np.ndarray]] = []
        for frame in frames:
            blob, _ratio, pad_source = self._preprocess(frame)
            blobs.append(blob)
            height, width = frame.shape[:2]
            metadata.append((width, height, pad_source))
        self.net.setInput(np.concatenate(blobs, axis=0))
        names = self.net.getUnconnectedOutLayersNames()
        raw = self.net.forward(names) if names else self.net.forward()
        outputs = list(raw) if isinstance(raw, (tuple, list)) else [raw]
        regression, classification = self._select_outputs_batched(outputs, len(frames))
        return [self._postprocess(
            regression[index], classification[index], width=metadata[index][0],
            height=metadata[index][1], pad_source=metadata[index][2],
        ) for index in range(len(frames))]

    def detect(self, frame_bgr: np.ndarray) -> list[SemanticRegion]:
        return self.detect_batch([frame_bgr])[0]


@dataclass
class PersonDetector:
    """Semantic multi-person detector using OpenCV DNN, with offline fallback."""

    model_dir: str | Path | None = None
    allow_download: bool = True
    progress: ProgressCallback | None = None
    dnn_engine: str = "auto"
    batch_size: int = 1

    def __post_init__(self) -> None:
        self.backend = "none"
        self._person: _MediaPipePersonDetector | None = None
        self._yunet = None
        self._haar = None
        self._haar_fallback = None
        self._load_seconds = 0.0
        self._person_batch_disabled = False
        self._person_batch_failures = 0
        self._person_frame_failures = 0
        self._face_failures = 0
        self._face_fallback_uses = 0
        self._warnings_emitted: set[str] = set()
        self._runtime_warnings: list[str] = []
        self._last_person_error: str | None = None
        self._last_face_error: str | None = None
        person_model = ensure_model(
            "person_mediapipe", model_dir=self.model_dir, download=self.allow_download,
            progress=self.progress, strict=False,
        )
        if person_model is not None:
            try:
                self._person = _MediaPipePersonDetector(
                    person_model, dnn_engine=self.dnn_engine, batch_size=self.batch_size,
                )
                self._load_seconds += float(getattr(self._person, "load_seconds", 0.0))
            except (cv.error, RuntimeError, ValueError, TypeError):
                self._person = None
        face_model = ensure_model(
            "yunet", model_dir=self.model_dir, download=self.allow_download,
            progress=self.progress, strict=False,
        )
        if face_model is not None and hasattr(cv, "FaceDetectorYN"):
            started = time.perf_counter()
            try:
                self._yunet = cv.FaceDetectorYN.create(
                    str(face_model), "", (320, 320), 0.68, 0.3, 5000
                )
                self._yunet.setInputSize((320, 320))
                self._yunet.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            except (cv.error, RuntimeError, ValueError, TypeError):
                self._yunet = None
            self._load_seconds += time.perf_counter() - started
        # OpenCV 5 wheels no longer ship the Haar cascades, so only load one that exists.
        cascade_dir = getattr(getattr(cv, "data", None), "haarcascades", "")
        cascade_path = Path(cascade_dir) / "haarcascade_frontalface_default.xml"
        cascade = cv.CascadeClassifier(str(cascade_path)) if cascade_dir and cascade_path.is_file() else None
        if cascade is not None and not cascade.empty():
            self._haar_fallback = cascade
            if self._person is None and self._yunet is None:
                self._haar = cascade
        active: list[str] = []
        if self._person is not None:
            active.append("opencv_dnn_mediapipe_person")
        if self._yunet is not None:
            active.append("opencv_dnn_yunet")
        if self._haar is not None:
            active.append("opencv_haar_fallback")
        self.backend = "+".join(active) if active else "none"

    @property
    def available(self) -> bool:
        return self._person is not None or self._yunet is not None or self._haar is not None

    def runtime_metadata(self) -> dict[str, object]:
        person = getattr(self, "_person", None)
        return {
            "backend": getattr(self, "backend", "none"),
            "load_seconds": float(getattr(self, "_load_seconds", 0.0)),
            "requested_batch_size": int(max(1, getattr(self, "batch_size", 1))),
            "person_batch_supported": bool(person is not None and hasattr(person, "detect_batch")),
            "person_engine": getattr(person, "engine", None),
            "person_engine_benchmark_ms": getattr(person, "engine_benchmark_ms", {}),
            "yunet_engine": "opencv_high_level_api" if getattr(self, "_yunet", None) is not None else None,
            "person_batch_disabled": bool(getattr(self, "_person_batch_disabled", False)),
            "person_batch_failures": int(getattr(self, "_person_batch_failures", 0)),
            "person_frame_failures": int(getattr(self, "_person_frame_failures", 0)),
            "face_failures": int(getattr(self, "_face_failures", 0)),
            "face_fallback_uses": int(getattr(self, "_face_fallback_uses", 0)),
            "last_person_error": getattr(self, "_last_person_error", None),
            "last_face_error": getattr(self, "_last_face_error", None),
            "runtime_warnings": list(getattr(self, "_runtime_warnings", [])),
        }

    def _warn_once(self, key: str, message: str) -> None:
        emitted = getattr(self, "_warnings_emitted", None)
        if emitted is None:
            emitted = set()
            self._warnings_emitted = emitted
        if key in emitted:
            return
        emitted.add(key)
        warnings = getattr(self, "_runtime_warnings", None)
        if warnings is None:
            warnings = []
            self._runtime_warnings = warnings
        warnings.append(message)

    def drain_runtime_warnings(self) -> list[str]:
        warnings = list(getattr(self, "_runtime_warnings", []))
        self._runtime_warnings = []
        return warnings

    def reset_runtime_state(self) -> None:
        """Reset per-job diagnostics while retaining safe capability fallbacks."""

        self._person_batch_failures = 0
        self._person_frame_failures = 0
        self._face_failures = 0
        self._face_fallback_uses = 0
        self._warnings_emitted = set()
        self._runtime_warnings = []
        self._last_person_error = None
        self._last_face_error = None

    @staticmethod
    def _face_to_region(
        detection: tuple[float, float, float, float, float], width: int, height: int
    ) -> SemanticRegion:
        x, y, box_width, box_height, score = detection
        body_width = min(float(width), box_width * 2.25)
        body_height = min(float(height), box_height * 4.2)
        body_x = float(np.clip(x + box_width / 2.0 - body_width / 2.0, 0.0, width - body_width))
        body_y = float(np.clip(y - box_height * 0.30, 0.0, height - body_height))
        return SemanticRegion(
            kind="person", x=body_x / width, y=body_y / height,
            width=body_width / width, height=body_height / height,
            score=float(np.clip(score, 0.0, 1.0)), face_x=x / width, face_y=y / height,
            face_width=box_width / width, face_height=box_height / height,
        )

    @staticmethod
    def _sanitize_region(region: SemanticRegion) -> SemanticRegion | None:
        values = np.asarray(
            [
                region.x,
                region.y,
                region.width,
                region.height,
                region.score,
                region.face_x,
                region.face_y,
                region.face_width,
                region.face_height,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            return None
        x0 = float(np.clip(region.x, 0.0, 1.0))
        y0 = float(np.clip(region.y, 0.0, 1.0))
        x1 = float(np.clip(region.x + region.width, x0, 1.0))
        y1 = float(np.clip(region.y + region.height, y0, 1.0))
        if x1 - x0 <= 1e-5 or y1 - y0 <= 1e-5:
            return None
        face_x0 = float(np.clip(region.face_x, 0.0, 1.0))
        face_y0 = float(np.clip(region.face_y, 0.0, 1.0))
        face_x1 = float(np.clip(region.face_x + region.face_width, face_x0, 1.0))
        face_y1 = float(np.clip(region.face_y + region.face_height, face_y0, 1.0))
        if face_x1 - face_x0 <= 1e-5 or face_y1 - face_y0 <= 1e-5:
            face_x0 = face_y0 = face_x1 = face_y1 = 0.0
        return replace(
            region,
            x=x0,
            y=y0,
            width=x1 - x0,
            height=y1 - y0,
            score=float(np.clip(region.score, 0.0, 1.0)),
            face_x=face_x0,
            face_y=face_y0,
            face_width=face_x1 - face_x0,
            face_height=face_y1 - face_y0,
        )

    @staticmethod
    def _valid_face_detection(
        detection: tuple[float, float, float, float, float], width: int, height: int
    ) -> tuple[float, float, float, float, float] | None:
        values = np.asarray(detection, dtype=np.float64)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            return None
        x, y, box_width, box_height, score = (float(value) for value in values)
        if box_width < 2.0 or box_height < 2.0 or width <= 0 or height <= 0:
            return None
        x0 = float(np.clip(x, 0.0, max(0.0, width - 1.0)))
        y0 = float(np.clip(y, 0.0, max(0.0, height - 1.0)))
        x1 = float(np.clip(x + box_width, x0 + 1.0, float(width)))
        y1 = float(np.clip(y + box_height, y0 + 1.0, float(height)))
        clipped_width = x1 - x0
        clipped_height = y1 - y0
        if clipped_width < 2.0 or clipped_height < 2.0:
            return None
        return x0, y0, clipped_width, clipped_height, float(np.clip(score, 0.0, 1.0))

    def _haar_detections(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        cascade = getattr(self, "_haar", None) or getattr(self, "_haar_fallback", None)
        if cascade is None:
            return []
        height, width = frame_bgr.shape[:2]
        try:
            gray = cv.equalizeHist(cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY))
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(max(16, width // 45), max(16, height // 45)),
            )
        except (cv.error, RuntimeError, ValueError, TypeError) as exc:
            self._face_failures += 1
            self._last_face_error = f"haar: {type(exc).__name__}: {exc}"
            return []
        return [
            (float(x), float(y), float(box_width), float(box_height), 0.72)
            for x, y, box_width, box_height in faces
        ]

    def _face_regions(self, frame_bgr: np.ndarray) -> list[SemanticRegion]:
        height, width = frame_bgr.shape[:2]
        detections: list[tuple[float, float, float, float, float]] = []
        if getattr(self, "_yunet", None) is not None:
            try:
                self._yunet.setInputSize((width, height))
                _, faces = self._yunet.detect(frame_bgr)
                if faces is not None:
                    face_array = np.asarray(faces)
                    if face_array.ndim == 1:
                        face_array = face_array[None, :]
                    for face in face_array:
                        if len(face) < 4:
                            continue
                        x, y, box_width, box_height = (float(value) for value in face[:4])
                        score = float(face[-1]) if len(face) >= 5 else 0.72
                        detections.append((x, y, box_width, box_height, score))
            except (cv.error, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                self._face_failures += 1
                self._last_face_error = f"yunet: {type(exc).__name__}: {exc}"
                self._warn_once(
                    "yunet_runtime_fallback",
                    "YuNet face detection failed on a frame; continuing with person boxes "
                    "and the classical face fallback.",
                )
                detections = self._haar_detections(frame_bgr)
                if detections:
                    self._face_fallback_uses += 1
        elif getattr(self, "_haar", None) is not None:
            detections = self._haar_detections(frame_bgr)
        valid = [
            checked
            for detection in detections
            if (checked := self._valid_face_detection(detection, width, height)) is not None
        ]
        return [self._face_to_region(item, width, height) for item in valid]

    def _detect_people_one(self, frame: np.ndarray) -> list[SemanticRegion]:
        person = getattr(self, "_person", None)
        if person is None:
            return []
        try:
            return list(person.detect(frame))
        except (cv.error, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            self._person_frame_failures += 1
            self._last_person_error = f"single: {type(exc).__name__}: {exc}"
            self._warn_once(
                "person_single_failure",
                "Person detection failed on a frame; continuing with face detection and "
                "the other crop signals.",
            )
            return []

    @staticmethod
    def _merge_regions(
        regions: list[SemanticRegion], face_regions: list[SemanticRegion]
    ) -> list[SemanticRegion]:
        regions = [
            valid
            for region in regions
            if (valid := PersonDetector._sanitize_region(region)) is not None
        ]
        valid_faces = [
            valid
            for region in face_regions
            if (valid := PersonDetector._sanitize_region(region)) is not None
        ]
        for face_region in valid_faces:
            face_center_x = face_region.face_x + face_region.face_width / 2.0
            face_center_y = face_region.face_y + face_region.face_height / 2.0
            owner_index: int | None = None
            for index, person_region in enumerate(regions):
                inside = (
                    person_region.x <= face_center_x <= person_region.x + person_region.width
                    and person_region.y <= face_center_y <= person_region.y + person_region.height
                )
                if inside or _intersection_over_union(person_region, face_region) >= 0.28:
                    owner_index = index
                    break
            if owner_index is None:
                regions.append(face_region)
            else:
                owner = regions[owner_index]
                regions[owner_index] = replace(
                    owner, face_x=face_region.face_x, face_y=face_region.face_y,
                    face_width=face_region.face_width, face_height=face_region.face_height,
                    score=max(owner.score, face_region.score),
                )
        regions.sort(key=lambda item: item.score, reverse=True)
        kept: list[SemanticRegion] = []
        for region in regions:
            if any(_intersection_over_union(region, existing) >= 0.62 for existing in kept):
                continue
            kept.append(region)
        kept.sort(key=lambda region: region.center_x)
        return kept

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[SemanticRegion]]:
        if not frames:
            return []
        person = getattr(self, "_person", None)
        base: list[list[SemanticRegion]]
        can_batch = bool(
            person is not None
            and hasattr(person, "detect_batch")
            and len(frames) > 1
            and not getattr(self, "_person_batch_disabled", False)
        )
        if can_batch:
            try:
                detected = person.detect_batch(frames)
                if len(detected) != len(frames):
                    raise RuntimeError(
                        f"Person detector returned {len(detected)} results for {len(frames)} frames"
                    )
                base = [list(regions) for regions in detected]
            except (cv.error, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                self._person_batch_disabled = True
                self._person_batch_failures += 1
                self._last_person_error = f"batch: {type(exc).__name__}: {exc}"
                self._warn_once(
                    "person_batch_fallback",
                    "Batched person detection is not supported by this OpenCV/model "
                    "combination; falling back to safe single-frame detection.",
                )
                base = [self._detect_people_one(frame) for frame in frames]
        elif person is not None:
            base = [self._detect_people_one(frame) for frame in frames]
        else:
            base = [[] for _ in frames]

        results: list[list[SemanticRegion]] = []
        for index, frame in enumerate(frames):
            try:
                faces = self._face_regions(frame)
                results.append(self._merge_regions(base[index], faces))
            except (cv.error, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                self._face_failures += 1
                self._last_face_error = f"merge: {type(exc).__name__}: {exc}"
                self._warn_once(
                    "face_merge_failure",
                    "Face metadata could not be merged for one frame; continuing with "
                    "person boxes only.",
                )
                results.append(list(base[index]))
        return results

    def detect(self, frame_bgr: np.ndarray) -> list[SemanticRegion]:
        return self.detect_batch([frame_bgr])[0]

def people_importance_map(
    shape: tuple[int, int],
    regions: list[SemanticRegion],
) -> tuple[np.ndarray, float]:
    height, width = shape
    result = np.zeros((height, width), dtype=np.float32)
    if not regions:
        return result, 0.0
    x_grid = np.arange(width, dtype=np.float32)[None, :]
    y_grid = np.arange(height, dtype=np.float32)[:, None]
    for region in regions:
        x0 = int(np.clip(round(region.x * width), 0, width - 1))
        y0 = int(np.clip(round(region.y * height), 0, height - 1))
        x1 = int(np.clip(round((region.x + region.width) * width), x0 + 1, width))
        y1 = int(np.clip(round((region.y + region.height) * height), y0 + 1, height))
        result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], region.score)
        face_cx = int(round((region.face_x + region.face_width / 2.0) * width))
        face_cy = int(round((region.face_y + region.face_height / 2.0) * height))
        sigma_x = max(4.0, region.face_width * width * 1.2)
        sigma_y = max(4.0, region.face_height * height * 1.4)
        gaussian = np.exp(
            -0.5 * np.square((x_grid - face_cx) / sigma_x)
            -0.5 * np.square((y_grid - face_cy) / sigma_y)
        )
        result = np.maximum(result, gaussian.astype(np.float32) * region.score)
    reliability = min(1.0, 0.72 + 0.08 * len(regions))
    return robust_normalize(result), float(reliability)


def speaker_region_strength(region: SemanticRegion) -> float:
    """Calibrate the small raw audio-visual scores into a useful 0..1 weight."""

    score = float(np.clip((region.speaker_score - 0.08) / 0.20, 0.0, 1.0))
    confidence = float(np.clip((region.speaker_confidence - 0.10) / 0.22, 0.0, 1.0))
    return float(np.clip(0.62 * score + 0.38 * confidence, 0.0, 1.0))


def speaker_importance_map(
    shape: tuple[int, int],
    regions: list[SemanticRegion],
) -> tuple[np.ndarray, float]:
    """Build a soft map for the currently likely active speaker.

    The body keeps useful context while the face receives a tighter Gaussian. The
    signal stays inactive when audio-visual confidence is weak.
    """

    height, width = shape
    result = np.zeros((height, width), dtype=np.float32)
    active = [
        region
        for region in regions
        if region.speaker_score >= 0.08
        and region.speaker_confidence >= 0.10
        and region.face_width > 0.0
        and region.face_height > 0.0
    ]
    if not active:
        return result, 0.0
    x_grid = np.arange(width, dtype=np.float32)[None, :]
    y_grid = np.arange(height, dtype=np.float32)[:, None]
    maximum = 0.0
    for region in active:
        weight = speaker_region_strength(region)
        maximum = max(maximum, weight)
        x0 = int(np.clip(round(region.x * width), 0, width - 1))
        y0 = int(np.clip(round(region.y * height), 0, height - 1))
        x1 = int(np.clip(round((region.x + region.width) * width), x0 + 1, width))
        y1 = int(np.clip(round((region.y + region.height) * height), y0 + 1, height))
        result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], weight * 0.55)
        face_cx = (region.face_x + region.face_width / 2.0) * width
        face_cy = (region.face_y + region.face_height / 2.0) * height
        sigma_x = max(4.0, region.face_width * width * 1.15)
        sigma_y = max(4.0, region.face_height * height * 1.25)
        gaussian = np.exp(
            -0.5 * np.square((x_grid - face_cx) / sigma_x)
            -0.5 * np.square((y_grid - face_cy) / sigma_y)
        )
        result = np.maximum(result, gaussian.astype(np.float32) * weight)
    return robust_normalize(result), float(np.clip(maximum, 0.0, 1.0))
