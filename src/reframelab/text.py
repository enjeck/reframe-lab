from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .dnn_runtime import load_onnx_net
from .model_assets import ensure_model
from .models import TextRegion
from .signals import robust_normalize

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TextDetectionResult:
    regions: list[TextRegion]
    importance_map: np.ndarray
    reliability: float
    backend: str


def _iou(first: TextRegion, second: TextRegion) -> float:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x + first.width, second.x + second.width)
    y1 = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / max(union, 1e-8)


def _suppress_regions(regions: list[TextRegion], threshold: float = 0.72) -> list[TextRegion]:
    kept: list[TextRegion] = []
    for region in sorted(regions, key=lambda item: item.score, reverse=True):
        if any(_iou(region, existing) >= threshold for existing in kept):
            continue
        kept.append(region)
    return sorted(kept, key=lambda item: (item.y, item.x))


def _merge_line_boxes(
    boxes: list[tuple[int, int, int, int, float]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int, float]]:
    """Merge nearby word fragments into lines without merging whole UI panels."""

    if not boxes:
        return []
    pending = sorted(boxes, key=lambda item: (item[1], item[0]))
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int, float]] = []
        used = [False] * len(pending)
        for index, first in enumerate(pending):
            if used[index]:
                continue
            x, y, width, height, score = first
            x1 = x + width
            y1 = y + height
            used[index] = True
            for other_index in range(index + 1, len(pending)):
                if used[other_index]:
                    continue
                ox, oy, ow, oh, oscore = pending[other_index]
                ox1 = ox + ow
                oy1 = oy + oh
                overlap_y = max(0, min(y1, oy1) - max(y, oy))
                vertical_ratio = overlap_y / max(1, min(height, oh))
                gap = max(0, max(x, ox) - min(x1, ox1))
                similar_height = max(height, oh) / max(1, min(height, oh)) <= 2.1
                close = gap <= max(5, int(round(max(height, oh) * 1.9)))
                if vertical_ratio < 0.42 or not close or not similar_height:
                    continue
                nx0 = min(x, ox)
                ny0 = min(y, oy)
                nx1 = max(x1, ox1)
                ny1 = max(y1, oy1)
                # Do not turn a row of unrelated controls into one full-width block.
                if nx1 - nx0 > frame_width * 0.78 and gap > max(height, oh) * 0.65:
                    continue
                x, y, x1, y1 = nx0, ny0, nx1, ny1
                width, height = x1 - x, y1 - y
                score = max(score, oscore) * 0.96 + min(score, oscore) * 0.04
                used[other_index] = True
                changed = True
            merged.append((x, y, x1 - x, y1 - y, float(min(1.0, score))))
        pending = sorted(merged, key=lambda item: (item[1], item[0]))
    return pending


def _classical_regions(frame_bgr: np.ndarray) -> list[TextRegion]:
    height, width = frame_bgr.shape[:2]
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    # Local contrast normalisation helps both dark-on-light UI and light-on-dark titles.
    normalized = cv.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    gradient = cv.Sobel(normalized, cv.CV_32F, 1, 0, ksize=3)
    gradient = cv.convertScaleAbs(gradient)
    gradient = cv.GaussianBlur(gradient, (3, 3), 0.0)
    _, thresholded = cv.threshold(gradient, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    horizontal = max(5, int(round(width * 0.014)))
    vertical = max(2, int(round(height * 0.010)))
    closed = cv.morphologyEx(
        thresholded,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (horizontal, vertical)),
    )
    closed = cv.morphologyEx(
        closed,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (2, 2)),
    )

    count, _, stats, _ = cv.connectedComponentsWithStats(closed, connectivity=8)
    frame_area = height * width
    boxes: list[tuple[int, int, int, int, float]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if box_width < max(6, width // 120) or box_height < max(3, height // 120):
            continue
        if box_height > height * 0.20 or box_width > width * 0.92:
            continue
        aspect = box_width / max(box_height, 1)
        if aspect < 1.05:
            continue
        if area < max(16, frame_area * 0.00006) or area > frame_area * 0.10:
            continue
        fill = area / max(box_width * box_height, 1)
        roi_edges = thresholded[y : y + box_height, x : x + box_width]
        edge_density = float(np.mean(roi_edges > 0)) if roi_edges.size else 0.0
        if fill < 0.10 or edge_density < 0.10:
            continue
        # Text usually contains several alternating strokes across a row.
        row_profile = np.mean(roi_edges > 0, axis=0) if roi_edges.size else np.zeros(1)
        transitions = float(np.mean(np.abs(np.diff(row_profile)))) if len(row_profile) > 1 else 0.0
        if transitions < 0.015 and aspect < 2.0:
            continue
        score = float(
            np.clip(
                0.34
                + 0.30 * min(1.0, edge_density / 0.34)
                + 0.20 * min(1.0, aspect / 7.0)
                + 0.16 * min(1.0, transitions / 0.16),
                0.0,
                1.0,
            )
        )
        boxes.append((x, y, box_width, box_height, score))

    boxes = _merge_line_boxes(boxes, width, height)
    regions: list[TextRegion] = []
    for x, y, box_width, box_height, score in boxes:
        pad_x = max(2, int(round(box_height * 0.45)))
        pad_y = max(1, int(round(box_height * 0.18)))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + box_width + pad_x)
        y1 = min(height, y + box_height + pad_y)
        regions.append(
            TextRegion(
                x=x0 / width,
                y=y0 / height,
                width=(x1 - x0) / width,
                height=(y1 - y0) / height,
                score=score,
            )
        )
    return _suppress_regions(regions)


def _classical_text_reliability(regions: list[TextRegion]) -> float:
    """Estimate whether classical detections resemble text rather than texture.

    The fallback detector is intentionally recall-oriented. Natural footage can
    contain repeated horizontal edges that look text-like to morphology. Real UI
    or title text usually produces several substantial, interior, horizontally
    elongated regions. We use that geometry to keep false positives from becoming
    hard crop constraints.
    """

    substantial = [
        region
        for region in regions
        if region.score >= 0.55
        and region.width >= 0.04
        and region.height >= 0.018
        and region.width * region.height >= 0.0010
        and region.y > 0.012
        and region.y + region.height < 0.988
    ]
    if not substantial:
        return 0.05 if regions else 0.0
    count_score = min(1.0, len(substantial) / 8.0)
    area = sum(region.width * region.height for region in substantial)
    area_score = min(1.0, area / 0.12)
    median_width = float(np.median([region.width for region in substantial]))
    median_height = float(np.median([region.height for region in substantial]))
    width_score = min(1.0, median_width / 0.12)
    height_score = min(1.0, median_height / 0.045)
    return float(
        np.clip(
            0.04
            + 0.34 * count_score
            + 0.32 * area_score
            + 0.16 * width_score
            + 0.14 * height_score,
            0.0,
            1.0,
        )
    )


def _screen_text_prior(frame_bgr: np.ndarray) -> float:
    """Return a conservative prior that the frame contains UI/title text.

    Screen recordings and title cards usually have a smaller colour vocabulary and
    lower mean saturation than natural footage. This is only used to calibrate the
    classical fallback; learned PP-OCR detections are not gated by it.
    """

    preview = cv.resize(frame_bgr, (320, 180), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(preview, cv.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    quantized = (preview // 32).astype(np.int32)
    ids = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
    _, counts = np.unique(ids, return_counts=True)
    probabilities = counts.astype(np.float64) / max(int(counts.sum()), 1)
    entropy = float(-np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-12))))
    saturation_score = float(np.clip((92.0 - mean_saturation) / 72.0, 0.0, 1.0))
    entropy_score = float(np.clip((4.9 - entropy) / 2.8, 0.0, 1.0))
    return float(np.sqrt(saturation_score * entropy_score))


def text_importance_map(
    shape: tuple[int, int],
    regions: list[TextRegion],
) -> tuple[np.ndarray, float]:
    height, width = shape
    result = np.zeros((height, width), dtype=np.float32)
    if not regions:
        return result, 0.0
    for region in regions:
        x0 = int(np.clip(round(region.x * width), 0, width - 1))
        y0 = int(np.clip(round(region.y * height), 0, height - 1))
        x1 = int(np.clip(round((region.x + region.width) * width), x0 + 1, width))
        y1 = int(np.clip(round((region.y + region.height) * height), y0 + 1, height))
        value = float(np.clip(0.55 + 0.45 * region.score, 0.0, 1.0))
        result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], value)
    # A small horizontal dilation means crops receive a penalty before they slice
    # directly through glyphs at the viewport edge.
    kernel_width = max(3, int(round(width * 0.012)))
    result = cv.dilate(result, cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 3)))
    result = cv.GaussianBlur(result, (7, 3), 1.2)
    reliability = _classical_text_reliability(regions)
    return robust_normalize(result), reliability


class TextDetector:
    """OpenCV DNN text detector with an always-available classical fallback."""

    input_size = 736

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        allow_download: bool = True,
        progress: ProgressCallback | None = None,
        dnn_engine: str = "auto",
        batch_size: int = 1,
    ) -> None:
        self.backend = "opencv_classical_text"
        self._model = None
        self._net = None
        self._engine = None
        self._engine_benchmark_ms: dict[str, float] = {}
        self._load_seconds = 0.0
        self.batch_size = max(1, int(batch_size))
        model_path = ensure_model(
            "text_ppocr", model_dir=model_dir, download=allow_download,
            progress=progress, strict=False,
        )
        if model_path is not None and hasattr(cv, "dnn_TextDetectionModel_DB"):
            try:
                sample = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
                loaded = load_onnx_net(
                    model_path, model_name="text_ppocr", sample_blob=sample, mode=dnn_engine,
                )
                self._net = loaded.net
                self._engine = loaded.engine
                self._engine_benchmark_ms = loaded.benchmark_ms
                self._load_seconds = loaded.load_seconds
                model = cv.dnn_TextDetectionModel_DB(self._net)
                model.setBinaryThreshold(0.30)
                model.setPolygonThreshold(0.50)
                model.setUnclipRatio(1.65)
                model.setMaxCandidates(400)
                model.setInputSize((self.input_size, self.input_size))
                model.setInputMean((123.675, 116.28, 103.53))
                model.setInputScale(1.0 / 255.0 / np.asarray([0.229, 0.224, 0.225]))
                self._model = model
                self.backend = "opencv_dnn_ppocr_text+opencv_classical_text"
            except (cv.error, ValueError, TypeError, RuntimeError):
                self._model = None
                self._net = None

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "load_seconds": float(self._load_seconds),
            "requested_batch_size": int(self.batch_size),
            "batch_supported": bool(self._net is not None),
            "engine": self._engine,
            "engine_benchmark_ms": dict(self._engine_benchmark_ms),
        }

    @staticmethod
    def _regions_from_probability(
        probability: np.ndarray, frame_width: int, frame_height: int
    ) -> list[TextRegion]:
        values = np.asarray(probability, dtype=np.float32)
        values = np.squeeze(values)
        if values.ndim != 2:
            return []
        if float(np.nanmax(values)) > 1.0 or float(np.nanmin(values)) < 0.0:
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
        binary = (values >= 0.30).astype(np.uint8) * 255
        binary = cv.morphologyEx(
            binary, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        )
        contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        map_height, map_width = values.shape
        regions: list[TextRegion] = []
        for contour in contours:
            if cv.contourArea(contour) < 8.0:
                continue
            x, y, width, height = cv.boundingRect(contour)
            if width < 3 or height < 2:
                continue
            score = float(np.mean(values[y : y + height, x : x + width]))
            if score < 0.42:
                continue
            # Approximate DB's polygon expansion. The high-level OpenCV wrapper is
            # still used for single images; this path enables real batched inference.
            expand_x = max(1, int(round(width * 0.18)))
            expand_y = max(1, int(round(height * 0.18)))
            x0 = max(0, x - expand_x)
            y0 = max(0, y - expand_y)
            x1 = min(map_width, x + width + expand_x)
            y1 = min(map_height, y + height + expand_y)
            sx0 = x0 / map_width * frame_width
            sy0 = y0 / map_height * frame_height
            sx1 = x1 / map_width * frame_width
            sy1 = y1 / map_height * frame_height
            box_width = sx1 - sx0
            box_height = sy1 - sy0
            if box_width < 4 or box_height < 3:
                continue
            pad_x = max(2.0, box_height * 0.32)
            pad_y = max(1.0, box_height * 0.16)
            fx0 = max(0.0, sx0 - pad_x)
            fy0 = max(0.0, sy0 - pad_y)
            fx1 = min(float(frame_width), sx1 + pad_x)
            fy1 = min(float(frame_height), sy1 + pad_y)
            regions.append(TextRegion(
                x=fx0 / frame_width, y=fy0 / frame_height,
                width=(fx1 - fx0) / frame_width, height=(fy1 - fy0) / frame_height,
                score=float(np.clip(score, 0.0, 1.0)),
            ))
        return _suppress_regions(regions, threshold=0.62)

    def _batch_blob(self, frames: list[np.ndarray]) -> np.ndarray:
        mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        scale = 1.0 / 255.0 / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensors: list[np.ndarray] = []
        for frame in frames:
            resized = cv.resize(
                frame, (self.input_size, self.input_size), interpolation=cv.INTER_LINEAR
            ).astype(np.float32)
            normalized = (resized - mean) * scale
            tensors.append(np.transpose(normalized, (2, 0, 1)))
        return np.stack(tensors, axis=0).astype(np.float32)

    def _dnn_regions_batch(self, frames: list[np.ndarray]) -> list[list[TextRegion]]:
        if self._net is None or not frames:
            return [[] for _ in frames]
        self._net.setInput(self._batch_blob(frames))
        names = self._net.getUnconnectedOutLayersNames()
        raw = self._net.forward(names) if names else self._net.forward()
        outputs = list(raw) if isinstance(raw, (tuple, list)) else [raw]
        if not outputs:
            return [[] for _ in frames]
        value = np.asarray(outputs[0])
        if value.shape[0] != len(frames):
            raise RuntimeError(f"PP-OCR batch output shape {value.shape} does not match {len(frames)} frames")
        return [
            self._regions_from_probability(value[index], frame.shape[1], frame.shape[0])
            for index, frame in enumerate(frames)
        ]

    def _dnn_regions(self, frame_bgr: np.ndarray) -> list[TextRegion]:
        if self._model is None:
            return []
        height, width = frame_bgr.shape[:2]
        try:
            polygons, confidences = self._model.detect(frame_bgr)
        except cv.error:
            return []
        if polygons is None:
            return []
        confidence_values = np.asarray(confidences if confidences is not None else []).reshape(-1)
        regions: list[TextRegion] = []
        for index, polygon in enumerate(polygons):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                continue
            x, y, box_width, box_height = cv.boundingRect(points)
            if box_width < 4 or box_height < 3:
                continue
            score = float(confidence_values[index]) if index < len(confidence_values) else 0.72
            pad_x = max(2, int(round(box_height * 0.32)))
            pad_y = max(1, int(round(box_height * 0.16)))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(width, x + box_width + pad_x)
            y1 = min(height, y + box_height + pad_y)
            regions.append(TextRegion(
                x=x0 / width, y=y0 / height, width=(x1 - x0) / width,
                height=(y1 - y0) / height, score=float(np.clip(score, 0.0, 1.0)),
            ))
        return _suppress_regions(regions, threshold=0.62)

    def _combine(
        self, frame_bgr: np.ndarray, classical: list[TextRegion], learned: list[TextRegion],
        *, use_dnn: bool, batched: bool = False,
    ) -> TextDetectionResult:
        regions = _suppress_regions(learned + classical, threshold=0.58) if learned else classical
        importance, reliability = text_importance_map(frame_bgr.shape[:2], regions)
        if learned:
            reliability = max(reliability, min(1.0, 0.48 + 0.06 * len(learned)))
        else:
            reliability *= _screen_text_prior(frame_bgr)
            if len(regions) <= 5 and any(
                region.score >= 0.72 and region.width >= 0.24
                and 0.026 <= region.height <= 0.12
                and region.width / max(region.height, 1e-6) >= 2.4
                for region in regions
            ):
                reliability = max(reliability, 0.42)
        backend = (
            "opencv_dnn_ppocr_batch+opencv_classical_text" if learned and batched
            else "opencv_dnn_ppocr_text+opencv_classical_text" if learned
            else "opencv_classical_text_cadence" if self._model is not None and not use_dnn
            else "opencv_classical_text"
        )
        return TextDetectionResult(
            regions=regions, importance_map=importance, reliability=float(reliability),
            backend=backend,
        )

    def detect_batch(
        self, frames: list[np.ndarray], *, use_dnn: list[bool] | None = None
    ) -> list[TextDetectionResult]:
        if not frames:
            return []
        flags = use_dnn if use_dnn is not None else [True] * len(frames)
        if len(flags) != len(frames):
            raise ValueError("use_dnn flags must match the frame batch")
        classical = [_classical_regions(frame) for frame in frames]
        learned: list[list[TextRegion]] = [[] for _ in frames]
        indices = [index for index, enabled in enumerate(flags) if enabled]
        batched = False
        if indices and self._net is not None and len(indices) > 1:
            try:
                batch_regions = self._dnn_regions_batch([frames[index] for index in indices])
                for index, regions in zip(indices, batch_regions, strict=False):
                    learned[index] = regions
                batched = True
            except (cv.error, RuntimeError, ValueError):
                for index in indices:
                    learned[index] = self._dnn_regions(frames[index])
        else:
            for index in indices:
                learned[index] = self._dnn_regions(frames[index])
        return [
            self._combine(
                frame, classical[index], learned[index], use_dnn=flags[index],
                batched=batched and index in indices,
            )
            for index, frame in enumerate(frames)
        ]

    def detect(self, frame_bgr: np.ndarray, *, use_dnn: bool = True) -> TextDetectionResult:
        return self.detect_batch([frame_bgr], use_dnn=[use_dnn])[0]
