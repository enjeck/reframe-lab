"""Advanced OpenCV 5 evidence layers.

The learned models are optional. Every class exposes the backend that actually ran, so
reports never imply that a DNN path was used when the package fell back to a classical
OpenCV algorithm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .model_assets import ensure_model
from .signals import robust_normalize

ProgressCallback = Callable[[str], None]
ColorFrame = NDArray[np.uint8]
FloatMap = NDArray[np.float32]
FlowMap = NDArray[np.float32]
Mask = NDArray[np.uint8]


def _load_onnx(path: Path, *, prefer_ort: bool = False) -> cv.dnn.Net:
    """Load ONNX with OpenCV 5's ORT engine when available, then fall back safely."""

    if prefer_ort:
        engine = getattr(cv.dnn, "ENGINE_ORT", None)
        if engine is not None:
            # OpenCV 5 accepts an engine hint in readNet. Older runtimes do not, so
            # this call is deliberately guarded for API-compatible fallback tests.
            try:
                return cv.dnn.readNet(str(path), "", "onnx", engine)
            except (TypeError, cv.error):
                pass
    return cv.dnn.readNetFromONNX(str(path))


@dataclass(frozen=True)
class DenseFlowResult:
    flow: FlowMap
    residual_map: FloatMap
    reliability: float
    backend: str
    global_transform: NDArray[np.float64] | None = None


class DenseFlowEstimator:
    """Farneback by default, with RAFT available only by explicit opt-in."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        allow_download: bool = True,
        progress: ProgressCallback | None = None,
        enable_learned: bool = False,
    ) -> None:
        self.model_path = (
            ensure_model(
                "raft",
                model_dir=model_dir,
                download=allow_download,
                progress=progress,
                strict=False,
            )
            if enable_learned
            else None
        )
        self.net: cv.dnn.Net | None = None
        self.backend = "farneback_fallback"
        if self.model_path is not None:
            try:
                self.net = _load_onnx(self.model_path, prefer_ort=True)
                self.backend = "opencv_dnn_raft"
            except cv.error:
                self.net = None
                self.backend = "farneback_fallback"

    @staticmethod
    def _farneback(previous_bgr: ColorFrame, current_bgr: ColorFrame) -> FlowMap:
        previous = cv.cvtColor(previous_bgr, cv.COLOR_BGR2GRAY)
        current = cv.cvtColor(current_bgr, cv.COLOR_BGR2GRAY)
        return cv.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=23,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN,
        ).astype(np.float32)

    def _raft(self, previous_bgr: ColorFrame, current_bgr: ColorFrame) -> FlowMap:
        assert self.net is not None
        model_width, model_height = 480, 360
        height, width = current_bgr.shape[:2]

        def prepare(frame: ColorFrame) -> NDArray[np.float32]:
            resized = cv.resize(frame, (model_width, model_height), interpolation=cv.INTER_LINEAR)
            rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB).astype(np.float32)
            return np.transpose(rgb, (2, 0, 1))[None, ...]

        first = prepare(previous_bgr)
        second = prepare(current_bgr)
        self.net.setInput(first, "0")
        self.net.setInput(second, "1")
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        raw = outputs[-1]
        if raw.ndim == 4:
            raw = raw[0]
        if raw.shape[0] == 2:
            raw = np.transpose(raw, (1, 2, 0))
        flow = cv.resize(raw.astype(np.float32), (width, height), interpolation=cv.INTER_LINEAR)
        flow[..., 0] *= width / model_width
        flow[..., 1] *= height / model_height
        return flow

    def infer_flow(self, previous_bgr: ColorFrame, current_bgr: ColorFrame) -> FlowMap:
        if previous_bgr.shape[:2] != current_bgr.shape[:2]:
            previous_bgr = cv.resize(
                previous_bgr,
                (current_bgr.shape[1], current_bgr.shape[0]),
                interpolation=cv.INTER_AREA,
            )
        if self.net is not None:
            try:
                return self._raft(previous_bgr, current_bgr)
            except cv.error:
                self.net = None
                self.backend = "farneback_fallback"
        return self._farneback(previous_bgr, current_bgr)

    @staticmethod
    def _global_transform(flow: FlowMap) -> NDArray[np.float64] | None:
        """Choose the simpler affine model unless a homography is clearly better."""

        height, width = flow.shape[:2]
        stride = max(5, int(round(min(width, height) / 36)))
        yy, xx = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
        source = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
        sampled = flow[yy, xx].reshape(-1, 2)
        finite = np.isfinite(sampled).all(axis=1)
        source = source[finite]
        destination = source + sampled[finite]
        if len(source) < 12:
            return None

        candidates: list[tuple[float, NDArray[np.float64]]] = []
        affine, affine_inliers = cv.estimateAffinePartial2D(
            source,
            destination,
            method=cv.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1500,
            confidence=0.995,
            refineIters=10,
        )
        if affine is not None and affine_inliers is not None and int(affine_inliers.sum()) >= 8:
            predicted = cv.transform(source[None, ...], affine)[0]
            error = np.linalg.norm(predicted - destination, axis=1)
            score = float(np.median(error[affine_inliers.reshape(-1).astype(bool)]))
            candidates.append((score, affine.astype(np.float64)))

        if len(source) >= 20:
            homography, homography_inliers = cv.findHomography(
                source,
                destination,
                method=cv.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=1500,
                confidence=0.995,
            )
            if (
                homography is not None
                and homography_inliers is not None
                and int(homography_inliers.sum()) >= 12
            ):
                predicted = cv.perspectiveTransform(source[None, ...], homography)[0]
                error = np.linalg.norm(predicted - destination, axis=1)
                score = float(np.median(error[homography_inliers.reshape(-1).astype(bool)]))
                # A projective model has more freedom. Require a clear improvement.
                candidates.append((score * 1.08, homography.astype(np.float64)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def residual_motion(
        self,
        previous_bgr: ColorFrame | None,
        current_bgr: ColorFrame,
    ) -> DenseFlowResult:
        if previous_bgr is None:
            zeros = np.zeros((*current_bgr.shape[:2], 2), dtype=np.float32)
            return DenseFlowResult(
                flow=zeros,
                residual_map=np.zeros(current_bgr.shape[:2], dtype=np.float32),
                reliability=0.0,
                backend=self.backend,
            )

        flow = self.infer_flow(previous_bgr, current_bgr)
        transform = self._global_transform(flow)
        height, width = flow.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        if transform is not None:
            if transform.shape == (3, 3):
                homogeneous = np.stack((xx, yy, np.ones_like(xx)), axis=2)
                projected = homogeneous @ transform.T
                denominator = np.maximum(np.abs(projected[..., 2]), 1e-6)
                predicted_x = projected[..., 0] / denominator
                predicted_y = projected[..., 1] / denominator
            else:
                predicted_x = transform[0, 0] * xx + transform[0, 1] * yy + transform[0, 2]
                predicted_y = transform[1, 0] * xx + transform[1, 1] * yy + transform[1, 2]
            global_flow = np.stack((predicted_x - xx, predicted_y - yy), axis=2)
        else:
            median = np.nanmedian(flow.reshape(-1, 2), axis=0)
            global_flow = np.empty_like(flow)
            global_flow[..., 0] = median[0]
            global_flow[..., 1] = median[1]

        residual = flow - global_flow
        magnitude = cv.magnitude(residual[..., 0], residual[..., 1])
        magnitude = cv.GaussianBlur(magnitude, (7, 7), 1.5)
        normalized = robust_normalize(magnitude)
        active_fraction = float(np.mean(normalized > 0.16))
        strength = float(np.mean(normalized))
        spread = float(np.std(normalized))
        flat = np.sort(normalized.reshape(-1))
        total_mass = float(flat.sum())
        top_mass = float(flat[int(0.90 * len(flat)) :].sum()) / max(total_mass, 1e-6)
        # Dense flow is most useful when residual motion is concentrated. A broad map
        # often means parallax or imperfect camera compensation, not one subject.
        broad_penalty = float(np.exp(-5.0 * max(0.0, active_fraction - 0.28)))
        reliability = float(
            np.clip(
                (0.75 * strength + 1.25 * spread + 0.85 * top_mass) * broad_penalty,
                0.0,
                1.0,
            )
        )
        if transform is None:
            reliability *= 0.70
        return DenseFlowResult(
            flow=flow,
            residual_map=normalized,
            reliability=reliability,
            backend=self.backend,
            global_transform=transform,
        )

    def warp_mask_forward(
        self,
        previous_mask: Mask,
        previous_bgr: ColorFrame,
        current_bgr: ColorFrame,
    ) -> Mask:
        """Warp a previous mask into the current frame using backward dense flow."""

        backward = self.infer_flow(current_bgr, previous_bgr)
        height, width = previous_mask.shape[:2]
        if backward.shape[:2] != (height, width):
            original_height, original_width = backward.shape[:2]
            backward = cv.resize(backward, (width, height), interpolation=cv.INTER_LINEAR)
            backward[..., 0] *= width / max(original_width, 1)
            backward[..., 1] *= height / max(original_height, 1)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        map_x = xx + backward[..., 0]
        map_y = yy + backward[..., 1]
        warped = cv.remap(
            previous_mask,
            map_x,
            map_y,
            cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 127).astype(np.uint8) * 255


@dataclass(frozen=True)
class DepthResult:
    importance_map: FloatMap
    relative_depth: FloatMap
    reliability: float
    backend: str


class DepthEstimator:
    """Depth Anything V2 Small through OpenCV DNN with a disabled fallback.

    Depth is deliberately advisory. If the model is not installed or produces a flat
    result, reliability is zero and the planner ignores it.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        allow_download: bool = True,
        progress: ProgressCallback | None = None,
        enable_learned: bool = False,
    ) -> None:
        self.model_path = (
            ensure_model(
                "depth_anything_v2_small",
                model_dir=model_dir,
                download=allow_download,
                progress=progress,
                strict=False,
            )
            if enable_learned
            else None
        )
        self.net: cv.dnn.Net | None = None
        self.backend = "disabled"
        if self.model_path is not None:
            try:
                self.net = _load_onnx(self.model_path, prefer_ort=True)
                self.backend = "opencv_dnn_depth_anything_v2_small"
            except cv.error:
                self.net = None
                self.backend = "disabled"

    def infer(self, frame_bgr: ColorFrame) -> DepthResult:
        height, width = frame_bgr.shape[:2]
        if self.net is None:
            zeros = np.zeros((height, width), dtype=np.float32)
            return DepthResult(zeros, zeros, 0.0, self.backend)

        model_size = 518
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        resized = cv.resize(rgb, (model_size, model_size), interpolation=cv.INTER_CUBIC)
        values = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        values = (values - mean) / std
        blob = np.transpose(values, (2, 0, 1))[None, ...]
        try:
            self.net.setInput(blob)
            output = self.net.forward()
        except cv.error:
            self.net = None
            self.backend = "disabled"
            zeros = np.zeros((height, width), dtype=np.float32)
            return DepthResult(zeros, zeros, 0.0, self.backend)

        depth = np.squeeze(output).astype(np.float32)
        if depth.ndim != 2:
            zeros = np.zeros((height, width), dtype=np.float32)
            return DepthResult(zeros, zeros, 0.0, self.backend)
        depth = cv.resize(depth, (width, height), interpolation=cv.INTER_CUBIC)
        finite = depth[np.isfinite(depth)]
        if finite.size < 16:
            zeros = np.zeros((height, width), dtype=np.float32)
            return DepthResult(zeros, zeros, 0.0, self.backend)
        low, high = np.percentile(finite, [2.0, 98.0])
        if high - low <= 1e-6:
            zeros = np.zeros((height, width), dtype=np.float32)
            return DepthResult(zeros, zeros, 0.0, self.backend)
        relative_depth = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)

        # Depth Anything V2 produces affine-invariant inverse depth. After robust
        # normalization, larger values are treated as nearer to the camera.
        foreground = relative_depth
        gx = cv.Sobel(relative_depth, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(relative_depth, cv.CV_32F, 0, 1, ksize=3)
        edges = robust_normalize(cv.magnitude(gx, gy))
        importance = robust_normalize(0.68 * foreground + 0.32 * edges)
        spread = float(np.percentile(relative_depth, 90) - np.percentile(relative_depth, 10))
        reliability = float(np.clip(spread * 1.6 + np.mean(edges) * 1.4, 0.0, 1.0))
        return DepthResult(importance, relative_depth, reliability, self.backend)


@dataclass(frozen=True)
class SegmentationResult:
    mask: Mask
    confidence: float
    backend: str


class PromptSegmenter:
    """EfficientSAM through OpenCV DNN with GrabCut as an OpenCV fallback."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        allow_download: bool = True,
        progress: ProgressCallback | None = None,
        enable_learned: bool = False,
    ) -> None:
        self.model_path = (
            ensure_model(
                "efficient_sam",
                model_dir=model_dir,
                download=allow_download,
                progress=progress,
                strict=False,
            )
            if enable_learned
            else None
        )
        self.net: cv.dnn.Net | None = None
        self.backend = "grabcut_fallback"
        if self.model_path is not None:
            try:
                self.net = _load_onnx(self.model_path, prefer_ort=True)
                self.backend = "opencv_dnn_efficient_sam"
            except cv.error:
                self.net = None
                self.backend = "grabcut_fallback"

    @staticmethod
    def _clean_mask(mask: Mask, preferred_point: tuple[int, int] | None = None) -> Mask:
        binary = (mask > 0).astype(np.uint8) * 255
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return binary
        selected = -1
        if preferred_point is not None:
            px, py = preferred_point
            if 0 <= py < labels.shape[0] and 0 <= px < labels.shape[1]:
                selected = int(labels[py, px])
        if selected <= 0:
            selected = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        return (labels == selected).astype(np.uint8) * 255

    def _efficient_sam(
        self,
        frame_bgr: ColorFrame,
        *,
        bbox: tuple[int, int, int, int] | None,
        point: tuple[float, float] | None,
    ) -> SegmentationResult:
        assert self.net is not None
        height, width = frame_bgr.shape[:2]
        model_size = 1024
        image = cv.resize(frame_bgr, (model_size, model_size), interpolation=cv.INTER_LINEAR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_blob = np.transpose(image, (2, 0, 1))[None, ...]

        coords = np.zeros((1, 1, 6, 2), dtype=np.float32)
        # The OpenCV EfficientSAM export expects one scalar label per prompt
        # point, including the trailing singleton dimension.
        labels = np.full((1, 1, 6, 1), -1, dtype=np.float32)
        preferred: tuple[int, int] | None = None
        if bbox is not None:
            x, y, box_width, box_height = bbox
            coords[0, 0, 0] = [x * model_size / width, y * model_size / height]
            coords[0, 0, 1] = [
                (x + box_width) * model_size / width,
                (y + box_height) * model_size / height,
            ]
            labels[0, 0, 0, 0] = 2
            labels[0, 0, 1, 0] = 3
            preferred = (int(x + box_width / 2), int(y + box_height / 2))
        elif point is not None:
            px, py = point
            coords[0, 0, 0] = [px * model_size / width, py * model_size / height]
            labels[0, 0, 0, 0] = 1
            preferred = (int(px), int(py))
        else:
            raise ValueError("A point or box prompt is required")

        self.net.setInput(image_blob, "batched_images")
        self.net.setInput(coords, "batched_point_coords")
        self.net.setInput(labels, "batched_point_labels")
        try:
            outputs = self.net.forward(["output_masks", "iou_predictions"])
        except cv.error:
            # Keep compatibility with repackaged exports while preferring the
            # canonical OpenCV model names above.
            output_names = list(self.net.getUnconnectedOutLayersNames())
            outputs = self.net.forward(output_names)
        masks = outputs[0]
        ious = outputs[1] if len(outputs) > 1 else np.zeros((1, 1, masks.shape[-3]), np.float32)
        masks = np.squeeze(masks)
        ious = np.squeeze(ious)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if np.ndim(ious) == 0:
            best_index = 0
            confidence = float(ious)
        else:
            best_index = int(np.argmax(ious))
            confidence = float(np.asarray(ious).reshape(-1)[best_index])
        selected = (masks[best_index] >= 0.0).astype(np.uint8) * 255
        selected = cv.resize(selected, (width, height), interpolation=cv.INTER_NEAREST)
        selected = self._clean_mask(selected, preferred)
        return SegmentationResult(selected, float(np.clip(confidence, 0.0, 1.0)), self.backend)

    @staticmethod
    def _grabcut(
        frame_bgr: ColorFrame,
        *,
        bbox: tuple[int, int, int, int] | None,
        point: tuple[float, float] | None,
    ) -> SegmentationResult:
        height, width = frame_bgr.shape[:2]
        mask = np.full((height, width), cv.GC_PR_BGD, dtype=np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        preferred: tuple[int, int] | None = None
        try:
            if bbox is not None:
                x, y, box_width, box_height = bbox
                x = int(np.clip(x, 0, width - 2))
                y = int(np.clip(y, 0, height - 2))
                box_width = int(np.clip(box_width, 2, width - x))
                box_height = int(np.clip(box_height, 2, height - y))
                preferred = (x + box_width // 2, y + box_height // 2)
                cv.grabCut(
                    frame_bgr,
                    mask,
                    (x, y, box_width, box_height),
                    bgd,
                    fgd,
                    4,
                    cv.GC_INIT_WITH_RECT,
                )
            elif point is not None:
                px = int(np.clip(point[0], 0, width - 1))
                py = int(np.clip(point[1], 0, height - 1))
                preferred = (px, py)
                border = max(2, int(round(min(width, height) * 0.018)))
                mask[:border, :] = cv.GC_BGD
                mask[-border:, :] = cv.GC_BGD
                mask[:, :border] = cv.GC_BGD
                mask[:, -border:] = cv.GC_BGD
                radius = max(3, int(round(min(width, height) * 0.018)))
                cv.circle(mask, (px, py), radius * 2, cv.GC_PR_FGD, -1)
                cv.circle(mask, (px, py), radius, cv.GC_FGD, -1)
                cv.grabCut(frame_bgr, mask, None, bgd, fgd, 4, cv.GC_INIT_WITH_MASK)
            else:
                raise ValueError("A point or box prompt is required")
        except cv.error:
            binary = np.zeros((height, width), dtype=np.uint8)
            if bbox is not None:
                x, y, box_width, box_height = bbox
                binary[max(0, y) : min(height, y + box_height), max(0, x) : min(width, x + box_width)] = 255
            elif point is not None:
                cv.circle(binary, (int(point[0]), int(point[1])), max(4, min(width, height) // 40), 255, -1)
            return SegmentationResult(binary, 0.20, "box_fallback")

        binary = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
        binary = PromptSegmenter._clean_mask(binary, preferred)
        area = float(np.mean(binary > 0))
        confidence = float(np.clip(0.35 + 2.6 * min(area, 0.16), 0.0, 0.78))
        return SegmentationResult(binary, confidence, "grabcut_fallback")

    def segment(
        self,
        frame_bgr: ColorFrame,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        point: tuple[float, float] | None = None,
    ) -> SegmentationResult:
        if self.net is not None:
            try:
                return self._efficient_sam(frame_bgr, bbox=bbox, point=point)
            except cv.error:
                self.net = None
                self.backend = "grabcut_fallback"
        return self._grabcut(frame_bgr, bbox=bbox, point=point)


def mask_geometry(mask: Mask, *, bins: int = 48) -> tuple[list[list[float]], list[float], list[float], tuple[int, int, int, int] | None]:
    """Return a compact contour and one-dimensional mass profiles for crop scoring."""

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], [], [], None
    contour = max(contours, key=cv.contourArea)
    if cv.contourArea(contour) < 4.0:
        return [], [], [], None
    epsilon = max(1.0, 0.006 * cv.arcLength(contour, True))
    contour = cv.approxPolyDP(contour, epsilon, True)
    height, width = binary.shape
    polygon = [
        [float(point[0][0] / max(width, 1)), float(point[0][1] / max(height, 1))]
        for point in contour
    ]
    x_profile_raw = binary.sum(axis=0).astype(np.float32)
    y_profile_raw = binary.sum(axis=1).astype(np.float32)

    def compress(values: NDArray[np.float32]) -> list[float]:
        edges = np.linspace(0, len(values), bins + 1, dtype=np.int32)
        compressed = np.asarray(
            [float(values[edges[i] : edges[i + 1]].sum()) for i in range(bins)],
            dtype=np.float32,
        )
        total = float(compressed.sum())
        if total > 0.0:
            compressed /= total
        return compressed.tolist()

    x, y, box_width, box_height = cv.boundingRect(contour)
    return polygon, compress(x_profile_raw), compress(y_profile_raw), (x, y, box_width, box_height)
