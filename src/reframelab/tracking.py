from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .advanced_cv import DenseFlowEstimator, PromptSegmenter, mask_geometry
from .model_assets import ensure_model
from .signals import shot_change_score
from .targets import TargetAnchor, TargetObservation, TargetSpec, TargetTrack
from .video import capture_timestamp_seconds, even, probe_video, require_ffmpeg, run_command

ProgressCallback = Callable[[str], None]
GrayFrame = NDArray[np.uint8]
ColorFrame = NDArray[np.uint8]
BBox = tuple[float, float, float, float]


@dataclass
class _TemplateModel:
    gray: GrayFrame
    edge: GrayFrame
    mask: GrayFrame | None
    width: int
    height: int
    descriptors: NDArray[np.uint8] | None
    keypoints: NDArray[np.float32] | None


@dataclass
class _Candidate:
    bbox: BBox
    confidence: float
    source: str
    uncertainty: float = 0.0


@dataclass
class _TrackerState:
    target: TargetSpec
    templates: list[_TemplateModel]
    anchor_boxes: list[tuple[float, BBox]]
    kalman: cv.KalmanFilter
    last_bbox: BBox | None = None
    last_visible: bool = False
    last_confidence: float = 0.0
    lost_frames: int = 0
    cv_tracker: object | None = None
    learned_tracker: object | None = None
    learned_model: Path | None = None
    pending_bbox: BBox | None = None
    pending_count: int = 0
    observations: list[TargetObservation] = field(default_factory=list)


def _analysis_dimensions(width: int, height: int, analysis_height: int) -> tuple[int, int]:
    scale = min(1.0, max(32, analysis_height) / max(height, 1))
    return even(int(round(width * scale))), even(int(round(height * scale)))


def _edge_image(gray: GrayFrame) -> GrayFrame:
    blurred = cv.GaussianBlur(gray, (3, 3), 0.0)
    gx = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)
    magnitude = cv.magnitude(gx, gy)
    high = float(np.percentile(magnitude, 99.0)) if magnitude.size else 0.0
    if high <= 1e-6:
        return np.zeros_like(gray)
    return np.clip(magnitude * (255.0 / high), 0.0, 255.0).astype(np.uint8)


def _clip_bbox(bbox: BBox, width: int, height: int, minimum: int = 4) -> BBox:
    x, y, box_width, box_height = bbox
    box_width = float(np.clip(box_width, minimum, width))
    box_height = float(np.clip(box_height, minimum, height))
    x = float(np.clip(x, 0.0, max(0.0, width - box_width)))
    y = float(np.clip(y, 0.0, max(0.0, height - box_height)))
    return x, y, box_width, box_height


def _anchor_bbox(anchor: TargetAnchor, kind: str, width: int, height: int) -> BBox:
    if kind == "box":
        bbox = (
            anchor.x * width,
            anchor.y * height,
            anchor.width * width,
            anchor.height * height,
        )
    else:
        patch = max(18, int(round(min(width, height) * 0.075)))
        patch = min(patch, max(8, min(width, height) // 3))
        bbox = (
            anchor.center_x * width - patch / 2.0,
            anchor.center_y * height - patch / 2.0,
            float(patch),
            float(patch),
        )
    return _clip_bbox(bbox, width, height)


def _crop(gray: GrayFrame, bbox: BBox) -> GrayFrame:
    x, y, width, height = bbox
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(gray.shape[1], int(round(x + width)))
    y1 = min(gray.shape[0], int(round(y + height)))
    return gray[y0:y1, x0:x1].copy()


def _make_template(gray: GrayFrame, bbox: BBox, kind: str) -> _TemplateModel:
    patch = _crop(gray, bbox)
    if patch.size == 0:
        raise ValueError("The selected target is outside the video frame")
    edge = _edge_image(patch)

    positive = edge[edge > 0]
    if positive.size >= 8:
        threshold = max(20.0, float(np.percentile(positive, 48.0)))
        edge_mask = (edge >= threshold).astype(np.uint8) * 255
        edge_mask = cv.dilate(edge_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    else:
        edge_mask = np.zeros_like(edge)

    if kind == "point":
        border = np.concatenate((patch[0], patch[-1], patch[:, 0], patch[:, -1])).astype(np.float32)
        background = float(np.median(border)) if border.size else float(np.median(patch))
        difference = np.abs(patch.astype(np.float32) - background)
        positive_difference = difference[difference > 0]
        difference_threshold = max(18.0, float(np.percentile(positive_difference, 58.0))) if positive_difference.size else 18.0
        foreground = (difference >= difference_threshold).astype(np.uint8) * 255
        foreground = cv.morphologyEx(foreground, cv.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        foreground = cv.dilate(foreground, np.ones((3, 3), dtype=np.uint8), iterations=1)
        centre_mask = np.zeros_like(foreground)
        axes = (max(2, patch.shape[1] // 2 - 1), max(2, patch.shape[0] // 2 - 1))
        cv.ellipse(centre_mask, (patch.shape[1] // 2, patch.shape[0] // 2), axes, 0, 0, 360, 255, -1)
        mask = cv.bitwise_and(cv.bitwise_or(foreground, edge_mask), centre_mask)
    else:
        mask = edge_mask
    if int(np.count_nonzero(mask)) < 6:
        mask = None

    orb = cv.ORB_create(
        nfeatures=320,
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=5,
        patchSize=15,
        fastThreshold=7,
    )
    keypoints, descriptors = orb.detectAndCompute(patch, None)
    point_array: NDArray[np.float32] | None
    if keypoints and descriptors is not None and len(keypoints) >= 4:
        point_array = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    else:
        point_array = None
        descriptors = None

    return _TemplateModel(
        gray=patch,
        edge=edge,
        mask=mask,
        width=patch.shape[1],
        height=patch.shape[0],
        descriptors=descriptors,
        keypoints=point_array,
    )


def _read_frame_at(path: Path, time_seconds: float, size: tuple[int, int]) -> ColorFrame:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not decode {path}")
    try:
        cap.set(cv.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            # Some backends seek to the first frame after EOF for the final timestamp.
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read a target anchor frame at {time_seconds:.3f}s")
        return cv.resize(frame, size, interpolation=cv.INTER_AREA)
    finally:
        cap.release()


def _make_kalman(center_x: float, center_y: float) -> cv.KalmanFilter:
    kalman = cv.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.diag([0.08, 0.08, 0.5, 0.5]).astype(np.float32)
    kalman.measurementNoiseCov = np.diag([1.8, 1.8]).astype(np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    kalman.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], dtype=np.float32)
    return kalman


def _kalman_predict(state: _TrackerState, dt: float) -> tuple[float, float]:
    state.kalman.transitionMatrix[0, 2] = float(dt)
    state.kalman.transitionMatrix[1, 3] = float(dt)
    prediction = state.kalman.predict()
    return float(prediction[0, 0]), float(prediction[1, 0])


def _kalman_correct(state: _TrackerState, bbox: BBox) -> None:
    x, y, width, height = bbox
    measurement = np.array([[x + width / 2.0], [y + height / 2.0]], dtype=np.float32)
    state.kalman.correct(measurement)


def _search_rect(
    predicted: tuple[float, float],
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    lost_frames: int,
) -> tuple[int, int, int, int]:
    _, _, box_width, box_height = bbox
    expansion = 2.8 + min(4.0, lost_frames * 0.65)
    search_width = max(box_width * expansion, frame_width * 0.18)
    search_height = max(box_height * expansion, frame_height * 0.18)
    x0 = int(max(0.0, predicted[0] - search_width / 2.0))
    y0 = int(max(0.0, predicted[1] - search_height / 2.0))
    x1 = int(min(frame_width, predicted[0] + search_width / 2.0))
    y1 = int(min(frame_height, predicted[1] + search_height / 2.0))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _template_match_one(
    frame_gray: GrayFrame,
    frame_edge: GrayFrame,
    model: _TemplateModel,
    search: tuple[int, int, int, int],
    scales: tuple[float, ...],
    *,
    fast_edge_only: bool = False,
) -> _Candidate | None:
    search_x, search_y, search_width, search_height = search
    gray_region = frame_gray[search_y : search_y + search_height, search_x : search_x + search_width]
    edge_region = frame_edge[search_y : search_y + search_height, search_x : search_x + search_width]
    best: _Candidate | None = None

    for scale in scales:
        target_width = max(4, int(round(model.width * scale)))
        target_height = max(4, int(round(model.height * scale)))
        if target_width > search_width or target_height > search_height:
            continue
        if (target_width, target_height) == (model.width, model.height):
            template_gray = model.gray
            template_edge = model.edge
            template_mask = model.mask
        else:
            template_gray = cv.resize(model.gray, (target_width, target_height), interpolation=cv.INTER_AREA)
            template_edge = cv.resize(model.edge, (target_width, target_height), interpolation=cv.INTER_AREA)
            template_mask = (
                cv.resize(model.mask, (target_width, target_height), interpolation=cv.INTER_NEAREST)
                if model.mask is not None
                else None
            )

        if fast_edge_only:
            # Full-frame point reacquisition is the expensive path. One edge
            # correlation map is enough to propose a location; the normal
            # appearance validator below still rejects weak or ambiguous matches.
            try:
                if template_mask is not None:
                    edge_result = cv.matchTemplate(
                        edge_region,
                        template_edge,
                        cv.TM_CCORR_NORMED,
                        mask=template_mask,
                    )
                else:
                    edge_result = cv.matchTemplate(
                        edge_region, template_edge, cv.TM_CCOEFF_NORMED
                    )
            except cv.error:
                edge_result = cv.matchTemplate(
                    edge_region, template_edge, cv.TM_CCOEFF_NORMED
                )
            edge_result = np.nan_to_num(
                edge_result, nan=-1.0, posinf=-1.0, neginf=-1.0
            )
            if template_mask is not None:
                combined = np.clip(edge_result, 0.0, 1.0)
            else:
                combined = np.clip((edge_result + 1.0) * 0.5, 0.0, 1.0)
        else:
            try:
                if template_mask is not None:
                    edge_result = cv.matchTemplate(
                        edge_region,
                        template_edge,
                        cv.TM_CCORR_NORMED,
                        mask=template_mask,
                    )
                    difference_result = cv.matchTemplate(
                        gray_region,
                        template_gray,
                        cv.TM_SQDIFF_NORMED,
                        mask=template_mask,
                    )
                else:
                    edge_result = cv.matchTemplate(
                        edge_region, template_edge, cv.TM_CCOEFF_NORMED
                    )
                    difference_result = cv.matchTemplate(
                        gray_region, template_gray, cv.TM_SQDIFF_NORMED
                    )
            except cv.error:
                edge_result = cv.matchTemplate(
                    edge_region, template_edge, cv.TM_CCOEFF_NORMED
                )
                difference_result = cv.matchTemplate(
                    gray_region, template_gray, cv.TM_SQDIFF_NORMED
                )
            gray_result = cv.matchTemplate(
                gray_region, template_gray, cv.TM_CCOEFF_NORMED
            )
            edge_result = np.nan_to_num(
                edge_result, nan=-1.0, posinf=-1.0, neginf=-1.0
            )
            difference_result = np.nan_to_num(
                difference_result, nan=1.0, posinf=1.0, neginf=1.0
            )
            gray_result = np.nan_to_num(
                gray_result, nan=-1.0, posinf=-1.0, neginf=-1.0
            )
            edge_score = np.clip((edge_result + 1.0) * 0.5, 0.0, 1.0)
            difference_score = np.clip(1.0 - difference_result, 0.0, 1.0)
            gray_score = np.clip((gray_result + 1.0) * 0.5, 0.0, 1.0)
            combined = edge_score * 0.42 + difference_score * 0.43 + gray_score * 0.15
        _, maximum, _, location = cv.minMaxLoc(combined.astype(np.float32))
        candidate = _Candidate(
            bbox=(
                float(search_x + location[0]),
                float(search_y + location[1]),
                float(target_width),
                float(target_height),
            ),
            confidence=float(np.clip(maximum, 0.0, 1.0)),
            source="template",
        )
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def _template_candidate(
    frame_gray: GrayFrame,
    frame_edge: GrayFrame,
    state: _TrackerState,
    predicted: tuple[float, float],
    *,
    global_search: bool,
) -> _Candidate | None:
    frame_height, frame_width = frame_gray.shape
    if global_search or state.last_bbox is None:
        search = (0, 0, frame_width, frame_height)
        if state.target.kind == "point":
            scales = (0.88, 1.0, 1.12) if state.lost_frames >= 6 else (1.0,)
        else:
            scales = (0.82, 1.0, 1.18) if state.lost_frames >= 3 else (1.0,)
    else:
        search = _search_rect(
            predicted,
            state.last_bbox,
            frame_width,
            frame_height,
            state.lost_frames,
        )
        scales = (1.0,) if state.target.kind == "point" else (0.88, 1.0, 1.12)

    best: _Candidate | None = None
    for model in state.templates:
        candidate = _template_match_one(
            frame_gray,
            frame_edge,
            model,
            search,
            scales,
            fast_edge_only=global_search and state.target.kind == "point",
        )
        if candidate is not None and (best is None or candidate.confidence > best.confidence):
            best = candidate
    if best is not None:
        best.source = "template_global" if global_search else "template_local"
    return best


def _flow_candidate(
    previous_gray: GrayFrame | None,
    current_gray: GrayFrame,
    bbox: BBox | None,
) -> _Candidate | None:
    if previous_gray is None or bbox is None:
        return None
    x, y, width, height = _clip_bbox(bbox, previous_gray.shape[1], previous_gray.shape[0])
    mask = np.zeros_like(previous_gray)
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + width)), int(round(y + height))
    mask[y0:y1, x0:x1] = 255
    points0 = cv.goodFeaturesToTrack(
        previous_gray,
        maxCorners=80,
        qualityLevel=0.008,
        minDistance=3,
        blockSize=5,
        mask=mask,
    )
    if points0 is None or len(points0) < 4:
        return None
    points1, status1, _ = cv.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if points1 is None or status1 is None:
        return None
    points_back, status2, _ = cv.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        points1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if points_back is None or status2 is None:
        return None
    good = status1.reshape(-1).astype(bool) & status2.reshape(-1).astype(bool)
    source = points0.reshape(-1, 2)[good]
    destination = points1.reshape(-1, 2)[good]
    backward = points_back.reshape(-1, 2)[good]
    if len(source) < 4:
        return None
    error = np.linalg.norm(source - backward, axis=1)
    stable = error <= max(1.5, float(np.percentile(error, 70.0)))
    source = source[stable]
    destination = destination[stable]
    if len(source) < 4:
        return None
    transform, inliers = cv.estimateAffinePartial2D(
        source,
        destination,
        method=cv.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=800,
        confidence=0.98,
        refineIters=8,
    )
    if transform is None or inliers is None or int(inliers.sum()) < 3:
        return None
    corners = np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    transformed = cv.transform(corners, transform).reshape(-1, 2)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    candidate_bbox = _clip_bbox(
        (
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0] - minimum[0]),
            float(maximum[1] - minimum[1]),
        ),
        current_gray.shape[1],
        current_gray.shape[0],
    )
    inlier_ratio = float(inliers.mean())
    confidence = float(np.clip(0.35 + 0.55 * inlier_ratio - 0.03 * np.median(error), 0.0, 0.92))
    return _Candidate(candidate_bbox, confidence, "optical_flow")


def _orb_candidate(
    frame_keypoints: list[cv.KeyPoint],
    frame_descriptors: NDArray[np.uint8] | None,
    state: _TrackerState,
    frame_width: int,
    frame_height: int,
) -> _Candidate | None:
    if frame_descriptors is None or len(frame_keypoints) < 4:
        return None
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    best: _Candidate | None = None
    frame_points = np.asarray([keypoint.pt for keypoint in frame_keypoints], dtype=np.float32)

    for model in state.templates:
        if model.descriptors is None or model.keypoints is None or len(model.descriptors) < 4:
            continue
        matches = matcher.knnMatch(model.descriptors, frame_descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.76 * pair[1].distance]
        if len(good) < 4:
            continue
        source = np.asarray([model.keypoints[match.queryIdx] for match in good], dtype=np.float32)
        destination = np.asarray([frame_points[match.trainIdx] for match in good], dtype=np.float32)
        transform, inliers = cv.estimateAffinePartial2D(
            source,
            destination,
            method=cv.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1200,
            confidence=0.99,
            refineIters=8,
        )
        if transform is None or inliers is None or int(inliers.sum()) < 4:
            continue
        corners = np.array(
            [[0, 0], [model.width, 0], [model.width, model.height], [0, model.height]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv.transform(corners, transform).reshape(-1, 2)
        minimum = transformed.min(axis=0)
        maximum = transformed.max(axis=0)
        bbox = _clip_bbox(
            (
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0] - minimum[0]),
                float(maximum[1] - minimum[1]),
            ),
            frame_width,
            frame_height,
        )
        inlier_ratio = float(inliers.mean())
        count_score = min(1.0, len(good) / 14.0)
        confidence = float(np.clip(0.28 + 0.48 * inlier_ratio + 0.24 * count_score, 0.0, 0.98))
        candidate = _Candidate(bbox, confidence, "orb_reacquire")
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def _create_cv_tracker(frame: ColorFrame, bbox: BBox) -> object | None:
    factory = getattr(cv, "TrackerCSRT_create", None) or getattr(cv, "TrackerKCF_create", None)
    if factory is None and hasattr(cv, "legacy"):
        factory = getattr(cv.legacy, "TrackerCSRT_create", None) or getattr(
            cv.legacy,
            "TrackerKCF_create",
            None,
        )
    if factory is None:
        return None
    try:
        tracker = factory()
        initialised = tracker.init(frame, tuple(float(value) for value in bbox))
        if initialised is False:
            return None
        return tracker
    except (cv.error, TypeError):
        return None


def _cv_tracker_candidate(state: _TrackerState, frame: ColorFrame) -> _Candidate | None:
    if state.cv_tracker is None:
        return None
    try:
        ok, bbox = state.cv_tracker.update(frame)
    except cv.error:
        state.cv_tracker = None
        return None
    if not ok:
        state.cv_tracker = None
        return None
    clipped = _clip_bbox(tuple(float(value) for value in bbox), frame.shape[1], frame.shape[0])
    return _Candidate(clipped, 0.64, "opencv_tracker")


def _create_learned_tracker(
    frame: ColorFrame,
    bbox: BBox,
    model_path: Path | None,
) -> object | None:
    if model_path is None or not hasattr(cv, "TrackerVit_create"):
        return None
    try:
        params = cv.TrackerVit_Params()
        params.net = str(model_path)
        params.tracking_score_threshold = 0.18
        tracker = cv.TrackerVit_create(params)
        initialised = tracker.init(frame, tuple(int(round(value)) for value in bbox))
        if initialised is False:
            return None
        return tracker
    except (cv.error, TypeError, AttributeError):
        return None


def _learned_tracker_candidate(state: _TrackerState, frame: ColorFrame) -> _Candidate | None:
    if state.learned_tracker is None:
        return None
    try:
        ok, bbox = state.learned_tracker.update(frame)
        score = float(state.learned_tracker.getTrackingScore())
    except (cv.error, AttributeError):
        state.learned_tracker = None
        return None
    if not ok or not np.isfinite(score):
        return None
    clipped = _clip_bbox(tuple(float(value) for value in bbox), frame.shape[1], frame.shape[0])
    # TrackerVit exposes an actual tracking score. Retain that uncertainty instead of
    # turning every successful update into a fixed confidence.
    uncertainty = float(np.clip(1.0 - score, 0.0, 1.0))
    return _Candidate(clipped, float(np.clip(score, 0.0, 1.0)), "vittrack", uncertainty)


def _bbox_distance(first: BBox, second: BBox) -> float:
    first_center = np.array([first[0] + first[2] / 2.0, first[1] + first[3] / 2.0])
    second_center = np.array([second[0] + second[2] / 2.0, second[1] + second[3] / 2.0])
    scale = max(first[2], first[3], second[2], second[3], 1.0)
    return float(np.linalg.norm(first_center - second_center) / scale)


def _candidate_appearance(
    frame_gray: GrayFrame,
    frame_edge: GrayFrame,
    bbox: BBox,
    templates: list[_TemplateModel],
) -> float:
    """Measure how closely a proposed box resembles the selected appearance.

    Motion-only trackers can lock onto static background texture after a small target
    moves away. This check deliberately revalidates every optical-flow, Kalman and
    OpenCV-tracker proposal against the user-selected template.
    """

    patch_gray = _crop(frame_gray, bbox)
    patch_edge = _crop(frame_edge, bbox)
    if patch_gray.size == 0:
        return 0.0
    best = 0.0
    for model in templates:
        gray = cv.resize(patch_gray, (model.width, model.height), interpolation=cv.INTER_AREA)
        edge = cv.resize(patch_edge, (model.width, model.height), interpolation=cv.INTER_AREA)
        mask = model.mask > 0 if model.mask is not None else np.ones_like(model.gray, dtype=bool)
        if int(mask.sum()) < 4:
            continue
        gray_values = gray.astype(np.float32)[mask]
        template_values = model.gray.astype(np.float32)[mask]
        edge_values = edge.astype(np.float32)[mask]
        template_edge = model.edge.astype(np.float32)[mask]

        difference = float(np.mean(np.abs(gray_values - template_values))) / 255.0
        difference_score = float(np.clip(1.0 - difference * 1.8, 0.0, 1.0))

        gray_a = gray_values - float(gray_values.mean())
        gray_b = template_values - float(template_values.mean())
        gray_denominator = float(np.linalg.norm(gray_a) * np.linalg.norm(gray_b))
        gray_corr = float(np.dot(gray_a, gray_b) / gray_denominator) if gray_denominator > 1e-6 else 0.0
        gray_score = float(np.clip((gray_corr + 1.0) * 0.5, 0.0, 1.0))

        edge_denominator = float(np.linalg.norm(edge_values) * np.linalg.norm(template_edge))
        edge_corr = float(np.dot(edge_values, template_edge) / edge_denominator) if edge_denominator > 1e-6 else 0.0
        edge_score = float(np.clip(edge_corr, 0.0, 1.0))
        score = 0.52 * difference_score + 0.20 * gray_score + 0.28 * edge_score
        best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))


def _validated_candidate(
    candidate: _Candidate | None,
    frame_gray: GrayFrame,
    frame_edge: GrayFrame,
    state: _TrackerState,
) -> _Candidate | None:
    if candidate is None:
        return None
    appearance = _candidate_appearance(frame_gray, frame_edge, candidate.bbox, state.templates)
    if candidate.source.startswith("template"):
        confidence = 0.58 * candidate.confidence + 0.42 * appearance
    elif candidate.source == "orb_reacquire":
        confidence = 0.46 * candidate.confidence + 0.54 * appearance
    elif candidate.source == "vittrack":
        confidence = 0.72 * candidate.confidence + 0.28 * appearance
    else:
        confidence = 0.35 * candidate.confidence + 0.65 * appearance
    return _Candidate(
        candidate.bbox,
        float(np.clip(confidence, 0.0, 1.0)),
        candidate.source,
        candidate.uncertainty,
    )


def _candidate_threshold(kind: str, source: str) -> float:
    if kind == "point":
        return {
            "template_global": 0.78,
            "template_local": 0.84,
            "optical_flow": 0.82,
            "opencv_tracker": 0.84,
            "vittrack": 0.58,
        }.get(source, 0.82)
    return {
        "template_global": 0.72,
        "template_local": 0.70,
        "orb_reacquire": 0.70,
        "optical_flow": 0.74,
        "opencv_tracker": 0.76,
        "vittrack": 0.54,
    }.get(source, 0.72)


def _choose_candidate(candidates: list[_Candidate], kind: str) -> _Candidate | None:
    accepted = [
        candidate
        for candidate in candidates
        if candidate.confidence >= _candidate_threshold(kind, candidate.source)
    ]
    if not accepted:
        return None
    accepted.sort(key=lambda item: item.confidence, reverse=True)
    best = accepted[0]
    uncertainty = best.uncertainty
    if len(accepted) >= 2:
        second = accepted[1]
        score_gap = max(0.0, best.confidence - second.confidence)
        disagreement = _bbox_distance(best.bbox, second.bbox)
        ambiguity = float(np.clip((0.12 - score_gap) / 0.12, 0.0, 1.0))
        if disagreement < 0.55:
            best = _Candidate(
                best.bbox,
                min(1.0, best.confidence + 0.05),
                best.source,
                min(uncertainty, 0.18),
            )
        else:
            uncertainty = max(uncertainty, ambiguity * min(1.0, disagreement / 2.0))
            best = _Candidate(best.bbox, best.confidence, best.source, uncertainty)
    return best


def _nearest_anchor_box(state: _TrackerState, time_seconds: float, tolerance: float) -> BBox | None:
    nearest: tuple[float, BBox] | None = None
    for anchor_time, bbox in state.anchor_boxes:
        distance = abs(anchor_time - time_seconds)
        if distance <= tolerance and (nearest is None or distance < nearest[0]):
            nearest = (distance, bbox)
    return nearest[1] if nearest is not None else None


def _normalised_observation(
    frame_index: int,
    time_seconds: float,
    visible: bool,
    confidence: float,
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    source: str,
    *,
    uncertainty: float = 0.0,
    needs_review: bool = False,
) -> TargetObservation:
    x, y, width, height = _clip_bbox(bbox, frame_width, frame_height)
    return TargetObservation(
        frame_index=frame_index,
        time=float(time_seconds),
        visible=bool(visible),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        center_x=float(np.clip((x + width / 2.0) / frame_width, 0.0, 1.0)),
        center_y=float(np.clip((y + height / 2.0) / frame_height, 0.0, 1.0)),
        width=float(np.clip(width / frame_width, 0.0, 1.0)),
        height=float(np.clip(height / frame_height, 0.0, 1.0)),
        source=source,
        uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        needs_review=bool(needs_review),
    )


def _visible_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def _postprocess_track(track: TargetTrack, sample_fps: float) -> None:
    observations = track.observations
    if not observations:
        return
    anchor_times = [anchor.time for anchor in track.target.anchors]
    visible = [observation.visible for observation in observations]

    for start, end in _visible_runs(visible):
        contains_anchor = any(
            observations[start].time - 0.5 / sample_fps
            <= anchor_time
            <= observations[end].time + 0.5 / sample_fps
            for anchor_time in anchor_times
        )
        if end - start + 1 < 2 and not contains_anchor:
            observations[start].visible = False
            observations[start].confidence *= 0.4
            observations[start].source = "rejected_isolated"
            observations[start].needs_review = True
            observations[start].uncertainty = 1.0

    maximum_gap = max(1, int(round(sample_fps * 0.45)))
    visible_indices = [index for index, item in enumerate(observations) if item.visible]
    for left, right in zip(visible_indices, visible_indices[1:], strict=False):
        gap = right - left - 1
        if not 0 < gap <= maximum_gap:
            continue
        before = observations[left]
        after = observations[right]
        if min(before.confidence, after.confidence) < 0.62:
            continue
        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            item = observations[left + offset]
            item.visible = True
            item.confidence = float(min(before.confidence, after.confidence) * 0.72)
            item.center_x = float((1.0 - alpha) * before.center_x + alpha * after.center_x)
            item.center_y = float((1.0 - alpha) * before.center_y + alpha * after.center_y)
            item.width = float((1.0 - alpha) * before.width + alpha * after.width)
            item.height = float((1.0 - alpha) * before.height + alpha * after.height)
            item.source = "gap_interpolation"
            item.needs_review = True
            item.uncertainty = max(item.uncertainty, 0.55)

    # Force every user anchor into the track; an annotation is a hard observation.
    times = np.asarray([item.time for item in observations], dtype=np.float64)
    for anchor in track.target.anchors:
        index = int(np.argmin(np.abs(times - anchor.time)))
        item = observations[index]
        item.visible = True
        item.confidence = 1.0
        item.center_x = anchor.center_x
        item.center_y = anchor.center_y
        if track.target.kind == "box":
            item.width = anchor.width
            item.height = anchor.height
        item.source = "user_anchor"
        item.needs_review = False
        item.uncertainty = 0.0

    for item in observations:
        if not item.visible or item.confidence < 0.58 or item.uncertainty > 0.42:
            item.needs_review = True

    visible_array = np.asarray([item.visible for item in observations], dtype=bool)
    confidences = [item.confidence for item in observations if item.visible]
    reacquisitions = 0
    previous = False
    for current in visible_array:
        if current and not previous:
            reacquisitions += 1
        previous = bool(current)
    global_match_observations = sum(
        item.source in {"template_global", "orb_reacquire"}
        for item in observations
    )
    track.metrics = {
        "visible_fraction": float(np.mean(visible_array)),
        "mean_visible_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "reacquisition_count": float(max(0, reacquisitions - 1)),
        "global_match_observations": float(global_match_observations),
        "sample_count": float(len(observations)),
    }


@dataclass(frozen=True)
class _SampledFrame:
    frame_index: int
    time: float
    gray: GrayFrame
    cut_before: bool


def _sample_tracking_frames(
    source: Path,
    *,
    frame_width: int,
    frame_height: int,
    sample_fps: float,
    progress: ProgressCallback | None,
) -> list[_SampledFrame]:
    video = probe_video(source)
    cap = cv.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not decode {source}")
    frames: list[_SampledFrame] = []
    next_sample_time = 0.0
    frame_index = 0
    previous_frame_time: float | None = None
    previous_color: ColorFrame | None = None
    expected_samples = max(1, int(math.ceil(video.duration * sample_fps)))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_seconds = capture_timestamp_seconds(
                cap,
                frame_index,
                video.fps,
                previous_frame_time,
            )
            previous_frame_time = time_seconds
            if time_seconds + 0.5 / video.fps < next_sample_time:
                frame_index += 1
                continue
            resized = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)
            cut = (
                shot_change_score(previous_color, resized) >= 0.52
                if previous_color is not None
                else False
            )
            frames.append(
                _SampledFrame(
                    frame_index=frame_index,
                    time=float(time_seconds),
                    gray=cv.cvtColor(resized, cv.COLOR_BGR2GRAY),
                    cut_before=cut,
                )
            )
            previous_color = resized
            interval = 1.0 / sample_fps
            next_sample_time = max(next_sample_time + interval, time_seconds + interval)
            frame_index += 1
            if progress and (
                len(frames) == 1 or len(frames) % max(1, int(sample_fps * 8)) == 0
            ):
                progress(f"Prepared {len(frames)}/{expected_samples} tracking frames")
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be decoded from {source}")
    return frames


def _state_for_anchor(
    target: TargetSpec,
    templates: list[_TemplateModel],
    anchor_boxes: list[tuple[float, BBox]],
    bbox: BBox,
    frame: GrayFrame,
    learned_model: Path | None = None,
) -> _TrackerState:
    center_x = bbox[0] + bbox[2] / 2.0
    center_y = bbox[1] + bbox[3] / 2.0
    state = _TrackerState(
        target=target,
        templates=templates,
        anchor_boxes=anchor_boxes,
        kalman=_make_kalman(center_x, center_y),
        last_bbox=bbox,
        last_visible=True,
        last_confidence=1.0,
        learned_model=learned_model,
    )
    color = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    state.learned_tracker = _create_learned_tracker(color, bbox, learned_model)
    if target.kind == "box":
        state.cv_tracker = _create_cv_tracker(color, bbox)
    return state


def _spatially_adjust(candidate: _Candidate, predicted: tuple[float, float], frame: GrayFrame) -> _Candidate:
    x, y, width, height = candidate.bbox
    center = np.array([x + width / 2.0, y + height / 2.0], dtype=np.float32)
    predicted_array = np.asarray(predicted, dtype=np.float32)
    distance = float(np.linalg.norm(center - predicted_array))
    diagonal = max(float(np.hypot(frame.shape[1], frame.shape[0])), 1.0)
    if candidate.source in {"template_local", "optical_flow", "opencv_tracker"}:
        penalty = min(0.24, distance / diagonal * 0.55)
    else:
        penalty = min(0.08, distance / diagonal * 0.12)
    return _Candidate(
        candidate.bbox,
        max(0.0, candidate.confidence - penalty),
        candidate.source,
        candidate.uncertainty,
    )


def _track_step(
    state: _TrackerState,
    previous: _SampledFrame,
    current: _SampledFrame,
    *,
    crossing_cut: bool,
    sample_fps: float,
) -> TargetObservation:
    if crossing_cut:
        state.last_visible = False
        state.lost_frames = max(state.lost_frames, 2)
        state.cv_tracker = None
        state.learned_tracker = None
        state.pending_bbox = None
        state.pending_count = 0

    dt = max(1e-3, abs(current.time - previous.time))
    predicted = _kalman_predict(state, dt)
    frame_gray = current.gray
    frame_edge = _edge_image(frame_gray)
    frame_color = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR)
    candidates: list[_Candidate] = []

    if state.last_visible and not crossing_cut:
        learned_candidate = _validated_candidate(
            _learned_tracker_candidate(state, frame_color),
            frame_gray,
            frame_edge,
            state,
        )
        if learned_candidate is not None:
            candidates.append(_spatially_adjust(learned_candidate, predicted, frame_gray))

    if state.last_visible and not crossing_cut:
        tracker_candidate = _validated_candidate(
            _cv_tracker_candidate(state, frame_color),
            frame_gray,
            frame_edge,
            state,
        )
        if tracker_candidate is not None:
            candidates.append(_spatially_adjust(tracker_candidate, predicted, frame_gray))
        flow_candidate = _validated_candidate(
            _flow_candidate(previous.gray, frame_gray, state.last_bbox),
            frame_gray,
            frame_edge,
            state,
        )
        if flow_candidate is not None:
            candidates.append(_spatially_adjust(flow_candidate, predicted, frame_gray))

    local_candidate = _validated_candidate(
        _template_candidate(
            frame_gray,
            frame_edge,
            state,
            predicted,
            global_search=False,
        ),
        frame_gray,
        frame_edge,
        state,
    )
    if local_candidate is not None:
        candidates.append(_spatially_adjust(local_candidate, predicted, frame_gray))

    # Global template matching is expensive, especially for tiny point targets.
    # Search immediately when a visible track weakens, but while lost rely on the
    # expanding local window and perform a full-frame search only at a bounded cadence.
    local_reacquire_threshold = 0.80 if state.target.kind == "point" else 0.64
    weak_visible_track = state.last_visible and (
        local_candidate is None
        or local_candidate.confidence < local_reacquire_threshold
    )
    global_interval = max(1, int(round(sample_fps * 0.75)))
    periodic_lost_search = (
        not state.last_visible
        and (state.lost_frames == 0 or state.lost_frames % global_interval == 0)
    )
    should_search_global = weak_visible_track or periodic_lost_search
    if should_search_global:
        global_candidate = _validated_candidate(
            _template_candidate(
                frame_gray,
                frame_edge,
                state,
                predicted,
                global_search=True,
            ),
            frame_gray,
            frame_edge,
            state,
        )
        if global_candidate is not None:
            candidates.append(global_candidate)
        if state.target.kind == "box":
            orb = cv.ORB_create(
                nfeatures=1000,
                fastThreshold=10,
                edgeThreshold=9,
                patchSize=21,
            )
            frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)
            orb_candidate = _validated_candidate(
                _orb_candidate(
                    frame_keypoints or [],
                    frame_descriptors,
                    state,
                    frame_gray.shape[1],
                    frame_gray.shape[0],
                ),
                frame_gray,
                frame_edge,
                state,
            )
            if orb_candidate is not None:
                candidates.append(orb_candidate)

    candidate = _choose_candidate(candidates, state.target.kind)
    if candidate is not None and not state.last_visible and candidate.source in {
        "template_global",
        "orb_reacquire",
    }:
        if state.pending_bbox is not None and _bbox_distance(state.pending_bbox, candidate.bbox) < 0.75:
            state.pending_count += 1
        else:
            state.pending_bbox = candidate.bbox
            state.pending_count = 1
        required = 1 if candidate.confidence >= 0.88 else 2
        if state.pending_count < required:
            candidate = None
    else:
        state.pending_bbox = None
        state.pending_count = 0

    if candidate is not None:
        bbox = _clip_bbox(
            candidate.bbox,
            frame_gray.shape[1],
            frame_gray.shape[0],
        )
        _kalman_correct(state, bbox)
        state.last_bbox = bbox
        state.last_visible = True
        state.last_confidence = candidate.confidence
        state.lost_frames = 0
        state.pending_bbox = None
        state.pending_count = 0
        if state.target.kind == "box" and candidate.confidence >= 0.76:
            state.cv_tracker = _create_cv_tracker(frame_color, bbox)
        if candidate.source != "vittrack" and candidate.confidence >= 0.74:
            state.learned_tracker = _create_learned_tracker(
                frame_color, bbox, state.learned_model
            )
        visible = True
        source = candidate.source
        confidence = candidate.confidence
        uncertainty = candidate.uncertainty
    else:
        bbox = state.last_bbox or state.anchor_boxes[0][1]
        state.last_visible = False
        state.last_confidence *= 0.68
        state.lost_frames += 1
        if state.lost_frames > max(2, int(sample_fps * 0.8)):
            state.cv_tracker = None
        visible = False
        source = "lost"
        confidence = state.last_confidence
        uncertainty = 1.0

    return _normalised_observation(
        current.frame_index,
        current.time,
        visible,
        confidence,
        bbox,
        frame_gray.shape[1],
        frame_gray.shape[0],
        source,
        uncertainty=uncertainty,
        needs_review=(not visible or uncertainty > 0.42 or confidence < 0.58),
    )


def _track_one_target(
    target: TargetSpec,
    samples: list[_SampledFrame],
    *,
    frame_width: int,
    frame_height: int,
    sample_fps: float,
    progress: ProgressCallback | None,
    completed_offset: int,
    total_steps: int,
    learned_model: Path | None = None,
) -> TargetTrack:
    times = np.asarray([sample.time for sample in samples], dtype=np.float64)
    anchor_entries: list[tuple[int, TargetAnchor, BBox]] = []
    templates: list[_TemplateModel] = []
    anchor_boxes: list[tuple[float, BBox]] = []
    for anchor in target.anchors:
        index = int(np.argmin(np.abs(times - anchor.time)))
        bbox = _anchor_bbox(anchor, target.kind, frame_width, frame_height)
        anchor_entries.append((index, anchor, bbox))
        templates.append(_make_template(samples[index].gray, bbox, target.kind))
        anchor_boxes.append((anchor.time, bbox))

    # If two correction anchors land on the same sampled frame, keep the later one.
    unique: dict[int, tuple[int, TargetAnchor, BBox]] = {}
    for entry in anchor_entries:
        unique[entry[0]] = entry
    anchor_entries = [unique[index] for index in sorted(unique)]

    observations: list[TargetObservation | None] = [None] * len(samples)
    processed = 0
    for anchor_number, (anchor_index, _anchor, anchor_bbox) in enumerate(anchor_entries):
        previous_anchor = anchor_entries[anchor_number - 1][0] if anchor_number > 0 else None
        next_anchor = (
            anchor_entries[anchor_number + 1][0]
            if anchor_number + 1 < len(anchor_entries)
            else None
        )
        left_boundary = 0 if previous_anchor is None else (previous_anchor + anchor_index) // 2 + 1
        right_boundary = (
            len(samples) - 1
            if next_anchor is None
            else (anchor_index + next_anchor) // 2
        )
        observations[anchor_index] = _normalised_observation(
            samples[anchor_index].frame_index,
            samples[anchor_index].time,
            True,
            1.0,
            anchor_bbox,
            frame_width,
            frame_height,
            "user_anchor",
        )

        interval_templates = templates
        forward_state = _state_for_anchor(
            target,
            interval_templates,
            anchor_boxes,
            anchor_bbox,
            samples[anchor_index].gray,
            learned_model,
        )
        for index in range(anchor_index + 1, right_boundary + 1):
            observation = _track_step(
                forward_state,
                samples[index - 1],
                samples[index],
                crossing_cut=samples[index].cut_before,
                sample_fps=sample_fps,
            )
            observations[index] = observation
            processed += 1
            if progress and processed % max(1, int(sample_fps * 4)) == 0:
                progress(f"Tracked {completed_offset + processed}/{total_steps} sampled frames")

        backward_state = _state_for_anchor(
            target,
            interval_templates,
            anchor_boxes,
            anchor_bbox,
            samples[anchor_index].gray,
            learned_model,
        )
        for index in range(anchor_index - 1, left_boundary - 1, -1):
            observation = _track_step(
                backward_state,
                samples[index + 1],
                samples[index],
                crossing_cut=samples[index + 1].cut_before,
                sample_fps=sample_fps,
            )
            observations[index] = observation
            processed += 1
            if progress and processed % max(1, int(sample_fps * 4)) == 0:
                progress(f"Tracked {completed_offset + processed}/{total_steps} sampled frames")

    fallback_bbox = anchor_entries[0][2]
    final_observations: list[TargetObservation] = []
    for index, observation in enumerate(observations):
        if observation is None:
            observation = _normalised_observation(
                samples[index].frame_index,
                samples[index].time,
                False,
                0.0,
                fallback_bbox,
                frame_width,
                frame_height,
                "untracked",
            )
        final_observations.append(observation)
    track = TargetTrack(target=target, observations=final_observations)
    _postprocess_track(track, sample_fps)
    return track



def _bbox_from_observation(
    observation: TargetObservation,
    frame_width: int,
    frame_height: int,
) -> BBox:
    width = max(4, int(round(observation.width * frame_width)))
    height = max(4, int(round(observation.height * frame_height)))
    center_x = observation.center_x * frame_width
    center_y = observation.center_y * frame_height
    return _clip_bbox(
        (int(round(center_x - width / 2)), int(round(center_y - height / 2)), width, height),
        frame_width,
        frame_height,
    )


def _mask_plausible(mask: np.ndarray, bbox: BBox) -> bool:
    area = float(np.count_nonzero(mask))
    box_area = float(max(1, bbox[2] * bbox[3]))
    if area < max(12.0, box_area * 0.12):
        return False
    if area > box_area * 3.6:
        return False
    x, y, width, height = (int(round(value)) for value in bbox)
    expanded_x = max(0, x - width // 2)
    expanded_y = max(0, y - height // 2)
    expanded_x1 = min(mask.shape[1], x + width + width // 2)
    expanded_y1 = min(mask.shape[0], y + height + height // 2)
    outside = area - float(np.count_nonzero(mask[expanded_y:expanded_y1, expanded_x:expanded_x1]))
    return outside <= area * 0.18


def _enrich_tracks_with_masks(
    source: Path,
    tracks: list[TargetTrack],
    *,
    frame_width: int,
    frame_height: int,
    sample_fps: float,
    segmenter: PromptSegmenter,
    flow_estimator: DenseFlowEstimator,
    progress: ProgressCallback | None,
) -> None:
    if not tracks:
        return
    video = probe_video(source)
    cap = cv.VideoCapture(str(source))
    if not cap.isOpened():
        return
    next_sample_time = 0.0
    frame_index = 0
    previous_time: float | None = None
    previous_frame: ColorFrame | None = None
    previous_masks: dict[str, np.ndarray] = {}
    previous_visible: dict[str, bool] = {}
    refresh_every = max(1, int(round(sample_fps * 1.0)))
    sample_index = 0
    expected = max(1, int(math.ceil(video.duration * sample_fps)))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_seconds = capture_timestamp_seconds(cap, frame_index, video.fps, previous_time)
            previous_time = time_seconds
            if time_seconds + 0.5 / video.fps < next_sample_time:
                frame_index += 1
                continue
            resized = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)
            for track in tracks:
                observation = track.nearest(time_seconds, tolerance=0.72 / max(sample_fps, 1e-6))
                if observation is None or not observation.visible:
                    previous_visible[track.target.id] = False
                    previous_masks.pop(track.target.id, None)
                    continue
                bbox = _bbox_from_observation(observation, frame_width, frame_height)
                previous_mask = previous_masks.get(track.target.id)
                should_refresh = (
                    previous_frame is None
                    or previous_mask is None
                    or not previous_visible.get(track.target.id, False)
                    or observation.source == "user_anchor"
                    or sample_index % refresh_every == 0
                )
                mask: np.ndarray
                confidence: float
                backend: str
                if not should_refresh and previous_frame is not None and previous_mask is not None:
                    mask = flow_estimator.warp_mask_forward(previous_mask, previous_frame, resized)
                    if not _mask_plausible(mask, bbox):
                        should_refresh = True
                    else:
                        confidence = float(np.clip(observation.confidence * 0.78, 0.0, 0.92))
                        backend = f"flow_propagation:{flow_estimator.backend}"
                if should_refresh:
                    prompt_bbox = bbox if track.target.kind == "box" else None
                    point = (
                        observation.center_x * frame_width,
                        observation.center_y * frame_height,
                    )
                    result = segmenter.segment(
                        resized,
                        bbox=prompt_bbox,
                        point=(None if prompt_bbox is not None else point),
                    )
                    mask = result.mask
                    confidence = result.confidence
                    backend = result.backend
                polygon, profile_x, profile_y, mask_bbox = mask_geometry(mask)
                if mask_bbox is not None and polygon:
                    observation.mask_polygon = polygon
                    observation.mask_x_profile = profile_x
                    observation.mask_y_profile = profile_y
                    observation.mask_confidence = float(np.clip(confidence, 0.0, 1.0))
                    observation.mask_backend = backend
                    if track.target.kind == "box" and confidence >= 0.25:
                        mx, my, mw, mh = mask_bbox
                        observation.center_x = float(np.clip((mx + mw / 2) / frame_width, 0.0, 1.0))
                        observation.center_y = float(np.clip((my + mh / 2) / frame_height, 0.0, 1.0))
                        observation.width = float(np.clip(mw / frame_width, 0.0, 1.0))
                        observation.height = float(np.clip(mh / frame_height, 0.0, 1.0))
                    previous_masks[track.target.id] = mask
                    previous_visible[track.target.id] = True
                else:
                    observation.mask_backend = "none"
                    previous_masks.pop(track.target.id, None)
                    previous_visible[track.target.id] = False
            previous_frame = resized
            sample_index += 1
            next_sample_time = max(next_sample_time + 1.0 / sample_fps, time_seconds + 1.0 / sample_fps)
            frame_index += 1
            if progress and sample_index % max(1, int(sample_fps * 8)) == 0:
                progress(f"Built masks for {sample_index}/{expected} tracking frames")
    finally:
        cap.release()

    for track in tracks:
        mask_items = [item for item in track.observations if item.mask_polygon]
        track.metrics["mask_visible_fraction"] = float(len(mask_items) / max(len(track.observations), 1))
        track.metrics["mean_mask_confidence"] = float(
            np.mean([item.mask_confidence for item in mask_items]) if mask_items else 0.0
        )
        track.metrics["mask_backend_count"] = float(
            len({item.mask_backend for item in mask_items if item.mask_backend != "none"})
        )


def track_targets(
    input_path: str | Path,
    targets: list[TargetSpec],
    *,
    analysis_height: int = 240,
    sample_fps: float = 5.0,
    enable_learned: bool = False,
    enable_masks: bool = False,
    enable_dense_flow: bool = False,
    model_dir: str | Path | None = None,
    allow_model_download: bool = True,
    progress: ProgressCallback | None = None,
) -> list[TargetTrack]:
    """Track selected points and boxes before and after their annotation frames.

    The production stack uses masked template matching, pyramidal Lucas-Kanade
    optical flow, ORB/RANSAC reacquisition, Kalman prediction and CSRT/KCF
    fallbacks. ViTTrack and promptable masks are explicit research options.
    Every proposal is revalidated against the selected appearance and exported
    with uncertainty for pre-render review.
    """

    if not targets:
        return []
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive")
    for target in targets:
        target.validate()

    source = Path(input_path)
    learned_model = (
        ensure_model(
            "vittrack",
            model_dir=model_dir,
            download=allow_model_download,
            progress=progress,
            strict=False,
        )
        if enable_learned
        else None
    )
    video = probe_video(source)
    timeline_tolerance = max(0.15, 1.5 / max(video.fps, 1.0))
    for target in targets:
        for anchor in target.anchors:
            if anchor.time > video.duration + timeline_tolerance:
                raise ValueError(
                    f"Target {target.label!r} has a keyframe at {anchor.time:.3f}s, "
                    f"after the video ends at {video.duration:.3f}s"
                )
    # Tracking small cursors benefits from resolution, but cap the cache size for long videos.
    requested_height = min(max(analysis_height, 180), 480)
    estimated_samples = max(1, int(math.ceil(video.duration * sample_fps)))
    memory_budget = 180 * 1024 * 1024
    aspect = video.width / max(video.height, 1)
    budget_height = int(math.sqrt(memory_budget / max(estimated_samples * aspect, 1.0)))
    tracking_height = max(144, min(requested_height, budget_height))
    frame_width, frame_height = _analysis_dimensions(video.width, video.height, tracking_height)
    samples = _sample_tracking_frames(
        source,
        frame_width=frame_width,
        frame_height=frame_height,
        sample_fps=sample_fps,
        progress=progress,
    )

    total_steps = max(1, len(samples) * len(targets))
    tracks: list[TargetTrack] = []
    completed = 0
    for target in targets:
        track = _track_one_target(
            target,
            samples,
            frame_width=frame_width,
            frame_height=frame_height,
            sample_fps=sample_fps,
            progress=progress,
            completed_offset=completed,
            total_steps=total_steps,
            learned_model=learned_model,
        )
        track.metrics["learned_tracker_available"] = float(learned_model is not None)
        track.metrics["review_flag_fraction"] = float(
            np.mean([item.needs_review for item in track.observations])
        )
        tracks.append(track)
        completed += len(samples)
        if progress:
            progress(f"Tracked {min(completed, total_steps)}/{total_steps} sampled frames")

    if enable_masks:
        segmenter = PromptSegmenter(
            model_dir=model_dir,
            allow_download=allow_model_download,
            progress=progress,
            enable_learned=enable_learned,
        )
        flow_estimator = DenseFlowEstimator(
            model_dir=model_dir,
            allow_download=allow_model_download,
            progress=progress,
            enable_learned=enable_learned and enable_dense_flow,
        )
        _enrich_tracks_with_masks(
            source,
            tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            sample_fps=sample_fps,
            segmenter=segmenter,
            flow_estimator=flow_estimator,
            progress=progress,
        )
    return tracks


def _track_color(index: int) -> tuple[int, int, int]:
    palette = [
        (50, 205, 50),
        (0, 200, 255),
        (255, 140, 0),
        (255, 80, 180),
        (180, 120, 255),
        (80, 220, 220),
        (220, 180, 80),
        (120, 255, 120),
    ]
    return palette[index % len(palette)]


def render_tracking_preview(
    input_path: str | Path,
    tracks: list[TargetTrack],
    output_path: str | Path,
    *,
    sample_fps: float,
    max_width: int = 960,
    progress: ProgressCallback | None = None,
) -> Path | None:
    if not tracks:
        return None
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    video = probe_video(source)
    scale = min(1.0, max_width / max(video.width, 1))
    width = even(int(round(video.width * scale)))
    height = even(int(round(video.height * scale)))
    size = (width, height)

    temporary_root = Path(tempfile.mkdtemp(prefix="reframe-tracking-"))
    silent = temporary_root / "tracking.silent.mp4"
    writer = cv.VideoWriter(
        str(silent),
        cv.VideoWriter_fourcc(*"mp4v"),
        sample_fps,
        size,
    )
    if not writer.isOpened():
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise RuntimeError("Could not create the tracking preview")

    cap = cv.VideoCapture(str(source))
    if not cap.isOpened():
        writer.release()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise RuntimeError(f"Could not decode {source}")
    next_sample_time = 0.0
    frame_index = 0
    sample_index = 0
    previous_time: float | None = None
    expected_samples = max(1, int(math.ceil(video.duration * sample_fps)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_seconds = capture_timestamp_seconds(cap, frame_index, video.fps, previous_time)
            previous_time = time_seconds
            if time_seconds + 0.5 / video.fps < next_sample_time:
                frame_index += 1
                continue
            output = cv.resize(frame, size, interpolation=cv.INTER_AREA)
            for target_index, track in enumerate(tracks):
                observation = track.nearest(time_seconds, tolerance=0.7 / sample_fps)
                if observation is None or not observation.visible:
                    continue
                color = _track_color(target_index)
                if observation.source in {"template_global", "orb_reacquire", "gap_interpolation"}:
                    color = (0, 165, 255)
                center = (
                    int(round(observation.center_x * width)),
                    int(round(observation.center_y * height)),
                )
                if observation.mask_polygon:
                    contour = np.asarray(
                        [[int(round(x * width)), int(round(y * height))] for x, y in observation.mask_polygon],
                        dtype=np.int32,
                    )
                    if len(contour) >= 3:
                        overlay = output.copy()
                        cv.fillPoly(overlay, [contour], color)
                        cv.addWeighted(overlay, 0.18, output, 0.82, 0.0, output)
                        cv.polylines(output, [contour], True, color, 2, cv.LINE_AA)
                if track.target.kind == "point":
                    cv.circle(output, center, max(6, width // 180), color, 2, cv.LINE_AA)
                    cv.drawMarker(output, center, color, cv.MARKER_CROSS, 16, 2, cv.LINE_AA)
                    text_origin = (center[0] + 8, max(18, center[1] - 8))
                else:
                    box_width = max(2, int(round(observation.width * width)))
                    box_height = max(2, int(round(observation.height * height)))
                    x0 = int(round(center[0] - box_width / 2.0))
                    y0 = int(round(center[1] - box_height / 2.0))
                    cv.rectangle(
                        output,
                        (x0, y0),
                        (x0 + box_width, y0 + box_height),
                        color,
                        2,
                        cv.LINE_AA,
                    )
                    text_origin = (max(2, x0), max(18, y0 - 6))
                cv.putText(
                    output,
                    f"{track.target.label} {observation.confidence:.2f} {observation.source}",
                    text_origin,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    1,
                    cv.LINE_AA,
                )
            writer.write(output)
            sample_index += 1
            next_sample_time = max(
                next_sample_time + 1.0 / sample_fps,
                time_seconds + 1.0 / sample_fps,
            )
            frame_index += 1
            if progress and sample_index % max(1, int(sample_fps * 8)) == 0:
                progress(f"Built tracking preview {sample_index}/{expected_samples} frames")
    finally:
        cap.release()
        writer.release()

    try:
        ffmpeg = require_ffmpeg()
        run_command(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return target
