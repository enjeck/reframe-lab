from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

FloatMap = NDArray[np.float32]


@dataclass
class SignalResult:
    maps: dict[str, FloatMap]
    reliability: dict[str, float]


def robust_normalize(image: NDArray[np.generic]) -> FloatMap:
    values = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if values.size == 0:
        return values
    high = float(np.percentile(values, 99.0))
    if high <= 1e-7:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip(values / high, 0.0, 1.0)
    return normalized.astype(np.float32)


def spectral_residual_saliency(frame_bgr: NDArray[np.uint8]) -> FloatMap:
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    complex_spectrum = cv.dft(gray, flags=cv.DFT_COMPLEX_OUTPUT)
    real = complex_spectrum[:, :, 0]
    imaginary = complex_spectrum[:, :, 1]
    magnitude, phase = cv.cartToPolar(real, imaginary)
    log_magnitude = np.log(magnitude + 1e-7)
    average_log = cv.blur(log_magnitude, (3, 3))
    residual = log_magnitude - average_log
    reconstructed_magnitude = np.exp(residual)
    real, imaginary = cv.polarToCart(reconstructed_magnitude, phase)
    reconstructed = cv.merge((real, imaginary))
    inverse = cv.idft(reconstructed, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    saliency = inverse * inverse
    saliency = cv.GaussianBlur(saliency, (9, 9), 2.5)
    return robust_normalize(saliency)


def structure_map(frame_bgr: NDArray[np.uint8]) -> FloatMap:
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0.0)
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    magnitude = cv.magnitude(gx, gy)
    magnitude = cv.GaussianBlur(magnitude, (7, 7), 1.6)
    return robust_normalize(magnitude)


def textness_map(frame_bgr: NDArray[np.uint8]) -> tuple[FloatMap, float]:
    """Estimate text-like horizontal regions without requiring an external model."""

    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    gradient = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gradient = cv.convertScaleAbs(gradient)
    gradient = cv.GaussianBlur(gradient, (3, 3), 0.0)

    _, thresholded = cv.threshold(gradient, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    width = max(5, int(round(frame_bgr.shape[1] * 0.018)))
    if width % 2 == 0:
        width += 1
    closed = cv.morphologyEx(
        thresholded,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (width, 3)),
    )

    count, labels, stats, _ = cv.connectedComponentsWithStats(closed, connectivity=8)
    result = np.zeros(gray.shape, dtype=np.float32)
    accepted_area = 0
    frame_area = gray.shape[0] * gray.shape[1]

    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        if box_width < 8 or box_height < 3:
            continue
        if box_width / max(box_height, 1) < 1.35:
            continue
        if area < 18 or area > frame_area * 0.12:
            continue
        density = area / max(box_width * box_height, 1)
        if density < 0.12:
            continue
        padding_x = max(2, box_width // 16)
        padding_y = max(1, box_height // 5)
        x0 = max(0, x - padding_x)
        y0 = max(0, y - padding_y)
        x1 = min(gray.shape[1], x + box_width + padding_x)
        y1 = min(gray.shape[0], y + box_height + padding_y)
        result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], float(density))
        accepted_area += box_width * box_height

    result = cv.GaussianBlur(result, (9, 5), 1.6)
    reliability = min(1.0, accepted_area / max(frame_area * 0.035, 1.0))
    return robust_normalize(result), float(reliability)


def residual_motion_map(
    previous_bgr: NDArray[np.uint8] | None,
    current_bgr: NDArray[np.uint8],
) -> tuple[FloatMap, float]:
    if previous_bgr is None:
        shape = current_bgr.shape[:2]
        return np.zeros(shape, dtype=np.float32), 0.0

    previous_gray = cv.cvtColor(previous_bgr, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_bgr, cv.COLOR_BGR2GRAY)

    points0 = cv.goodFeaturesToTrack(
        previous_gray,
        maxCorners=320,
        qualityLevel=0.012,
        minDistance=7,
        blockSize=7,
    )

    aligned = previous_gray
    valid_mask = np.ones_like(current_gray, dtype=np.uint8) * 255
    used_compensation = False

    if points0 is not None and len(points0) >= 10:
        points1, status, _ = cv.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if points1 is not None and status is not None:
            valid = status.reshape(-1).astype(bool)
            source_points = points0.reshape(-1, 2)[valid]
            destination_points = points1.reshape(-1, 2)[valid]
            if len(source_points) >= 8:
                transform, inliers = cv.estimateAffinePartial2D(
                    source_points,
                    destination_points,
                    method=cv.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=1500,
                    confidence=0.99,
                    refineIters=10,
                )
                if transform is not None and inliers is not None and int(inliers.sum()) >= 6:
                    size = (current_gray.shape[1], current_gray.shape[0])
                    aligned = cv.warpAffine(
                        previous_gray,
                        transform,
                        size,
                        flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    valid_mask = cv.warpAffine(
                        np.ones_like(previous_gray, dtype=np.uint8) * 255,
                        transform,
                        size,
                        flags=cv.INTER_NEAREST,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    used_compensation = True

    difference = cv.absdiff(aligned, current_gray).astype(np.float32)
    difference[valid_mask == 0] = 0.0
    difference = cv.GaussianBlur(difference, (7, 7), 1.8)

    positive = difference[difference > 0]
    if positive.size:
        # Remove a bounded noise floor without erasing a uniformly changed object.
        threshold = min(float(np.percentile(positive, 67.0)), 18.0)
        difference = np.maximum(difference - threshold, 0.0)

    normalized = robust_normalize(difference)
    active_fraction = float(np.mean(normalized > 0.18))
    mean_strength = float(np.mean(normalized))
    reliability = min(1.0, active_fraction * 5.0 + mean_strength * 2.0)
    if not used_compensation:
        reliability *= 0.75
    return normalized, float(reliability)


def compute_signals(
    frame_bgr: NDArray[np.uint8],
    previous_bgr: NDArray[np.uint8] | None,
    *,
    text_map: FloatMap | None = None,
    text_reliability: float | None = None,
) -> SignalResult:
    saliency = spectral_residual_saliency(frame_bgr)
    structure = structure_map(frame_bgr)
    motion, motion_reliability = residual_motion_map(previous_bgr, frame_bgr)
    if text_map is None or text_reliability is None:
        text, detected_text_reliability = textness_map(frame_bgr)
        if text_map is None:
            text_map = text
        if text_reliability is None:
            text_reliability = detected_text_reliability

    saliency_reliability = min(1.0, float(np.std(saliency)) * 5.5 + 0.25)
    structure_reliability = min(1.0, float(np.mean(structure)) * 3.5 + 0.20)

    return SignalResult(
        maps={
            "saliency": saliency,
            "motion": motion,
            "structure": structure,
            "text": text_map.astype(np.float32),
        },
        reliability={
            "saliency": float(saliency_reliability),
            "motion": float(motion_reliability),
            "structure": float(structure_reliability),
            "text": float(text_reliability),
        },
    )


def shot_change_score(
    previous_bgr: NDArray[np.uint8] | None,
    current_bgr: NDArray[np.uint8],
) -> float:
    if previous_bgr is None:
        return 0.0

    previous_gray = cv.cvtColor(previous_bgr, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_bgr, cv.COLOR_BGR2GRAY)
    difference = float(np.mean(cv.absdiff(previous_gray, current_gray))) / 255.0

    previous_hsv = cv.cvtColor(previous_bgr, cv.COLOR_BGR2HSV)
    current_hsv = cv.cvtColor(current_bgr, cv.COLOR_BGR2HSV)
    previous_hist = cv.calcHist([previous_hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    current_hist = cv.calcHist([current_hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv.normalize(previous_hist, previous_hist, alpha=1.0, norm_type=cv.NORM_L1)
    cv.normalize(current_hist, current_hist, alpha=1.0, norm_type=cv.NORM_L1)
    histogram_distance = float(cv.compareHist(previous_hist, current_hist, cv.HISTCMP_BHATTACHARYYA))

    return float(np.clip(0.56 * histogram_distance + 0.44 * difference, 0.0, 1.0))
