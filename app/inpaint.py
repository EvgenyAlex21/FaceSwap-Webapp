"""
Региональный инпейнт по маске (кисть / ластик).
Базовый режим — OpenCV (всегда работает).
Опционально — лучше качество через простой patch-based refine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import OUTPUTS, unique_name, write_image


def normalize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Приводит маску к grayscale uint8 0/255 размера изображения."""
    if mask is None:
        raise ValueError("Маска пустая — закрасьте область кистью")
    m = mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY) if m.shape[2] >= 3 else m[:, :, 0]
    h, w = target_hw
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # Бинаризация: всё > 10 считаем выделенным
    _, m = cv2.threshold(m.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    # Чуть расширим края маски для более гладкого шва
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.dilate(m, kernel, iterations=1)
    return m


def inpaint_opencv(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: int = 7,
) -> np.ndarray:
    m = normalize_mask(mask, image_bgr.shape[:2])
    if int(m.sum()) == 0:
        raise ValueError("Маска пустая — закрасьте область кистью")
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image_bgr, m, inpaintRadius=int(radius), flags=flag)


def inpaint_region(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: int = 7,
    refine: bool = True,
) -> Path:
    """
    Инпейнт выделенной области.
    method: telea | ns
    radius: радиус OpenCV inpaint (3–15)
    """
    result = inpaint_opencv(image_bgr, mask, method=method, radius=radius)

    if refine:
        # Лёгкое сглаживание границы: смешиваем с оригиналом у края маски
        m = normalize_mask(mask, image_bgr.shape[:2])
        edge = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
        edge = cv2.GaussianBlur(edge, (0, 0), 2.0)
        alpha = (edge.astype(np.float32) / 255.0)[..., None]
        result = (
            result.astype(np.float32) * alpha + image_bgr.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

    out = OUTPUTS / unique_name("inpaint", ".png")
    write_image(out, result, fmt="png")
    return out
