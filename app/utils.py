import os
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"


def save_upload(file_bytes: bytes, prefix: str, ext: str) -> Path:
    path = UPLOADS / unique_name(prefix, ext)
    path.write_bytes(file_bytes)
    return path


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    cv2.imwrite(str(path), img)
    return path


def get_video_info(path: str | Path) -> Tuple[int, int, float, int]:
    """Возвращает width, height, fps, total_frames"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, total


def create_video_writer(path: str | Path, width: int, height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать VideoWriter")
    return writer