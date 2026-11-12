from pathlib import Path
import uuid
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


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


def write_image(path: str | Path, img: np.ndarray, fmt: str = "jpg") -> Path:
    path = Path(path)
    if fmt.lower() == "png":
        path = path.with_suffix(".png")
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        path = path.with_suffix(".jpg")
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def get_video_info(path: str | Path):
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


def upscale_image(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Простой качественный апскейл через INTER_CUBIC / LANCZOS4."""
    if scale <= 1:
        return img
    h, w = img.shape[:2]
    new_w, new_h = w * scale, h * scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def enhance_face_region(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Лёгкое улучшение резкости и контраста (без тяжёлых моделей).
    strength: 0.0–1.0
    """
    if strength <= 0:
        return img

    # Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)

    # Лёгкий контраст (CLAHE по яркости)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Смешиваем с оригиналом
    return cv2.addWeighted(img, 1.0 - strength * 0.5, enhanced, strength * 0.5, 0)
