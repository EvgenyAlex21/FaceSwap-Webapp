from pathlib import Path
import uuid
import shutil
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

for d in (UPLOADS, OUTPUTS, MODELS):
    d.mkdir(exist_ok=True)


def unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"


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
    duration = total / fps if fps > 0 else 0
    cap.release()
    return w, h, fps, total, duration


def create_video_writer(path: str | Path, width: int, height: int, fps: float):
    # H264 более совместим с браузерами; fallback mp4v
    path = Path(path)
    for fourcc_name in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Не удалось создать VideoWriter")


def upscale_image(img: np.ndarray, scale: int = 2) -> np.ndarray:
    if scale <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def enhance_face_region(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    if strength <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(img, 1.0 - strength * 0.5, enhanced, strength * 0.5, 0)


def safe_copy_to_outputs(src: Path, prefix: str = "result") -> Path:
    """Копирует файл в outputs с уникальным именем (обход PermissionError Gradio)."""
    src = Path(src)
    ext = src.suffix or ".mp4"
    dst = OUTPUTS / unique_name(prefix, ext)
    shutil.copy2(src, dst)
    return dst
