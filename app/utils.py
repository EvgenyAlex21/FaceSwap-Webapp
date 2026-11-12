from pathlib import Path
import uuid
import shutil
import subprocess
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
    path = Path(path)
    width = int(width) // 2 * 2
    height = int(height) // 2 * 2
    fps = float(fps) if fps and fps > 1 else 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        path = path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать VideoWriter")
    return writer, path, width, height


def make_browser_compatible(video_path: str | Path) -> Path:
    """
    Перекодирует видео в H.264 + yuv420p для превью в браузере.
    mp4v часто не играет в Chrome — поэтому нужен этот шаг.
    """
    video_path = Path(video_path)
    out_path = video_path.with_name(video_path.stem + "_preview.mp4")

    # 1) imageio-ffmpeg (есть в requirements)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[Video] Превью H.264: {out_path}")
            return out_path
        print(f"[Video] ffmpeg exit {r.returncode}: {r.stderr[-500:] if r.stderr else ''}")
    except Exception as e:
        print(f"[Video] imageio-ffmpeg: {e}")

    # 2) системный ffmpeg
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
    except Exception as e:
        print(f"[Video] system ffmpeg: {e}")

    print("[Video] H.264 не удался — отдаём оригинал (может не играть в браузере)")
    return video_path


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
