from pathlib import Path
import uuid
import subprocess
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


def copy_to_uploads(src: str | Path, prefix: str = "upload") -> Path:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Файл не найден: {src}")
    ext = src.suffix or ".mp4"
    dst = UPLOADS / unique_name(prefix, ext)
    shutil.copy2(src, dst)
    return dst


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


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return None


def merge_audio(video_no_audio: str | Path, source_video: str | Path,
                start_sec: float = 0.0, end_sec: float = 0.0) -> Path:

    video_no_audio = Path(video_no_audio)
    source_video = Path(source_video)
    out_path = video_no_audio.with_name(video_no_audio.stem + "_audio.mp4")
    ff = _ffmpeg_exe()
    if not ff:
        print("[Audio] ffmpeg не найден — видео без звука")
        return video_no_audio

    cmd = [ff, "-y"]
    if start_sec and start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(video_no_audio)]
    if start_sec and start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(source_video)]
    if end_sec and end_sec > start_sec:
        dur = float(end_sec) - float(start_sec or 0)
        cmd += ["-t", str(dur)]

    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[Audio] Звук добавлен: {out_path}")
            return out_path
        print(f"[Audio] merge fail ({r.returncode}): {(r.stderr or '')[-400:]}")
    except Exception as e:
        print(f"[Audio] error: {e}")
    return video_no_audio


def make_browser_compatible(video_path: str | Path, crf: int = 20, preset: str = "fast") -> Path:
    video_path = Path(video_path)
    out_path = video_path.with_name(video_path.stem + "_h264.mp4")
    ff = _ffmpeg_exe()
    if not ff:
        print("[Video] ffmpeg не найден")
        return video_path

    cmd = [
        ff, "-y", "-i", str(video_path),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[Video] H.264 (crf={crf}): {out_path}")
            return out_path
        cmd = [
            ff, "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
        print(f"[Video] h264 fail: {(r.stderr or '')[-300:]}")
    except Exception as e:
        print(f"[Video] h264 error: {e}")
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
