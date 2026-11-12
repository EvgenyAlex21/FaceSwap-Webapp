"""
Оживление фото → короткое видео.
1) Предпочтительно: Stable Video Diffusion (diffusers)
2) Fallback: лёгкая 2.5D анимация (zoom/pan) без тяжёлых моделей
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils import OUTPUTS, unique_name, create_video_writer, write_image

_svd_pipe = None


def _has_svd():
    try:
        import torch
        from diffusers import StableVideoDiffusionPipeline
        return True
    except ImportError:
        return False


def get_svd_pipeline(device: str = "cuda"):
    global _svd_pipe
    if _svd_pipe is not None:
        return _svd_pipe

    if not _has_svd():
        raise RuntimeError(
            "Для нейро-анимации нужно:\n"
            "  pip install diffusers transformers accelerate torch\n"
            "Модель скачается при первом запуске (~8–10 GB)."
        )

    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image

    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[Animator] Загрузка SVD ({model_id}) на {device}...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass

    _svd_pipe = pipe
    print("[Animator] SVD готов.")
    return _svd_pipe


def animate_svd(
    image_bgr: np.ndarray,
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 80,
    noise_aug_strength: float = 0.02,
    seed: int = 0,
    prefer_cuda: bool = True,
) -> Path:
    import torch
    from PIL import Image

    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "Stable Video Diffusion практически не работает на CPU.\n"
            "Нужна рабочая NVIDIA CUDA. Пока используйте «Простую анимацию»."
        )

    pipe = get_svd_pipeline(device)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGB")
    # SVD ожидает 1024x576 или кратное
    pil = pil.resize((1024, 576), Image.LANCZOS)

    generator = torch.manual_seed(int(seed) if seed else 42)

    frames = pipe(
        pil,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug_strength),
        num_frames=int(num_frames),
    ).frames[0]

    out_path = OUTPUTS / unique_name("anim_svd", ".mp4")
    w, h = 1024, 576
    writer = create_video_writer(out_path, w, h, fps)

    for fr in frames:
        arr = np.array(fr)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return out_path


def animate_simple(
    image_bgr: np.ndarray,
    seconds: float = 3.0,
    fps: int = 24,
    mode: str = "zoom_in",
) -> Path:
    """
    Лёгкая анимация без нейросетей: zoom / pan / Ken Burns.
    Работает на CPU быстро.
    """
    h, w = image_bgr.shape[:2]
    total = max(8, int(seconds * fps))

    out_path = OUTPUTS / unique_name("anim_simple", ".mp4")
    writer = create_video_writer(out_path, w, h, fps)

    # Чуть увеличиваем холст для зума
    scale_max = 1.18 if mode in ("zoom_in", "zoom_out") else 1.12
    big = cv2.resize(image_bgr, (int(w * scale_max), int(h * scale_max)), interpolation=cv2.INTER_LANCZOS4)
    bh, bw = big.shape[:2]

    for i in range(total):
        t = i / max(total - 1, 1)

        if mode == "zoom_in":
            sc = 1.0 + (scale_max - 1.0) * t
        elif mode == "zoom_out":
            sc = scale_max - (scale_max - 1.0) * t
        elif mode == "pan_right":
            sc = 1.08
        elif mode == "pan_left":
            sc = 1.08
        else:
            sc = 1.0 + 0.08 * abs(2 * t - 1)  # pulse

        cw, ch = int(w * sc / scale_max * scale_max), int(h * sc / scale_max * scale_max)
        # Нормализуем размер окна
        win_w, win_h = w, h
        max_x = max(0, bw - win_w)
        max_y = max(0, bh - win_h)

        if mode == "pan_right":
            x = int(max_x * t)
            y = max_y // 2
        elif mode == "pan_left":
            x = int(max_x * (1 - t))
            y = max_y // 2
        else:
            x = max_x // 2
            y = max_y // 2

        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        crop = big[y:y + win_h, x:x + win_w]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        writer.write(crop)

    writer.release()
    return out_path
