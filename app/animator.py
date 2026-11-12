"""Оживление фото → видео."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .utils import OUTPUTS, unique_name, create_video_writer, make_browser_compatible

_svd_pipe = None


def _torch_cuda_ok() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "PyTorch не установлен. pip install torch"
    if not torch.cuda.is_available():
        return False, (
            "PyTorch не видит CUDA (установлена CPU-сборка torch).\n"
            "Для SVD нужен torch с CUDA, например:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "Либо используйте Zoom in / Pan — они работают сразу."
        )
    try:
        x = torch.zeros(1, device="cuda")
        del x
        return True, f"PyTorch CUDA OK ({torch.cuda.get_device_name(0)})"
    except Exception as e:
        return False, f"PyTorch CUDA error: {e}"


def animate_svd(
    image_bgr: np.ndarray,
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 80,
    noise_aug_strength: float = 0.02,
    seed: int = 0,
    progress_callback=None,
) -> Path:
    import torch
    from PIL import Image

    ok, msg = _torch_cuda_ok()
    print(f"[Animator] {msg}")
    if not ok:
        raise RuntimeError(msg)

    global _svd_pipe
    if _svd_pipe is None:
        try:
            from diffusers import StableVideoDiffusionPipeline
        except ImportError:
            raise RuntimeError(
                "Нужен пакет diffusers:\n  pip install diffusers transformers accelerate"
            )
        model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        if progress_callback:
            progress_callback(0.05, "Загрузка SVD (~10 GB при первом запуске)...")
        print(f"[Animator] Загрузка {model_id}...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16",
        )
        # 3050 4GB — лучше offload
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
        _svd_pipe = pipe
        print("[Animator] SVD загружен")

    if progress_callback:
        progress_callback(0.2, "Генерация кадров SVD...")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGB").resize((1024, 576), Image.LANCZOS)
    generator = torch.manual_seed(int(seed) if seed else 42)

    frames = _svd_pipe(
        pil,
        decode_chunk_size=4,
        generator=generator,
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug_strength),
        num_frames=int(num_frames),
    ).frames[0]

    if progress_callback:
        progress_callback(0.85, "Сохранение и перекодирование...")

    raw_path = OUTPUTS / unique_name("anim_svd", ".mp4")
    writer, raw_path, w, h = create_video_writer(raw_path, 1024, 576, fps)
    for fr in frames:
        bgr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    out = make_browser_compatible(raw_path)
    if progress_callback:
        progress_callback(1.0, "Готово")
    return out


def animate_simple(
    image_bgr: np.ndarray,
    seconds: float = 3.0,
    fps: int = 24,
    mode: str = "zoom_in",
    progress_callback=None,
) -> Path:
    h, w = image_bgr.shape[:2]
    w, h = w // 2 * 2, h // 2 * 2
    image_bgr = cv2.resize(image_bgr, (w, h))
    total = max(8, int(seconds * fps))

    raw_path = OUTPUTS / unique_name("anim_simple", ".mp4")
    writer, raw_path, w, h = create_video_writer(raw_path, w, h, fps)

    scale_max = 1.18 if mode in ("zoom_in", "zoom_out") else 1.12
    big = cv2.resize(image_bgr, (int(w * scale_max), int(h * scale_max)), interpolation=cv2.INTER_LANCZOS4)
    bh, bw = big.shape[:2]

    for i in range(total):
        t = i / max(total - 1, 1)
        if mode == "zoom_in":
            sc = 1.0 + (scale_max - 1.0) * t
        elif mode == "zoom_out":
            sc = scale_max - (scale_max - 1.0) * t
        else:
            sc = 1.08

        win_w, win_h = w, h
        max_x = max(0, bw - win_w)
        max_y = max(0, bh - win_h)

        if mode == "pan_right":
            x, y = int(max_x * t), max_y // 2
        elif mode == "pan_left":
            x, y = int(max_x * (1 - t)), max_y // 2
        else:
            x, y = max_x // 2, max_y // 2

        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        crop = big[y:y + win_h, x:x + win_w]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = cv2.resize(crop, (w, h))
        writer.write(crop)

        if progress_callback and i % 5 == 0:
            progress_callback(i / total, f"Кадр {i}/{total}")

    writer.release()
    out = make_browser_compatible(raw_path)
    if progress_callback:
        progress_callback(1.0, "Готово")
    return out
