"""«Преображение» — анимация фото через Stable Video Diffusion (SVD).

SVD не принимает текстовые промпты. Чтобы учесть пожелания пользователя,
реализован гибридный подход:
1. Если пользователь ввёл промпт, сначала генерируется изображение через SD 1.5.
2. Затем это изображение (или загруженное пользователем фото) анимируется через SVD.

Оптимизации для 4 ГБ VRAM:
  * FP16 + выгрузка на CPU (enable_model_cpu_offload);
  * UNet feed-forward chunking;
  * decode_chunk_size=1 или 2 для снижения пиков памяти при декодировании VAE.
"""

from __future__ import annotations

import gc
import io
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import MODELS, OUTPUTS, unique_name, create_video_writer, make_browser_compatible
from .vram import free_vram, vram_report, is_oom, activate
from .translate import to_english

SVD_REPO = "stabilityai/stable-video-diffusion-img2vid-xt"
SD15_REPO = "runwayml/stable-diffusion-v1-5" 

FPS = 6  
MOTION_BUCKET_ID = 127  

RESOLUTIONS = {
    "360p (360 px)": 360,
    "480p (480 px)": 480,
    "576p (576 px)": 576,
}

DURATIONS = [1, 2, 3, 4]  

UPSCALES = {"Как есть": 1.0, "×1.5": 1.5, "×2": 2.0}

DEFAULT_NEGATIVE = "worst quality, low quality, blurry, distorted, watermark, text"

ProgressFn = Callable[[float, str], None]


def _noop(p: float, msg: str = "") -> None:
    pass


def _hard_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def fit_size_64(size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    w, h = size
    scale = min(1.0, max_side / float(max(w, h)))
    nw = max(256, int(round(w * scale / 64)) * 64)
    nh = max(256, int(round(h * scale / 64)) * 64)
    return nw, nh


class Animator:
    def __init__(self):
        self.pipe_svd = None
        self.pipe_sd = None

    def unload(self) -> None:
        self.pipe_svd = None
        self.pipe_sd = None
        free_vram()
        _hard_cleanup()

    def load_svd(self, progress: ProgressFn = _noop) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Для анимации нужна видеокарта NVIDIA с CUDA")
        activate("animator-svd", self.unload)
        if self.pipe_svd is not None:
            return

        from diffusers import StableVideoDiffusionPipeline

        progress(0.05, "Загрузка Stable Video Diffusion (SVD-XT)...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            SVD_REPO,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.enable_model_cpu_offload()
        pipe.unet.enable_forward_chunking()
        self.pipe_svd = pipe
        vram_report("после загрузки SVD")

    def load_sd15(self, progress: ProgressFn = _noop) -> None:
        import torch
        from diffusers import StableDiffusionPipeline

        if self.pipe_sd is not None:
            return

        progress(0.02, "Загрузка SD 1.5 для генерации первого кадра...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD15_REPO,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        self.pipe_sd = pipe

    def generate_image_from_prompt(self, prompt: str, negative: str, progress: ProgressFn) -> Image.Image:
        import torch

        self.load_sd15(progress)
        progress(0.15, "Генерация первого кадра по промпту...")

        gen = torch.Generator("cuda").manual_seed(42)
        with torch.no_grad():
            image = self.pipe_sd(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=25,  
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=gen,
            ).images[0]

        self.pipe_sd = None
        _hard_cleanup()
        return image

    def run(self, image: Optional[Image.Image], prompt: str, negative: str,
            max_side: int, duration_sec: float, seed: int,
            progress: ProgressFn = _noop):
        import torch

        if prompt and prompt.strip():
            progress(0.01, "Используется промпт для создания первого кадра...")
            image = self.generate_image_from_prompt(prompt, negative, progress)
        elif image is None:
            raise ValueError("Загрузите фото или введите промпт для генерации.")

        self.load_svd(progress)
        _hard_cleanup()

        image = image.convert("RGB")

        num_frames = min(int(duration_sec * FPS), 25)
        num_frames = max(num_frames, 14) 

        ladder = [(max_side, num_frames)]
        for side, frames in (
            (int(max_side * 0.8), num_frames),
            (480, 14),
            (384, 14),
        ):
            if (side, frames) not in ladder and side >= 256:
                ladder.append((side, frames))

        last_err: Optional[BaseException] = None
        for attempt, (side, frames) in enumerate(ladder):
            w, h = fit_size_64(image.size, side)
            img = image.resize((w, h), Image.LANCZOS)
            gen = torch.Generator("cpu").manual_seed(int(seed))

            try:
                progress(0.30, f"Анимация SVD ({w}×{h}, {frames} кадров)...")
                decode_chunk_size = 1 if attempt > 0 else 2
                with torch.no_grad():
                    frames_out = self.pipe_svd(
                        image=img,
                        num_frames=int(frames),
                        fps=FPS,
                        motion_bucket_id=MOTION_BUCKET_ID,
                        decode_chunk_size=decode_chunk_size,
                        generator=gen,
                    ).frames[0]
                _hard_cleanup()
                self.unload()
                return frames_out, (w, h, frames)
            except Exception as e:
                if is_oom(e) and attempt < len(ladder) - 1:
                    last_err = e
                    _hard_cleanup()
                    nxt = ladder[attempt + 1]
                    progress(0.30, f"Не хватило видеопамяти — пробую {nxt[0]} px / {nxt[1]} кадров...")
                    continue
                raise
        raise last_err or RuntimeError("Не удалось создать анимацию")


_animator: Optional[Animator] = None


def _write_video(frames, upscale: float) -> Path:
    path = OUTPUTS / unique_name("anim", ".mp4")
    first = np.array(frames[0])
    h, w = first.shape[:2]
    ow, oh = int(round(w * upscale)), int(round(h * upscale))
    writer, path, ow, oh = create_video_writer(path, ow, oh, FPS)
    for fr in frames:
        bgr = cv2.cvtColor(np.array(fr.convert("RGB")), cv2.COLOR_RGB2BGR)
        if (bgr.shape[1], bgr.shape[0]) != (ow, oh):
            bgr = cv2.resize(bgr, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        writer.write(bgr)
    writer.release()
    return make_browser_compatible(path, crf=18, preset="medium")


def animate_image(
    image: Optional[Image.Image],
    prompt: str,
    negative: str = DEFAULT_NEGATIVE,
    resolution_key: str = "576p (576 px)",
    duration_sec: float = 2.0,
    steps: int = 25,
    guidance: float = 3.0,
    seed: int = -1,
    translate: bool = True,
    add_quality: bool = True,
    codec_prep: bool = True,
    upscale_key: str = "Как есть",
    progress: ProgressFn = _noop,
) -> Tuple[Path, str]:
    global _animator

    if image is None and not (prompt or "").strip():
        raise ValueError("Загрузите фото или введите промпт")

    t0 = time.time()
    translated = False
    prompt_en, neg_en = prompt, (negative or "").strip()
    if translate and prompt:
        progress(0.01, "Перевод промпта...")
        prompt_en, translated = to_english(prompt)
        if negative:
            neg_en, _ = to_english(neg_en)

    if seed is None or int(seed) < 0:
        seed = int.from_bytes(__import__("os").urandom(3), "big")

    max_side = RESOLUTIONS.get(resolution_key, 576)
    max_side = min(max_side, 576)  # SVD лучше работает при 576
    duration_sec = max(1.0, min(float(duration_sec), 4.0))

    _animator = _animator or Animator()
    pil_frames, (w, h, n) = _animator.run(
        image, prompt_en, neg_en, max_side, duration_sec, seed, progress,
    )

    progress(0.93, "Сборка видео...")
    path = _write_video(pil_frames, UPSCALES.get(upscale_key, 1.0))
    info = f"Готово за {time.time() - t0:.0f} с · {w}×{h} · {n} кадров · {n / FPS:.1f} сек · seed {seed}"
    if translated and prompt_en:
        info += f"\n\nПромпт для модели: *{prompt_en}*"
    return path, info