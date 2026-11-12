"""
Редактор изображения по текстовому промпту.
Использует InstructPix2Pix (diffusers) — опционально.
Если модели нет — понятная ошибка с инструкцией установки.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils import OUTPUTS, unique_name, write_image

_pipe = None
_pipe_device = None


def _check_diffusers():
    try:
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline
        return True
    except ImportError:
        return False


def get_edit_pipeline(device: str = "cuda"):
    global _pipe, _pipe_device
    if _pipe is not None and _pipe_device == device:
        return _pipe

    if not _check_diffusers():
        raise RuntimeError(
            "Для редактирования нужно установить:\n"
            "  pip install diffusers transformers accelerate torch\n"
            "И скачать модель при первом запуске (интернет, ~5 GB)."
        )

    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline

    model_id = "timbrooks/instruct-pix2pix"

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[Editor] Загрузка InstructPix2Pix ({model_id}) на {device}...")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _pipe = pipe
    _pipe_device = device
    print("[Editor] Модель редактирования готова.")
    return _pipe


def edit_image(
    image_bgr: np.ndarray,
    prompt: str,
    steps: int = 20,
    image_guidance: float = 1.5,
    guidance_scale: float = 7.5,
    seed: int = 0,
    prefer_cuda: bool = True,
) -> Path:
    """
    Редактирует изображение по текстовой инструкции.
    Пример prompt: "add sunglasses", "make the person smile", "change hair to blonde"
    """
    import torch
    from PIL import Image

    if not prompt or not prompt.strip():
        raise ValueError("Введите текстовую инструкцию")

    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    pipe = get_edit_pipeline(device)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGB")
    # InstructPix2Pix лучше на ~512
    pil = pil.resize((512, 512), Image.LANCZOS)

    generator = torch.Generator(device=device)
    if seed and seed > 0:
        generator = generator.manual_seed(int(seed))

    result = pipe(
        prompt=prompt.strip(),
        image=pil,
        num_inference_steps=int(steps),
        image_guidance_scale=float(image_guidance),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]

    out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    # Вернуть к исходному размеру
    h, w = image_bgr.shape[:2]
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)

    out_path = OUTPUTS / unique_name("edit", ".png")
    write_image(out_path, out, fmt="png")
    return out_path
