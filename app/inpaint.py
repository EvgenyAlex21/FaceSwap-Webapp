"""
Региональный инпейнт по маске (кисть / ластик).
Базовый режим — OpenCV (всегда работает).
Опционально — AI-инпейнт (диффузия) с поддержкой локальных .safetensors моделей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .utils import OUTPUTS, unique_name, write_image, MODELS


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


# === AI-инпейнт (диффузия) ================================

_pipe = None
_pipe_device = None
_pipe_model_name = None


def _check_diffusers_inpaint() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionInpaintPipeline  # noqa: F401
        from diffusers import StableDiffusionXLInpaintPipeline  # noqa: F401
    except ImportError:
        return False, (
            "Для AI-инпейнта нужно:\n"
            "  pip install diffusers transformers accelerate torch\n"
            "И модель при первом запуске (~4 GB)."
        )
    return True, ""


def get_inpaint_pipeline(device: str = "cuda", model_choice: str = "Realistic Vision (SD 1.5)"):
    global _pipe, _pipe_device, _pipe_model_name

    # Определяем файл модели и тип пайплайна
    if "TryOffAnyone" in model_choice:
        local_model_file = MODELS / "tryOffAnyone.safetensors"
        pipe_type = "sd15"
    elif "LUSTIFY" in model_choice:
        local_model_file = MODELS / "lustifySDXLNSFW_v20-inpainting.safetensors"
        pipe_type = "sdxl"
    else: # Realistic Vision
        local_model_file = MODELS / "realisticVisionV60B1_v50VAE-inpainting.safetensors"
        pipe_type = "sd15"

    # Если модель уже загружена и это та же модель — возвращаем её
    if _pipe is not None and _pipe_device == device and _pipe_model_name == str(local_model_file):
        return _pipe

    ok, err = _check_diffusers_inpaint()
    if not ok:
        raise RuntimeError(err)

    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    # ВАЖНО: dtype объявляется ДО его использования
    dtype = torch.float16 if device == "cuda" else torch.float32

    if not local_model_file.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {local_model_file}\n"
            "Скачайте её с Hugging Face и положите в папку models."
        )

    print(f"[InpaintAI] Загрузка ЛОКАЛЬНОЙ модели {local_model_file.name} ({pipe_type}) на {device}...")

    if pipe_type == "sdxl":
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            str(local_model_file),
            torch_dtype=dtype,
            safety_checker=None,
            load_in_8bit=True,
        )
    else:
        if "TryOffAnyone" in model_choice:
            # Для TryOffAnyone подгружаем текстовый энкодер отдельно от SD 1.5
            from transformers import CLIPTextModel, CLIPTokenizer
            print("[InpaintAI] Подгрузка текстового энкодера для TryOffAnyone...")
            text_encoder = CLIPTextModel.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                subfolder="tokenizer",
            )
            pipe = StableDiffusionInpaintPipeline.from_single_file(
                str(local_model_file),
                torch_dtype=dtype,
                safety_checker=None,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )
        else:
            pipe = StableDiffusionInpaintPipeline.from_single_file(
                str(local_model_file),
                torch_dtype=dtype,
                safety_checker=None,
            )

    # Логика выгрузки в память (вызывается ОДИН раз)
    if pipe_type == "sdxl":
        print("[InpaintAI] Включаем Sequential CPU Offload для SDXL (сохраняет память)...")
        pipe.enable_sequential_cpu_offload()
    else:
        print("[InpaintAI] Переносим SD 1.5 на видеокарту...")
        pipe = pipe.to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    _pipe, _pipe_device, _pipe_model_name = pipe, device, str(local_model_file)
    print(f"[InpaintAI] Модель {local_model_file.name} готова.")
    return _pipe


def _resize_for_diffusion(img_bgr, mask, max_side=768):
    """Приводим к кратному 8 и ограничиваем размер, чтобы влезло в VRAM."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    nw, nh = max(nw, 64), max(nh, 64)
    img_r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    m_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    _, m_r = cv2.threshold(m_r, 10, 255, cv2.THRESH_BINARY)
    return img_r, m_r, (w, h)


def inpaint_ai(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt: str = "nude body, realistic skin, natural anatomy, high detail, photorealistic",
    negative: str = (
        "clothes, clothing, fabric, text, watermark, logo, deformed, "
        "extra limbs, bad anatomy, low quality, blurry"
    ),
    steps: int = 30,
    guidance: float = 7.5,
    strength: float = 1.0,
    seed: int = 0,
    prefer_cuda: bool = True,
    model_choice: str = "Realistic Vision (SD 1.5)",
) -> Path:
    """
    AI-инпейнт по маске. Использует выбранную модель.
    """
    import torch
    from PIL import Image

    m = normalize_mask(mask, image_bgr.shape[:2])
    if int(m.sum()) == 0:
        raise ValueError("Маска пустая — закрасьте область кистью")

    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    pipe = get_inpaint_pipeline(device, model_choice)

    # Размер входа зависит от архитектуры (SDXL требует 1024, SD15 - 512/768)
    #target_size = 1024 if "LUSTIFY" in model_choice else 768
    #target_size = 768 if "LUSTIFY" in model_choice else 512
    target_size = 512  # для всех моделей, чтобы гарантированно влезло в 4GB
    img_r, m_r, (orig_w, orig_h) = _resize_for_diffusion(image_bgr, m, max_side=target_size)

    rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGB")
    pil_mask = Image.fromarray(m_r).convert("L")

    gen = torch.Generator(device=device)
    if seed and seed > 0:
        gen = gen.manual_seed(int(seed))
    else:
        gen = gen.manual_seed(42)

    # Обработка промптов
    actual_prompt = prompt
    actual_negative = negative
    
    if "TryOffAnyone" in model_choice:
        # Эта модель не использует текстовый промпт, 
        # но пайплайн может требовать передачи хоть какой-то строки.
        actual_prompt = "high quality, detailed"
        actual_negative = ""

    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    result = pipe(
        prompt=actual_prompt,
        negative_prompt=actual_negative,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        generator=gen,
    ).images[0]

    out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    # Смешиваем только внутри маски (края — плавно), чтобы фон не поплыл
    alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.0)[..., None]
    final = (out.astype(np.float32) * alpha + image_bgr.astype(np.float32) * (1 - alpha))
    final = np.clip(final, 0, 255).astype(np.uint8)

    out_path = OUTPUTS / unique_name("inpaint_ai", ".png")
    write_image(out_path, final, fmt="png")
    return out_path