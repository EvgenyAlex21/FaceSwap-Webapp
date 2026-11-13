# Региональный инпейнт по маске. AI-инпейнт (диффузия) с поддержкой локальных .safetensors моделей.


from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .utils import OUTPUTS, unique_name, write_image, MODELS


def normalize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        raise ValueError("Маска пустая — закрасьте область кистью")
    m = mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY) if m.shape[2] >= 3 else m[:, :, 0]
    h, w = target_hw
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
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


# ================================ AI-инпейнт (диффузия) ================================

_pipe = None
_pipe_device = None
_pipe_model_name = None


def _check_diffusers_inpaint() -> tuple[bool, str]:
    try:
        import torch  
        from diffusers import StableDiffusionInpaintPipeline  
        from diffusers import StableDiffusionXLInpaintPipeline  
    except ImportError:
        return False, (
            "Для AI-инпейнта нужно:\n"
            "  pip install diffusers transformers accelerate torch\n"
            "И модель при первом запуске (~4 GB)."
        )
    return True, ""


def _inspect_tryoff_checkpoint(path: Path) -> dict:
    """Определяет тип содержимого .safetensors: полный чекпоинт, LoRA, attention-веса или мусор."""
    result = {"ok": False, "kind": "unknown", "reason": ""}
    try:
        from safetensors.torch import load_file
    except Exception as e:
        result["reason"] = f"Не удалось импортировать safetensors: {e}"
        return result
    try:
        sd = load_file(str(path))
    except Exception as e:
        result["reason"] = f"Не удалось прочитать .safetensors: {e}"
        return result
    if not sd:
        result["reason"] = "Файл пустой"
        return result

    keys = list(sd.keys())
    has_unet = any(k.startswith("model.diffusion_model.") for k in keys)
    has_first_stage = any(k.startswith("first_stage_model.") for k in keys)
    is_lora = any(
        ("lora" in k.lower()) or k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight")
        for k in keys
    )
    # TryOffAnyone: набор attention-весов вида 0.0.*, 1.0.*, ... (proj_in/proj_out/norm/to_q/to_k/to_v/to_out)
    looks_like_attention = (
        any(k.endswith(".proj_in.weight") or k.endswith(".proj_out.weight") for k in keys)
        and any(("to_q" in k) or ("to_k" in k) or ("to_v" in k) or ("to_out" in k) for k in keys)
    )

    if has_unet and has_first_stage:
        result.update(ok=True, kind="checkpoint")
        return result
    if is_lora:
        result.update(ok=True, kind="lora")
        return result
    if looks_like_attention:
        result.update(ok=True, kind="attention")
        return result
    result["reason"] = (
        "Файл не похож ни на полный SD 1.5 inpaint-чекпоинт, ни на LoRA, "
        "ни на attention-веса TryOffAnyone. "
        f"Пример ключей: {keys[:5]}"
    )
    return result


def _build_tryoffanyone_pipeline(local_model_file: Path, dtype):
    """Собирает пайплайн TryOffAnyone по официальному рецепту:
    SD 1.5 inpaint UNet + VAE sd-vae-ft-mse + fine-tuned attention weights,
    cross-attention отключён (текст не используется).
    """
    import torch
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionInpaintPipeline,
        UNet2DConditionModel,
    )
    from diffusers.models.attention_processor import AttnProcessor

    print("[InpaintAI] Сборка TryOffAnyone: SD 1.5 inpaint UNet + VAE + attention weights...")

    # 1. Базовый UNet из SD 1.5 inpainting
    unet = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        subfolder="unet",
        torch_dtype=dtype,
    )

    # 2. VAE — официальный TryOffAnyone использует sd-vae-ft-mse
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=dtype,
    )

    # 3. Загружаем fine-tuned веса из локального файла
    from safetensors.torch import load_file
    state_dict = load_file(str(local_model_file))

    # Пробуем загрузить как есть. Если ключи не подходят напрямую —
    # используем set_attn_processor с кастомным процессором.
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    print(f"[InpaintAI] TryOffAnyone: загружено весов, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0 and len(unexpected) == 0:
        # Ничего не загрузилось — ключи не совпадают с именами UNet
        raise RuntimeError(
            "Веса tryOffAnyone не совпадают с UNet SD 1.5 inpaint. "
            "Проверьте, что файл — официальный model.safetensors из "
            "https://huggingface.co/ixarchakos/tryOffAnyone . "
            f"Пример ключей в файле: {list(state_dict.keys())[:5]}"
        )

    # 4. Отключаем cross-attention (текстовый промпт не используется)
    unet.set_attn_processor(AttnProcessor())

    # 5. Scheduler
    scheduler = DDIMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        subfolder="scheduler",
    )

    # 6. Собираем пайплайн без text_encoder/tokenizer
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    return pipe


def get_inpaint_pipeline(device: str = "cuda", model_choice: str = "Realistic Vision (SD 1.5)"):
    global _pipe, _pipe_device, _pipe_model_name

    if "TryOffAnyone" in model_choice:
        local_model_file = MODELS / "tryOffAnyone.safetensors"
        pipe_type = "tryoff"
    elif "LUSTIFY" in model_choice:
        local_model_file = MODELS / "lustifySDXLNSFW_v20-inpainting.safetensors"
        pipe_type = "sdxl"
    else: 
        local_model_file = MODELS / "realisticVisionV60B1_v50VAE-inpainting.safetensors"
        pipe_type = "sd15"

    if _pipe is not None and _pipe_device == device and _pipe_model_name == str(local_model_file):
        return _pipe

    ok, err = _check_diffusers_inpaint()
    if not ok:
        raise RuntimeError(err)

    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32

    if not local_model_file.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {local_model_file}\n"
            "Скачайте её с Hugging Face и положите в папку models."
        )

    print(f"[InpaintAI] Загрузка ЛОКАЛЬНОЙ модели {local_model_file.name} ({pipe_type}) на {device}...")

    if pipe_type == "sdxl":
        # SDXL: не передаём load_in_8bit (его нет в from_single_file),
        # включаем memory-friendly режимы.
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            str(local_model_file),
            torch_dtype=dtype,
            safety_checker=None,
        )
        print("[InpaintAI] SDXL: sequential offload + VAE slicing/tiling + attention slicing...")
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as e:
            print(f"[InpaintAI] sequential_cpu_offload: {e}")
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pass

    elif pipe_type == "tryoff":
        info = _inspect_tryoff_checkpoint(local_model_file)
        if not info["ok"]:
            raise RuntimeError(
                "tryOffAnyone.safetensors не подходит ни для from_single_file, "
                "ни для attention-весов.\n"
                f"Причина: {info['reason']}\n"
                "Официальный файл: https://huggingface.co/ixarchakos/tryOffAnyone/resolve/main/model.safetensors"
            )
        if info["kind"] == "checkpoint":
            # На случай, если пользователь подложил полный SD 1.5 inpaint чекпоинт
            pipe = StableDiffusionInpaintPipeline.from_single_file(
                str(local_model_file),
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipe = pipe.to(device)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        elif info["kind"] == "lora":
            # LoRA: базовый SD 1.5 inpaint + load_lora_weights
            print("[InpaintAI] tryOffAnyone определён как LoRA — накатываем на SD 1.5 inpaint...")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                subfolder="",
                torch_dtype=dtype,
                safety_checker=None,
            )
            try:
                pipe.load_lora_weights(str(local_model_file))
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить LoRA tryOffAnyone: {e}")
            pipe = pipe.to(device)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        else:
            # attention-веса: собираем пайплайн вручную
            pipe = _build_tryoffanyone_pipeline(local_model_file, dtype)
            pipe = pipe.to(device)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

    else:
        # SD 1.5 обычный чекпоинт (Realistic Vision)
        pipe = StableDiffusionInpaintPipeline.from_single_file(
            str(local_model_file),
            torch_dtype=dtype,
            safety_checker=None,
        )
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
    
    import torch
    from PIL import Image

    m = normalize_mask(mask, image_bgr.shape[:2])
    if int(m.sum()) == 0:
        raise ValueError("Маска пустая — закрасьте область кистью")

    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    pipe = get_inpaint_pipeline(device, model_choice)

    # Размер входа зависит от архитектуры и объёма VRAM.
    # Для 4 ГБ VRAM SDXL держим на 640 и ограничиваем шаги.
    if "LUSTIFY" in model_choice:
        target_size = 640
        steps = min(int(steps), 20)
    elif "TryOffAnyone" in model_choice:
        # TryOffAnyone — SD 1.5, 512 — родное разрешение
        target_size = 512
    else:
        target_size = 512

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
    kwargs = {}
    
    if "TryOffAnyone" in model_choice:
        # Эта модель не использует текстовый промпт (text_encoder=None).
        # Передаём пустые тензоры эмбеддингов, чтобы пайплайн не пытался
        # токенизировать строки через None-tokenizer.
        # Размерность: (batch_size, seq_len, hidden_size) = (1, 77, 768) для SD 1.5
        actual_prompt = None
        actual_negative = None
        kwargs["prompt_embeds"] = torch.zeros((1, 77, 768), dtype=torch.float16, device=device)
        kwargs["negative_prompt_embeds"] = torch.zeros((1, 77, 768), dtype=torch.float16, device=device)

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    call_kwargs = {
        "image": pil_img,
        "mask_image": pil_mask,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "strength": float(strength),
        "generator": gen,
    }
    if actual_prompt is not None:
        call_kwargs["prompt"] = actual_prompt
        call_kwargs["negative_prompt"] = actual_negative
    call_kwargs.update(kwargs)

    result = pipe(**call_kwargs).images[0]

    out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.0)[..., None]
    final = (out.astype(np.float32) * alpha + image_bgr.astype(np.float32) * (1 - alpha))
    final = np.clip(final, 0, 255).astype(np.uint8)

    out_path = OUTPUTS / unique_name("inpaint_ai", ".png")
    write_image(out_path, final, fmt="png")
    return out_path