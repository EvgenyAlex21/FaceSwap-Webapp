"""«Преображение» — оживление фото по текстовому описанию (image-to-video).

Модель: LTX-Video 0.9.5 (2B параметров, Lightricks), diffusers LTXConditionPipeline.
Подгонка под 4 ГБ VRAM:
  * трансформер хранится в fp8 (layerwise casting) и выгружается на CPU послойно (group offload);
  * VAE работает тайлами с послойной выгрузкой;
  * T5-XXL (≈9.5 ГБ) считается слой за слоем на GPU и удаляется из памяти сразу после
    кодирования промпта (эмбеддинги кешируются);
  * при нехватке видеопамяти разрешение/длина клипа автоматически уменьшаются.
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
from .vram import free_vram, vram_report, is_oom, activate, OffloadedTextEncoder
from .translate import to_english

LTX_REPO = "Lightricks/LTX-Video-0.9.5"
LTX_DIR = MODELS / "ltx-video-0.9.5"
FPS = 24

# (макс. сторона, кадров) — кадров всегда 8k+1
PRESETS = {
    "Быстро — до 512 px, 2 сек": (512, 49),
    "Баланс — до 640 px, 3 сек": (640, 73),
    "Качество — до 768 px, 4 сек": (768, 97),
}
UPSCALES = {"Как есть": 1.0, "×1.5": 1.5, "×2": 2.0}

DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, subtitles"
QUALITY_SUFFIX = "The video is high quality, smooth natural motion, realistic lighting and details."

ProgressFn = Callable[[float, str], None]


def _noop(p: float, msg: str = "") -> None:
    pass


def _hard_cleanup():
    """Жёсткая очистка RAM и VRAM."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def fit_size_32(size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    w, h = size
    scale = min(1.0, max_side / float(max(w, h)))
    nw = max(256, int(round(w * scale / 32)) * 32)
    nh = max(256, int(round(h * scale / 32)) * 32)
    return nw, nh


def _codec_like(img: Image.Image, quality: int = 80) -> Image.Image:
    """LTX обучалась на сжатых видео: «слишком чистый» кадр часто даёт почти статичный клип.
    JPEG-проход имитирует артефакты сжатия (эвристика, как рекомендуют авторы модели)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class Animator:
    def __init__(self):
        self.pipe = None
        self._emb_cache: dict[tuple, tuple] = {}

    # ---------- файлы ----------
    @staticmethod
    def ensure_files(progress: ProgressFn = _noop) -> Path:
        """Проверяет, что скачаны ВСЕ ключевые файлы, а не только папки.
        Если чего-то не хватает — докачивает через snapshot_download (resume)."""
        required = [
            LTX_DIR / "model_index.json",
            LTX_DIR / "text_encoder" / "config.json",
            LTX_DIR / "text_encoder" / "model-00001-of-00004.safetensors",
            LTX_DIR / "text_encoder" / "model-00002-of-00004.safetensors",
            LTX_DIR / "text_encoder" / "model-00003-of-00004.safetensors",
            LTX_DIR / "text_encoder" / "model-00004-of-00004.safetensors",
            LTX_DIR / "transformer" / "config.json",
            LTX_DIR / "vae" / "config.json",
        ]
        missing = [p for p in required if not p.exists()]
        if not missing:
            return LTX_DIR

        from huggingface_hub import snapshot_download

        progress(0.02, "Докачивание LTX-Video (не хватает файлов, один раз)...")
        snapshot_download(
            LTX_REPO,
            local_dir=str(LTX_DIR),
            ignore_patterns=[
                "ltx-video-*.safetensors", "*.mp4", "*.gif",
                "*.png", "*.jpg", "*.webp", "assets/*",
            ],
            resume_download=True,
        )
        return LTX_DIR

    # ---------- загрузка ----------
    def unload(self) -> None:
        self.pipe = None
        free_vram()
        _hard_cleanup()

    def load(self, progress: ProgressFn = _noop) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Для анимации нужна видеокарта NVIDIA с CUDA (torch.cuda.is_available() = False)")
        activate("animator-ltx", self.unload)
        if self.pipe is not None:
            return

        from diffusers import LTXConditionPipeline, LTXVideoTransformer3DModel
        from diffusers.hooks import apply_group_offloading

        root = self.ensure_files(progress)
        progress(0.08, "Загрузка модели в память...")
        cuda, cpu = torch.device("cuda"), torch.device("cpu")

        transformer = LTXVideoTransformer3DModel.from_pretrained(
            str(root), subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        # хранение весов в fp8 (≈2 ГБ вместо 4), вычисления в bf16
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
        )
        pipe = LTXConditionPipeline.from_pretrained(
            str(root), transformer=transformer, text_encoder=None, torch_dtype=torch.bfloat16,
        )
        pipe.transformer.enable_group_offload(
            onload_device=cuda, offload_device=cpu, offload_type="leaf_level", use_stream=True,
        )
        apply_group_offloading(pipe.vae, onload_device=cuda, offload_type="leaf_level")
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        self.pipe = pipe
        vram_report("после загрузки LTX")

    # ---------- текст ----------
    def _encode(self, prompt: str, negative: str, progress: ProgressFn):
        import torch

        key = (prompt, negative)
        if key in self._emb_cache:
            return self._emb_cache[key]

        from transformers import T5EncoderModel

        progress(0.18, "Кодирование промпта (T5, слой за слоем — первый раз до минуты)...")
        te = T5EncoderModel.from_pretrained(str(LTX_DIR / "text_encoder"), torch_dtype=torch.bfloat16)
        te.eval()
        object.__setattr__(self.pipe, "text_encoder", te)
        try:
            with OffloadedTextEncoder(te, "cuda"), torch.no_grad():
                pe, pm, ne, nm = self.pipe.encode_prompt(
                    prompt=prompt, negative_prompt=negative,
                    do_classifier_free_guidance=True,
                    device=torch.device("cuda"), dtype=torch.bfloat16,
                    max_sequence_length=256,
                )
            result = tuple(t.detach().to("cpu") for t in (pe, pm, ne, nm))
        finally:
            object.__setattr__(self.pipe, "text_encoder", None)
            del te
            _hard_cleanup()
            free_vram()
        if len(self._emb_cache) >= 3:
            self._emb_cache.pop(next(iter(self._emb_cache)))
        self._emb_cache[key] = result
        return result

    # ---------- генерация ----------
    def run(self, image: Image.Image, prompt: str, negative: str, max_side: int, num_frames: int,
            steps: int, guidance: float, seed: int, codec_prep: bool,
            progress: ProgressFn = _noop):
        import torch

        self.load(progress)
        _hard_cleanup()

        pe, pm, ne, nm = self._encode(prompt, negative, progress)
        _hard_cleanup()

        image = image.convert("RGB")

        # лесенка отката при OOM: (сторона, кадров)
        ladder = [(max_side, num_frames)]
        for side, frames in ((int(max_side * 0.8), num_frames), (int(max_side * 0.8), 49),
                             (384, 49), (320, 33)):
            if (side, frames) not in ladder and side >= 256 and (side < ladder[-1][0] or frames < ladder[-1][1]):
                ladder.append((side, frames))

        last_err: Optional[BaseException] = None
        for attempt, (side, frames) in enumerate(ladder):
            w, h = fit_size_32(image.size, side)
            img = image.resize((w, h), Image.LANCZOS)
            if codec_prep:
                img = _codec_like(img)
            gen = torch.Generator("cpu").manual_seed(int(seed))

            def cb(pipe, i, t, kwargs, _n=steps, _w=w, _h=h, _f=frames):
                progress(0.30 + 0.60 * (i + 1) / _n, f"Анимация: шаг {i + 1}/{_n} ({_w}×{_h}, {_f} кадров)")
                return kwargs

            try:
                progress(0.30, f"Анимация ({w}×{h}, {frames} кадров)...")
                cuda = torch.device("cuda")
                with torch.no_grad():
                    out = self.pipe(
                        image=img, frame_index=0, strength=1.0,
                        prompt_embeds=pe.to(cuda), prompt_attention_mask=pm.to(cuda),
                        negative_prompt_embeds=ne.to(cuda), negative_prompt_attention_mask=nm.to(cuda),
                        width=w, height=h, num_frames=int(frames), frame_rate=FPS,
                        num_inference_steps=int(steps), guidance_scale=float(guidance),
                        image_cond_noise_scale=0.025,
                        decode_timestep=0.05, decode_noise_scale=0.025,
                        generator=gen,
                        callback_on_step_end=cb,
                        output_type="pil",
                    )
                result = out.frames[0]
                _hard_cleanup()
                self.unload()
                return result, (w, h, frames)
            except Exception as e:                    # noqa: BLE001
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
    image: Image.Image,
    prompt: str,
    negative: str = DEFAULT_NEGATIVE,
    preset: str = "Быстро — до 512 px, 2 сек",
    steps: int = 30,
    guidance: float = 3.0,
    seed: int = -1,
    translate: bool = True,
    add_quality: bool = True,
    codec_prep: bool = True,
    upscale_key: str = "Как есть",
    progress: ProgressFn = _noop,
) -> Tuple[Path, str]:
    global _animator

    if image is None:
        raise ValueError("Загрузите фото")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Опишите, что должно происходить в кадре")

    t0 = time.time()
    translated = False
    prompt_en, neg_en = prompt, (negative or "").strip()
    if translate:
        progress(0.01, "Перевод промпта...")
        prompt_en, translated = to_english(prompt)
        neg_en, _ = to_english(neg_en)
    if add_quality:
        prompt_en = prompt_en.rstrip(". ") + ". " + QUALITY_SUFFIX

    if seed is None or int(seed) < 0:
        seed = int.from_bytes(__import__("os").urandom(3), "big")

    max_side, frames = PRESETS.get(preset, PRESETS["Быстро — до 512 px, 2 сек"])
    _animator = _animator or Animator()
    pil_frames, (w, h, n) = _animator.run(
        image, prompt_en, neg_en, max_side, frames, steps, guidance, seed, codec_prep, progress,
    )

    progress(0.93, "Сборка видео...")
    path = _write_video(pil_frames, UPSCALES.get(upscale_key, 1.0))
    info = f"Готово за {time.time() - t0:.0f} с · {w}×{h} · {n} кадров · {n / FPS:.1f} сек · seed {seed}"
    if translated:
        info += f"\n\nПромпт для модели: *{prompt_en}*"
    return path, info