"""Редактор фото по текстовому промпту."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image

from .utils import MODELS, OUTPUTS, unique_name
from .vram import (
    free_vram, vram_report, is_oom, activate, OffloadedTextEncoder,
)
from .translate import to_english

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
KLEIN_DIR = MODELS / "flux2-klein-4b"
KLEIN_GGUF_DIR = MODELS / "flux2-klein-4b-gguf"
KLEIN_QUANTS = ["Q4_K_M", "Q3_K_M", "Q5_K_M"]

IP2P_REPO = "timbrooks/instruct-pix2pix"

EDITOR_MODELS = {
    "FLUX.2 klein 4B — качественно (рекомендуется)": "klein",
    "InstructPix2Pix — быстро и проще": "ip2p",
}

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


def fit_size(size: Tuple[int, int], max_side: int, multiple: int) -> Tuple[int, int]:
    w, h = size
    scale = min(1.0, max_side / float(max(w, h)))
    nw = max(multiple * 8, int(round(w * scale / multiple)) * multiple)
    nh = max(multiple * 8, int(round(h * scale / multiple)) * multiple)
    return nw, nh


def size_ladder(max_side: int) -> list[int]:
    ladder = [max_side]
    for s in (int(max_side * 0.8), int(max_side * 0.64), 384, 320):
        if s < ladder[-1] and s >= 256:
            ladder.append(s)
    return ladder


# =====================================================================
class KleinEditor:
    def __init__(self):
        self.pipe = None
        self.quant: Optional[str] = None
        self._emb_cache: dict[str, object] = {}

    @staticmethod
    def ensure_files(quant: str = "Q4_K_M", progress: ProgressFn = _noop) -> Tuple[Path, Path]:
        from huggingface_hub import hf_hub_download, snapshot_download
        if not (KLEIN_DIR / "model_index.json").exists() or not (KLEIN_DIR / "text_encoder").exists():
            progress(0.02, "Скачивание FLUX.2 klein (энкодер, VAE, ~9 ГБ, один раз)...")
            snapshot_download(
                KLEIN_REPO, local_dir=str(KLEIN_DIR),
                allow_patterns=[
                    "model_index.json", "scheduler/*", "tokenizer/*",
                    "text_encoder/*", "vae/*", "transformer/config.json",
                ],
            )
        gguf_name = f"flux-2-klein-4b-{quant}.gguf"
        gguf_path = KLEIN_GGUF_DIR / gguf_name
        if not gguf_path.exists():
            progress(0.05, f"Скачивание трансформера {quant} (~2.6 ГБ)...")
            hf_hub_download(KLEIN_GGUF_REPO, filename=gguf_name, local_dir=str(KLEIN_GGUF_DIR))
        return KLEIN_DIR, gguf_path

    def unload(self) -> None:
        self.pipe = None
        self.quant = None
        free_vram()
        _hard_cleanup()

    def load(self, quant: str, progress: ProgressFn = _noop) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 klein требует видеокарту NVIDIA с CUDA")

        activate("editor-klein", self.unload)
        if self.pipe is not None and self.quant == quant:
            return
        if self.pipe is not None:
            self.unload()

        # Жёсткая очистка перед загрузкой
        _hard_cleanup()

        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig

        root, gguf_path = self.ensure_files(quant, progress)
        progress(0.10, "Загрузка модели в память...")
        try:
            transformer = Flux2Transformer2DModel.from_single_file(
                str(gguf_path),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                config=str(root),
                subfolder="transformer",
            )
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить GGUF-трансформер FLUX.2 klein: {e}\n"
                "Обновите библиотеки или выберите InstructPix2Pix."
            ) from e
        pipe = Flux2KleinPipeline.from_pretrained(
            str(root), transformer=transformer, text_encoder=None, torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        for fn in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, fn)()
            except Exception:
                pass
        self.pipe = pipe
        self.quant = quant
        vram_report("после загрузки klein")

    def _encode(self, prompt_en: str, progress: ProgressFn):
        import torch
        from torch import nn

        if prompt_en in self._emb_cache:
            return self._emb_cache[prompt_en]

        from transformers import Qwen3ForCausalLM

        progress(0.20, "Кодирование промпта (Qwen3, слой за слоем)...")
        te = Qwen3ForCausalLM.from_pretrained(
            str(KLEIN_DIR / "text_encoder"),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        te.eval()
        te.lm_head = nn.Identity()

        object.__setattr__(self.pipe, "text_encoder", te)
        try:
            with OffloadedTextEncoder(te, "cuda"), torch.no_grad():
                emb, _ids = self.pipe.encode_prompt(
                    prompt=prompt_en,
                    device=torch.device("cuda"),
                )
            emb = emb.detach().to("cpu")
        finally:
            object.__setattr__(self.pipe, "text_encoder", None)
            del te
            _hard_cleanup()
            free_vram()
        if len(self._emb_cache) >= 4:
            self._emb_cache.pop(next(iter(self._emb_cache)))
        self._emb_cache[prompt_en] = emb
        return emb

    def run(self, image: Image.Image, prompt_en: str, max_side: int, steps: int,
            seed: int, quant: str, progress: ProgressFn = _noop) -> Tuple[Image.Image, Tuple[int, int]]:
        import torch

        self.load(quant, progress)
        _hard_cleanup()

        emb = self._encode(prompt_en, progress)
        _hard_cleanup()

        image = image.convert("RGB")
        ladder = size_ladder(max_side)

        last_err: Optional[BaseException] = None
        for attempt, side in enumerate(ladder):
            w, h = fit_size(image.size, side, 16)
            img = image.resize((w, h), Image.LANCZOS)
            gen = torch.Generator("cpu").manual_seed(int(seed))

            def cb(pipe, i, t, kwargs, _n=steps):
                progress(0.35 + 0.6 * (i + 1) / _n, f"Редактирование: шаг {i + 1}/{_n} ({w}×{h})")
                return kwargs

            try:
                progress(0.35, f"Редактирование ({w}×{h})...")
                with torch.no_grad():
                    out = self.pipe(
                        image=img,
                        prompt_embeds=emb.to("cuda", torch.bfloat16),
                        height=h, width=w,
                        num_inference_steps=int(steps),
                        guidance_scale=1.0,
                        generator=gen,
                        callback_on_step_end=cb,
                    ).images[0]

                del emb
                _hard_cleanup()
                self.unload()
                return out, (w, h)
            except Exception as e:
                if is_oom(e) and attempt < len(ladder) - 1:
                    last_err = e
                    try:
                        self.pipe.maybe_free_model_hooks()
                    except Exception:
                        pass
                    _hard_cleanup()
                    progress(0.35, f"Не хватило видеопамяти — уменьшаю до {ladder[attempt + 1]} px...")
                    continue
                raise
        raise last_err or RuntimeError("Не удалось выполнить редактирование")


# =====================================================================
class Ip2pEditor:
    def __init__(self):
        self.pipe = None

    def unload(self) -> None:
        self.pipe = None
        free_vram()
        _hard_cleanup()

    def load(self, progress: ProgressFn = _noop) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Нужна видеокарта NVIDIA с CUDA")
        activate("editor-ip2p", self.unload)
        if self.pipe is not None:
            return

        # Жёсткая очистка перед загрузкой
        _hard_cleanup()

        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        progress(0.10, "Загрузка InstructPix2Pix...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            IP2P_REPO, torch_dtype=torch.float16, use_safetensors=True,
        )
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        self.pipe = pipe
        vram_report("после загрузки ip2p")

    def run(self, image: Image.Image, prompt_en: str, max_side: int, steps: int,
            image_guidance: float, text_guidance: float, seed: int,
            progress: ProgressFn = _noop) -> Tuple[Image.Image, Tuple[int, int]]:
        import torch

        self.load(progress)
        _hard_cleanup()

        image = image.convert("RGB")
        ladder = size_ladder(min(max_side, 768))
        last_err: Optional[BaseException] = None
        for attempt, side in enumerate(ladder):
            w, h = fit_size(image.size, side, 8)
            img = image.resize((w, h), Image.LANCZOS)
            gen = torch.Generator("cpu").manual_seed(int(seed))

            def cb(pipe, i, t, kwargs, _n=steps):
                progress(0.2 + 0.75 * (i + 1) / _n, f"Редактирование: шаг {i + 1}/{_n} ({w}×{h})")
                return kwargs

            try:
                with torch.no_grad():
                    res = self.pipe(
                        prompt_en, image=img,
                        num_inference_steps=int(steps),
                        image_guidance_scale=float(image_guidance),
                        guidance_scale=float(text_guidance),
                        generator=gen,
                        callback_on_step_end=cb,
                    )
                _hard_cleanup()
                self.unload()
                return res.images[0], (w, h)
            except Exception as e:
                if is_oom(e) and attempt < len(ladder) - 1:
                    last_err = e
                    try:
                        self.pipe.maybe_free_model_hooks()
                    except Exception:
                        pass
                    _hard_cleanup()
                    continue
                raise
        raise last_err or RuntimeError("Не удалось выполнить редактирование")


# =====================================================================
_klein: Optional[KleinEditor] = None
_ip2p: Optional[Ip2pEditor] = None


def edit_image(
    image: Image.Image,
    prompt: str,
    model_key: str = "klein",
    max_side: int = 640,
    steps: int = 4,
    seed: int = -1,
    quant: str = "Q4_K_M",
    image_guidance: float = 1.5,
    text_guidance: float = 7.5,
    translate: bool = True,
    restore_size: bool = False,
    progress: ProgressFn = _noop,
) -> Tuple[Image.Image, Path, str]:
    global _klein, _ip2p

    prompt = (prompt or "").strip()
    if image is None:
        raise ValueError("Загрузите фото")
    if not prompt:
        raise ValueError("Напишите, что нужно изменить на фото")

    t0 = time.time()
    translated = False
    prompt_en = prompt
    if translate:
        progress(0.01, "Перевод промпта...")
        prompt_en, translated = to_english(prompt)

    if seed is None or int(seed) < 0:
        seed = int.from_bytes(__import__("os").urandom(3), "big")

    orig_size = image.size
    if model_key == "ip2p":
        _ip2p = _ip2p or Ip2pEditor()
        out, (w, h) = _ip2p.run(image, prompt_en, max_side, steps or 20, image_guidance, text_guidance, seed, progress)
    else:
        _klein = _klein or KleinEditor()
        out, (w, h) = _klein.run(image, prompt_en, max_side, steps or 4, seed, quant, progress)

    if restore_size and out.size != orig_size:
        out = out.resize(orig_size, Image.LANCZOS)

    path = OUTPUTS / unique_name("edit", ".png")
    out.save(path)
    info = f"Готово за {time.time() - t0:.0f} с · {out.size[0]}×{out.size[1]} · seed {seed}"
    if translated:
        info += f"\n\nПромпт для модели: *{prompt_en}*"
    return out, path, info