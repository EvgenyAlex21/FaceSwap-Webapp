"""Управление памятью: на 4 ГБ VRAM одновременно держим только одну тяжёлую модель."""

from __future__ import annotations

import gc


def free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def vram_report(tag: str = "") -> str:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            msg = f"[VRAM]{' ' + tag if tag else ''} свободно {free / 1024**3:.2f} из {total / 1024**3:.2f} ГБ"
            print(msg, flush=True)
            return msg
    except Exception:
        pass
    return ""


def unload_faceswap() -> None:
    try:
        from app.swapper import _swapper
        if _swapper is not None and _swapper._initialized:
            _swapper.app = None
            _swapper.swapper = None
            _swapper._initialized = False
            print("[VRAM] Модели face-swap выгружены", flush=True)
    except Exception as e:
        print(f"[VRAM] Не удалось выгрузить face-swap: {e}", flush=True)
    free_vram()


def is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return "outofmemory" in name.lower() or "out of memory" in text or "cuda error: out of memory" in text


class OffloadedTextEncoder:

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    def __enter__(self):
        import torch
        from accelerate import cpu_offload
        if torch.cuda.is_available():
            cpu_offload(self.model, execution_device=torch.device(self.device), offload_buffers=True)
        return self.model

    def __exit__(self, *exc):
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(self.model, recurse=True)
        except Exception:
            pass
        return False


_active = {"name": None, "unload": None}


def activate(name: str, unload_fn) -> None:
    """Перед загрузкой модели `name` выгружает ранее активную (если это другая)."""
    prev = _active["name"]
    if prev is not None and prev != name and _active["unload"] is not None:
        print(f"[VRAM] Выгружаю «{prev}» ради «{name}»", flush=True)
        try:
            _active["unload"]()
        except Exception as e:
            print(f"[VRAM] Ошибка выгрузки {prev}: {e}", flush=True)
    if prev != name:
        unload_faceswap()
    _active["name"] = name
    _active["unload"] = unload_fn
    free_vram()
