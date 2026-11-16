"""Перевод русских промптов на английский (RU→EN) офлайн.
LTX-Video (T5) и InstructPix2Pix (CLIP) понимают только английский.
Модель Helsinki-NLP/opus-mt-ru-en ~300 МБ, работает на CPU за доли секунды."""

from __future__ import annotations

import re
from pathlib import Path

from .utils import MODELS

REPO = "Helsinki-NLP/opus-mt-ru-en"
LOCAL = MODELS / "opus-mt-ru-en"

_tok = None
_model = None
_failed = False
_CYR = re.compile(r"[а-яёА-ЯЁ]")


def has_cyrillic(text: str) -> bool:
    return bool(_CYR.search(text or ""))


def _load():
    global _tok, _model, _failed
    if _model is not None or _failed:
        return
    try:
        from transformers import MarianMTModel, MarianTokenizer
        src = str(LOCAL) if (LOCAL / "config.json").exists() else REPO
        _tok = MarianTokenizer.from_pretrained(src)
        _model = MarianMTModel.from_pretrained(src).eval()
    except Exception as e:
        _failed = True
        print(f"[Translate] Не удалось загрузить переводчик: {e}", flush=True)


def to_english(text: str) -> tuple[str, bool]:
    """Возвращает (текст, был_ли_переведён). Если переводчик недоступен — исходный текст."""
    text = (text or "").strip()
    if not text or not has_cyrillic(text):
        return text, False
    _load()
    if _model is None:
        return text, False
    try:
        import torch
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
        outs = []
        for p in parts:
            batch = _tok([p], return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                gen = _model.generate(**batch, num_beams=4, max_new_tokens=200)
            outs.append(_tok.decode(gen[0], skip_special_tokens=True))
        return " ".join(outs), True
    except Exception as e:
        print(f"[Translate] Ошибка перевода: {e}", flush=True)
        return text, False
