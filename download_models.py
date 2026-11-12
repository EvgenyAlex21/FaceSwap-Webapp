#!/usr/bin/env python3
"""
Скачивает необходимые модели для локального face-swap.
Запускать один раз после установки зависимостей.
"""

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Основные модели
MODELS = {
    # Детектор + распознавание (buffalo_l)
    "buffalo_l.zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    # InSwapper (основная модель замены лица)
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
}

# Опциональные (улучшение лица)
OPTIONAL = {
    # Можно добавить GFPGAN / CodeFormer позже
}


def download_file(url: str, dest: Path, force: bool = False):
    if dest.exists() and not force:
        print(f"[skip] {dest.name} уже существует")
        return

    print(f"[download] {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[ok] {dest.name}")
    except Exception as e:
        print(f"[error] Не удалось скачать {dest.name}: {e}")
        sys.exit(1)


def extract_buffalo(zip_path: Path):
    import zipfile
    extract_to = MODELS_DIR / "buffalo_l"
    if extract_to.exists():
        print("[skip] buffalo_l уже распакован")
        return
    print("[extract] buffalo_l.zip ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(MODELS_DIR)
    print("[ok] buffalo_l распакован")


def main():
    force = "--force" in sys.argv
    print("=== Local Face Swap — загрузка моделей ===\n")

    # buffalo_l
    buffalo_zip = MODELS_DIR / "buffalo_l.zip"
    download_file(MODELS["buffalo_l.zip"], buffalo_zip, force)
    extract_buffalo(buffalo_zip)

    # inswapper
    download_file(MODELS["inswapper_128.onnx"], MODELS_DIR / "inswapper_128.onnx", force)

    print("\n=== Готово! Модели лежат в папке models/ ===")
    print("Теперь можно запускать: python app/main.py")


if __name__ == "__main__":
    main()