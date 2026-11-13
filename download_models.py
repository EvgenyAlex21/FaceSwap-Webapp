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
    "buffalo_l.zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
}

# Опциональные (для вкладки "Выделение" / Инпейнт)
OPTIONAL = {
    "lustifySDXLNSFW_v20-inpainting.safetensors": "https://huggingface.co/andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING/resolve/main/lustifySDXLNSFW_v20-inpainting.safetensors?download=true",
    "tryOffAnyone.safetensors": "https://huggingface.co/ixarchakos/tryOffAnyone/resolve/main/model.safetensors?download=true",
    "realisticVisionV60B1_v50VAE-inpainting.safetensors": "https://huggingface.co/75dhsx/SG161222/resolve/0d09bc9938f3404178908aaeb6c9673d640910c1/realisticVisionV60B1_v50VAE-inpainting.safetensors?download=true",
}


def download_file(url: str, dest: Path, force: bool = False):
    if dest.exists() and not force:
        if dest.stat().st_size > 1024 * 1024:
            print(f"[skip] {dest.name} уже существует")
            return
        else:
            print(f"[warn] {dest.name} слишком маленький, перекачиваем...")
            dest.unlink()

    print(f"[download] {dest.name} ...")
    try:
        def _report(block_num, block_size, total_size):
            if total_size > 0:
                percent = int(block_num * block_size * 100 / total_size)
                if percent % 10 == 0:
                    sys.stdout.write(f"\r  ... {percent}%")
                    sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook=_report)
        print(f"\n[ok] {dest.name}")
    except Exception as e:
        print(f"\n[error] Не удалось скачать {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
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

    # 1. Основные модели (Face Swap)
    print("--- Основные модели (Face Swap) ---")
    buffalo_zip = MODELS_DIR / "buffalo_l.zip"
    download_file(MODELS["buffalo_l.zip"], buffalo_zip, force)
    extract_buffalo(buffalo_zip)

    download_file(MODELS["inswapper_128.onnx"], MODELS_DIR / "inswapper_128.onnx", force)

    # 2. Опциональные модели (Инпейнт)
    print("\n--- Модели для Инпейнта (Выделение) ---")
    for filename, url in OPTIONAL.items():
        download_file(url, MODELS_DIR / filename, force)

    print("\n=== Готово! Все модели лежат в папке models/ ===")
    print("Теперь можно запускать: python app/main.py")


if __name__ == "__main__":
    main()