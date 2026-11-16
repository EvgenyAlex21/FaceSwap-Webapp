#!/usr/bin/env python3
"""
Скачивает модели для локального приложения.

  python download_models.py             # только face-swap (быстро, ~0.6 ГБ)
  python download_models.py --inpaint   # + модели инпейнта (Realistic Vision / LUSTIFY / TryOffAnyone)
  python download_models.py --editor    # + Редактор: FLUX.2 klein 4B GGUF (~12 ГБ)
  python download_models.py --ip2p      # + Редактор (лёгкий): InstructPix2Pix (~5 ГБ)
  python download_models.py --animate   # + Преображение: LTX-Video 0.9.5 (15–25 ГБ)
  python download_models.py --translate # + переводчик RU→EN для промптов (~0.3 ГБ)
  python download_models.py --all       # всё перечисленное
  python download_models.py --force     # перекачать существующие файлы

Скачивание возобновляется после обрыва. Если файлы не скачать заранее, приложение
скачает их само при первом запуске соответствующей вкладки.
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Основные модели (face-swap)
MODELS = {
    "buffalo_l.zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
}

# Опциональные модели для вкладки «Выделение» (инпейнт)
INPAINT = {
    "lustifySDXLNSFW_v20-inpainting.safetensors": "https://huggingface.co/andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING/resolve/main/lustifySDXLNSFW_v20-inpainting.safetensors?download=true",
    "tryOffAnyone.safetensors": "https://huggingface.co/ixarchakos/tryOffAnyone/resolve/main/model.safetensors?download=true",
    "realisticVisionV60B1_v50VAE-inpainting.safetensors": "https://huggingface.co/75dhsx/SG161222/resolve/0d09bc9938f3404178908aaeb6c9673d640910c1/realisticVisionV60B1_v50VAE-inpainting.safetensors?download=true",
}


def download_file(url: str, dest: Path, force: bool = False):
    if dest.exists() and not force:
        if dest.stat().st_size > 1024 * 1024:
            print(f"[skip] {dest.name} уже существует")
            return
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


def _step(msg: str):
    def cb(p, text=""):
        print(f"  [{int(p * 100):3d}%] {text or msg}")
    return cb


def get_inpaint(force: bool = False):
    print("\n--- Модели для Инпейнта (Выделение) ---")
    for filename, url in INPAINT.items():
        download_file(url, MODELS_DIR / filename, force)
    print("[ok] модели инпейнта готовы")


def get_editor():
    print("\n--- Редактор: FLUX.2 klein 4B (GGUF Q4_K_M) ---")
    from app.editor import KleinEditor
    KleinEditor.ensure_files("Q4_K_M", _step("Редактор"))
    print("[ok] FLUX.2 klein готов")


def get_ip2p():
    print("\n--- Редактор (лёгкий): InstructPix2Pix ---")
    from huggingface_hub import snapshot_download
    from app.editor import IP2P_REPO
    snapshot_download(IP2P_REPO, allow_patterns=["*.json", "*.txt", "*.safetensors"],
                      ignore_patterns=["*.ckpt", "*.bin"])
    print("[ok] InstructPix2Pix готов")


def get_animate():
    print("\n--- Преображение: LTX-Video 0.9.5 ---")
    from app.animator import Animator
    Animator.ensure_files(_step("Преображение"))
    print("[ok] LTX-Video готов")


def get_translate():
    print("\n--- Переводчик RU→EN ---")
    from huggingface_hub import snapshot_download
    from app.translate import REPO, LOCAL
    snapshot_download(REPO, local_dir=str(LOCAL),
                      allow_patterns=["*.json", "*.txt", "*.model", "*.spm", "pytorch_model.bin", "*.safetensors"])
    print("[ok] переводчик готов")


def main():
    args = set(sys.argv[1:])
    force = "--force" in args
    everything = "--all" in args
    print("=== Local Face Swap — загрузка моделей ===\n")

    print("--- Основные модели (Face Swap) ---")
    buffalo_zip = MODELS_DIR / "buffalo_l.zip"
    download_file(MODELS["buffalo_l.zip"], buffalo_zip, force)
    extract_buffalo(buffalo_zip)
    download_file(MODELS["inswapper_128.onnx"], MODELS_DIR / "inswapper_128.onnx", force)

    if everything or "--inpaint" in args:
        get_inpaint(force)
    if everything or "--editor" in args:
        get_editor()
    if "--ip2p" in args or everything:
        get_ip2p()
    if everything or "--animate" in args:
        get_animate()
    if everything or "--translate" in args:
        get_translate()

    print("\n=== Готово! ===")
    print("Запуск: python app/main.py")


if __name__ == "__main__":
    main()