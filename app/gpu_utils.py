# Диагностика CUDA для onnxruntime.


from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Any


def _log(msg: str):
    print(msg, flush=True)


def _find_nvidia_dll_dirs() -> List[Path]:
    dirs = []
    try:
        import site
        candidates = list(site.getsitepackages()) + [site.getusersitepackages()]
        if hasattr(sys, "prefix"):
            candidates.append(str(Path(sys.prefix) / "Lib" / "site-packages"))
        seen = set()
        for base in candidates:
            nvidia_root = Path(base) / "nvidia"
            if not nvidia_root.is_dir():
                continue
            for sub in nvidia_root.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir() and str(bin_dir) not in seen:
                    seen.add(str(bin_dir))
                    dirs.append(bin_dir)
    except Exception as e:
        _log(f"[GPU] Поиск nvidia-пакетов: {e}")
    return dirs


def _preload_torch(verbose: bool = True) -> None:
    """Импортирует torch ДО добавления nvidia/*/bin в PATH/DLL-директории.

    torch/lib содержит собственный cuDNN 9. Если сначала добавить nvidia/*/bin,
    Windows загрузит cudnn64_9.dll из pip-пакета nvidia-cudnn-cu12, а затем
    torch/lib/cudnn_cnn64_9.dll будет искать символы в уже загруженной
    несовместимой версии → WinError 127 («процедура не найдена»).
    Предзагрузка torch кеширует его DLL первыми и предотвращает конфликт.
    """
    if "torch" in sys.modules:
        if verbose:
            try:
                import torch
                _log(f"[GPU] torch уже загружен: {torch.__version__}, cuda={torch.cuda.is_available()}")
            except Exception:
                pass
        return
    try:
        import torch
        if verbose:
            _log(f"[GPU] torch предзагружен: {torch.__version__}, cuda.is_available={torch.cuda.is_available()}")
    except Exception as e:
        if verbose:
            _log(f"[GPU] torch предзагрузка не удалась: {type(e).__name__}: {e}")
            _log("[GPU] Возможно, torch/lib/cudnn_* несовместим с системой. "
                 "Переустановите: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124")


def add_nvidia_dll_directories():
    if sys.platform != "win32":
        return
    dirs = _find_nvidia_dll_dirs()
    if not dirs:
        _log("[GPU] nvidia-*/bin не найдены. Для ORT CUDA: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12")
        return
    # PATH помогает найти транзитивные зависимости (например, cublasLt64_12 -> cublas64_12),
    # add_dll_directory — для прямых зависимостей самого onnxruntime.
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
            _log(f"[GPU] + dll_directory: {d}")
        except Exception as e:
            _log(f"[GPU] dll_directory fail: {d} ({e})")
        if str(d) not in path_parts:
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def _test_cuda_session() -> Tuple[bool, str]:
    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f"onnxruntime не импортируется: {e}"

    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = models_dir / "inswapper_128.onnx"
    providers_try = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        if not model_path.exists():
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" not in avail:
                return False, "CUDAExecutionProvider нет в списке провайдеров"
            return False, "Модель inswapper ещё не скачана — полный тест после download_models.py"

        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=providers_try,
        )
        active = sess.get_providers()
        if "CUDAExecutionProvider" in active:
            return True, ""
        return False, f"Сессия на CPU. Активные: {active}. Часто не хватает cuDNN 9: pip install nvidia-cudnn-cu12"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "126" in str(e) or "LoadLibrary" in str(e) or "cudnn" in str(e).lower():
            msg += (
                "\n  onnxruntime CUDA требует cuDNN 9 + CUDA 12 runtime.\n"
                "  Выполните:\n"
                "    pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12\n"
                "  Не удаляйте nvidia-cudnn-cu12 — без него face-swap уходит на CPU."
            )
        return False, msg


def setup_device(verbose: bool = True) -> Tuple[str, List[Any]]:
    if verbose:
        _log("=" * 60)
        _log("[GPU] Диагностика onnxruntime (CUDA → CPU)")
        _log("=" * 60)

    # КРИТИЧНО: сначала предзагружаем torch (его cuDNN кешируется первым),
    # затем добавляем nvidia/*/bin для onnxruntime. Иначе — WinError 127.
    _preload_torch(verbose=verbose)

    add_nvidia_dll_directories()

    if verbose:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                _log(f"[GPU] nvidia-smi: {r.stdout.strip()}")
            else:
                _log("[GPU] nvidia-smi: нет данных")
        except Exception as e:
            _log(f"[GPU] nvidia-smi: {e}")

    try:
        import onnxruntime as ort
        _log(f"[GPU] onnxruntime {ort.__version__}")
        _log(f"[GPU] available: {ort.get_available_providers()}")
    except Exception as e:
        _log(f"[GPU] ERROR: {e}")
        return "CPU", ["CPUExecutionProvider"]

    ok, err = _test_cuda_session()
    if ok:
        _log("[GPU] CUDA: OK — face-swap на видеокарте")
        try:
            import torch
            _log(f"[GPU] torch {torch.__version__}, cuda.is_available={torch.cuda.is_available()}")
        except Exception as e:
            _log(f"[GPU] torch: не загрузился ({e})")
        _log("=" * 60)
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
        return "CUDA", providers

    _log("[GPU] CUDA: НЕДОСТУПНА → face-swap на CPU")
    if err:
        _log("[GPU] ---- ОТЛАДКА ----")
        for line in err.splitlines():
            _log(f"[GPU] {line}")
        _log("[GPU] -----------------")
    try:
        import torch
        _log(f"[GPU] torch {torch.__version__}, cuda.is_available={torch.cuda.is_available()}")
    except Exception as e:
        _log(f"[GPU] torch: {e}")
    _log("=" * 60)
    return "CPU", ["CPUExecutionProvider"]


_DEVICE = None
_PROVIDERS = None


def get_providers(verbose: bool = True):
    global _DEVICE, _PROVIDERS
    if _PROVIDERS is None:
        _DEVICE, _PROVIDERS = setup_device(verbose=verbose)
    return _PROVIDERS


def get_device_name() -> str:
    global _DEVICE, _PROVIDERS
    if _PROVIDERS is None:
        _DEVICE, _PROVIDERS = setup_device(verbose=True)
    return _DEVICE or "CPU"