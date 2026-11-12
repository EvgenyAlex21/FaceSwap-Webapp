"""
Диагностика и выбор устройства: NVIDIA CUDA → CPU.
Подробный лог в консоль, если GPU недоступен.
"""

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


def add_nvidia_paths_to_dll_search():
    """Windows: добавляем DLL-пути pip-пакетов nvidia-*."""
    if sys.platform != "win32":
        return
    dirs = _find_nvidia_dll_dirs()
    if not dirs:
        _log("[GPU] nvidia-*/bin в site-packages не найдены")
        _log("[GPU] Подсказка: pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12")
        return
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
            _log(f"[GPU] + DLL dir: {d}")
        except Exception as e:
            _log(f"[GPU] add_dll_directory fail: {d} ({e})")
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def _test_cuda_session() -> Tuple[bool, str]:
    """
    Реальная проверка CUDA EP.
    Возвращает (ok, error_message).
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f"onnxruntime не импортируется: {e}"

    # Пробуем на модели inswapper, если есть; иначе — только список провайдеров
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = models_dir / "inswapper_128.onnx"

    providers_try = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        if model_path.exists():
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
            return False, f"Сессия создана, но CUDA не активен. Активные: {active}"
        else:
            # Модели ещё нет — проверяем, что провайдер хотя бы регистрируется
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" not in avail:
                return False, "CUDAExecutionProvider отсутствует в get_available_providers()"
            # Пробуем создать session options + provider options (лёгкий тест)
            try:
                # Некоторые сборки падают уже при GetProvider
                from onnxruntime.capi import _pybind_state as C
                # просто повторный available
            except Exception:
                pass
            return False, (
                "Модель models/inswapper_128.onnx ещё не скачана — "
                "полный тест CUDA будет после python download_models.py. "
                "Провайдер CUDA есть в списке, но LoadLibrary может упасть при загрузке модели."
            )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = str(e).lower()
        if "126" in str(e) or "loadlibrary" in low or "dll" in low:
            msg += (
                "\n  Причина: Windows не нашёл зависимость CUDA (cublas/cudart/cudnn .dll).\n"
                "  Решение 1: pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12\n"
                "  Решение 2: установите CUDA Toolkit 12.x с сайта NVIDIA и добавьте bin в PATH\n"
                "  Решение 3: перезапустите терминал после установки пакетов"
            )
        return False, msg


def setup_device(verbose: bool = True) -> Tuple[str, List[Any]]:
    if verbose:
        _log("=" * 60)
        _log("[GPU] Диагностика (CUDA → CPU fallback)")
        _log("=" * 60)

    add_nvidia_paths_to_dll_search()

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
                _log("[GPU] nvidia-smi: нет данных (драйвер не установлен?)")
        except FileNotFoundError:
            _log("[GPU] nvidia-smi не найден в PATH")
        except Exception as e:
            _log(f"[GPU] nvidia-smi error: {e}")

    try:
        import onnxruntime as ort
        _log(f"[GPU] onnxruntime {ort.__version__}")
        _log(f"[GPU] available: {ort.get_available_providers()}")
    except Exception as e:
        _log(f"[GPU] ERROR: {e}")
        return "CPU", ["CPUExecutionProvider"]

    ok, err = _test_cuda_session()
    if ok:
        _log("[GPU] CUDA: OK — используем видеокарту")
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

    _log("[GPU] CUDA: НЕДОСТУПНА → fallback на CPU")
    if err:
        _log("[GPU] ---- ОТЛАДКА ----")
        for line in err.splitlines():
            _log(f"[GPU] {line}")
        _log("[GPU] -----------------")
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
