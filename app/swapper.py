"""
Ядро face-swap. Оптимизировано, без шума TensorRT.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from .utils import (
    MODELS, read_image, write_image, get_video_info,
    create_video_writer, OUTPUTS, unique_name,
    upscale_image, enhance_face_region,
)


def _get_providers():
    """Только CUDA + CPU. TensorRT отключён — меньше ошибок в консоли."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "DEFAULT",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        providers.append("CPUExecutionProvider")
        return providers
    except Exception:
        return ["CPUExecutionProvider"]


class FaceSwapper:
    def __init__(self, det_size: Tuple[int, int] = (480, 480)):
        self.providers = _get_providers()
        self.det_size = det_size
        self.app = None
        self.swapper = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        print("[FaceSwapper] Загрузка моделей...")
        prov_names = [p[0] if isinstance(p, tuple) else p for p in self.providers]
        print(f"[FaceSwapper] Providers: {prov_names}")

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS.parent),
            providers=self.providers,
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size)

        swapper_path = MODELS / "inswapper_128.onnx"
        if not swapper_path.exists():
            raise FileNotFoundError(
                f"Модель {swapper_path} не найдена.\nЗапусти: python download_models.py"
            )

        self.swapper = get_model(str(swapper_path), providers=self.providers)
        self._initialized = True
        print("[FaceSwapper] Модели загружены.")

    def set_det_size(self, size: Tuple[int, int]):
        if size != self.det_size:
            self.det_size = size
            if self.app is not None:
                self.app.prepare(ctx_id=0, det_size=self.det_size)

    def get_faces(self, img: np.ndarray):
        self.initialize()
        faces = self.app.get(img)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces

    def detect_faces_preview(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        self.initialize()
        faces = self.get_faces(img)
        preview = img.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 128), 2)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(preview, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 255, 128), -1)
            cv2.putText(preview, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        return preview, len(faces)

    def swap_image(
        self,
        source_path: str | Path,
        target_path: str | Path,
        source_face_index: int = 0,
        target_face_indices: Optional[List[int]] = None,
        swap_all: bool = True,
        enhance: bool = False,
        upscale: int = 1,
        out_format: str = "jpg",
    ) -> Path:
        self.initialize()

        source_img = read_image(source_path)
        target_img = read_image(target_path)

        source_faces = self.get_faces(source_img)
        target_faces = self.get_faces(target_img)

        if not source_faces:
            raise ValueError("На исходном фото лицо не найдено")
        if not target_faces:
            raise ValueError("На целевом фото лицо не найдено")

        if source_face_index >= len(source_faces):
            source_face_index = 0

        source_face = source_faces[source_face_index]
        indices = list(range(len(target_faces))) if swap_all or not target_face_indices else target_face_indices

        result = target_img.copy()
        for idx in indices:
            if 0 <= idx < len(target_faces):
                result = self.swapper.get(result, target_faces[idx], source_face, paste_back=True)

        if enhance:
            result = enhance_face_region(result, strength=0.65)

        if upscale > 1:
            result = upscale_image(result, scale=upscale)

        out_path = OUTPUTS / unique_name("swap_img", f".{out_format}")
        write_image(out_path, result, fmt=out_format)
        return out_path

    def swap_video(
        self,
        source_path: str | Path,
        target_video_path: str | Path,
        source_face_index: int = 0,
        target_face_indices: Optional[List[int]] = None,
        swap_all: bool = True,
        max_frames: Optional[int] = None,
        detect_every_n: int = 2,
        enhance: bool = False,
        upscale: int = 1,
        progress_callback=None,
    ) -> Path:
        self.initialize()

        source_img = read_image(source_path)
        source_faces = self.get_faces(source_img)
        if not source_faces:
            raise ValueError("На исходном фото лицо не найдено")

        if source_face_index >= len(source_faces):
            source_face_index = 0
        source_face = source_faces[source_face_index]

        width, height, fps, total_frames = get_video_info(target_video_path)
        if max_frames:
            total_frames = min(total_frames, max_frames)

        out_w, out_h = width * upscale, height * upscale
        out_path = OUTPUTS / unique_name("swap_vid", ".mp4")
        writer = create_video_writer(out_path, out_w, out_h, fps)

        cap = cv2.VideoCapture(str(target_video_path))
        frame_idx = 0
        last_faces = []

        try:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % detect_every_n == 0 or not last_faces:
                    faces = self.get_faces(frame)
                    last_faces = faces
                else:
                    faces = last_faces

                if faces:
                    indices = list(range(len(faces))) if swap_all or not target_face_indices else target_face_indices
                    for idx in indices:
                        if 0 <= idx < len(faces):
                            frame = self.swapper.get(frame, faces[idx], source_face, paste_back=True)

                if enhance:
                    frame = enhance_face_region(frame, strength=0.5)

                if upscale > 1:
                    frame = upscale_image(frame, scale=upscale)

                writer.write(frame)
                frame_idx += 1

                if progress_callback and frame_idx % 5 == 0:
                    progress_callback(min(frame_idx / total_frames, 1.0))
        finally:
            cap.release()
            writer.release()

        return out_path


_swapper = None


def get_swapper(det_size=(480, 480)):
    global _swapper
    if _swapper is None:
        _swapper = FaceSwapper(det_size=det_size)
    return _swapper
