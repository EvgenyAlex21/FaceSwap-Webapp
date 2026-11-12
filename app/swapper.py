"""
Ядро face-swap на InsightFace + InSwapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from .utils import MODELS, read_image, write_image, get_video_info, create_video_writer, OUTPUTS, unique_name


class FaceSwapper:
    def __init__(self, providers: Optional[List[str]] = None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.providers = providers
        self.app: Optional[FaceAnalysis] = None
        self.swapper = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        print("[FaceSwapper] Загрузка моделей...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS.parent),
            providers=self.providers,
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        swapper_path = MODELS / "inswapper_128.onnx"
        if not swapper_path.exists():
            raise FileNotFoundError(
                f"Модель {swapper_path} не найдена.\n"
                "Запусти: python download_models.py"
            )

        self.swapper = get_model(str(swapper_path), providers=self.providers)
        self._initialized = True
        print("[FaceSwapper] Модели загружены.")

    def get_faces(self, img: np.ndarray):
        """Возвращает список лиц (крупные первые)."""
        self.initialize()
        faces = self.app.get(img)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces

    def detect_faces_preview(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Рисует номера лиц на изображении и возвращает
        (изображение с номерами, количество лиц).
        """
        self.initialize()
        faces = self.get_faces(img)
        preview = img.copy()

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            # Рамка
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 128), 2)
            # Номер
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(preview, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 255, 128), -1)
            cv2.putText(
                preview, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA
            )

        return preview, len(faces)

    def swap_image(
        self,
        source_path: str | Path,
        target_path: str | Path,
        source_face_index: int = 0,
        target_face_indices: Optional[List[int]] = None,
    ) -> Path:
        """
        Заменяет лица на фото.
        target_face_indices — список индексов лиц, которые нужно заменить.
        Если None — заменяет только лицо с индексом 0.
        """
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

        if target_face_indices is None:
            target_face_indices = [0]

        result = target_img.copy()
        for idx in target_face_indices:
            if idx < 0 or idx >= len(target_faces):
                continue
            target_face = target_faces[idx]
            result = self.swapper.get(result, target_face, source_face, paste_back=True)

        out_path = OUTPUTS / unique_name("swap_img", ".jpg")
        write_image(out_path, result)
        return out_path

    def swap_video(
        self,
        source_path: str | Path,
        target_video_path: str | Path,
        source_face_index: int = 0,
        target_face_indices: Optional[List[int]] = None,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> Path:
        """
        Заменяет лица в видео покадрово.
        target_face_indices — какие лица менять на каждом кадре.
        """
        self.initialize()

        source_img = read_image(source_path)
        source_faces = self.get_faces(source_img)
        if not source_faces:
            raise ValueError("На исходном фото лицо не найдено")

        if source_face_index >= len(source_faces):
            source_face_index = 0
        source_face = source_faces[source_face_index]

        if target_face_indices is None:
            target_face_indices = [0]

        width, height, fps, total_frames = get_video_info(target_video_path)
        if max_frames:
            total_frames = min(total_frames, max_frames)

        out_path = OUTPUTS / unique_name("swap_vid", ".mp4")
        writer = create_video_writer(out_path, width, height, fps)

        cap = cv2.VideoCapture(str(target_video_path))
        frame_idx = 0

        try:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = self.get_faces(frame)
                if faces:
                    for idx in target_face_indices:
                        if idx < 0 or idx >= len(faces):
                            continue
                        target_face = faces[idx]
                        frame = self.swapper.get(frame, target_face, source_face, paste_back=True)

                writer.write(frame)
                frame_idx += 1

                if progress_callback and frame_idx % 3 == 0:
                    progress_callback(min(frame_idx / total_frames, 1.0))
        finally:
            cap.release()
            writer.release()

        return out_path


_swapper: Optional[FaceSwapper] = None


def get_swapper() -> FaceSwapper:
    global _swapper
    if _swapper is None:
        _swapper = FaceSwapper()
    return _swapper
