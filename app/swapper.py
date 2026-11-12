"""Ядро face-swap. GPU/CPU, звук в видео, качество."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from .utils import (
    MODELS, read_image, write_image, get_video_info,
    create_video_writer, OUTPUTS, unique_name,
    upscale_image, enhance_face_region,
    make_browser_compatible, merge_audio,
)
from .gpu_utils import get_providers, get_device_name


class FaceSwapper:
    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        self.providers = get_providers(verbose=True)
        self.det_size = det_size
        self.app = None
        self.swapper = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        device = get_device_name()
        print(f"[FaceSwapper] Загрузка моделей на {device}...")
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
        print(f"[FaceSwapper] Модели загружены ({device}).")

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

    def detect_faces_preview(self, img: np.ndarray):
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

    def _load_source_face(self, source_path, face_index: int = 0):
        img = read_image(source_path)
        faces = self.get_faces(img)
        if not faces:
            raise ValueError(f"На фото {source_path} лицо не найдено")
        idx = min(int(face_index), len(faces) - 1)
        return faces[idx]

    def swap_image(
        self,
        source_map: Dict[int, Tuple],
        target_path,
        swap_all_with_first: bool = False,
        enhance: bool = False,
        enhance_strength: float = 0.65,
        upscale: int = 1,
        out_format: str = "jpg",
        progress_callback=None,
    ) -> Path:
        self.initialize()
        if progress_callback:
            progress_callback(0.1, "Поиск лиц...")

        target_img = read_image(target_path)
        target_faces = self.get_faces(target_img)
        if not target_faces:
            raise ValueError("На целевом фото лицо не найдено")

        source_cache = {}
        for t_idx, (src_path, s_idx) in source_map.items():
            key = (str(src_path), s_idx)
            if key not in source_cache:
                source_cache[key] = self._load_source_face(src_path, s_idx)

        if progress_callback:
            progress_callback(0.4, "Замена лиц...")

        result = target_img.copy()
        if swap_all_with_first and source_map:
            first_key = next(iter(source_map.values()))
            src_face = source_cache[(str(first_key[0]), first_key[1])]
            for tf in target_faces:
                result = self.swapper.get(result, tf, src_face, paste_back=True)
        else:
            for t_idx, (src_path, s_idx) in source_map.items():
                if 0 <= t_idx < len(target_faces):
                    src_face = source_cache[(str(src_path), s_idx)]
                    result = self.swapper.get(result, target_faces[t_idx], src_face, paste_back=True)

        if progress_callback:
            progress_callback(0.75, "Улучшение / апскейл...")

        if enhance and enhance_strength > 0:
            result = enhance_face_region(result, strength=enhance_strength)
        if upscale > 1:
            result = upscale_image(result, scale=upscale)

        out_path = OUTPUTS / unique_name("swap_img", f".{out_format}")
        write_image(out_path, result, fmt=out_format)
        if progress_callback:
            progress_callback(1.0, "Готово")
        return out_path

    def swap_video(
        self,
        source_map: Dict[int, Tuple],
        target_video_path,
        swap_all_with_first: bool = False,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        detect_every_n: int = 1,
        enhance: bool = False,
        enhance_strength: float = 0.45,
        upscale: int = 1,
        video_crf: int = 20,
        keep_audio: bool = True,
        progress_callback=None,
    ) -> Path:
        self.initialize()

        source_cache = {}
        for t_idx, (src_path, s_idx) in source_map.items():
            key = (str(src_path), s_idx)
            if key not in source_cache:
                source_cache[key] = self._load_source_face(src_path, s_idx)
        if not source_cache:
            raise ValueError("Не загружено ни одного исходного лица")

        width, height, fps, total_frames, duration = get_video_info(target_video_path)
        start_frame = int(max(0, start_sec) * fps)
        if end_sec and end_sec > start_sec:
            end_frame = min(int(end_sec * fps), total_frames)
        else:
            end_frame = total_frames
        frames_to_process = max(0, end_frame - start_frame)
        if frames_to_process == 0:
            raise ValueError("Пустой отрезок видео")

        print(f"[FaceSwapper] Видео: {frames_to_process} кадров @ {fps:.1f}fps, device={get_device_name()}")

        out_w = (width * upscale) // 2 * 2
        out_h = (height * upscale) // 2 * 2
        raw_path = OUTPUTS / unique_name("swap_vid", ".mp4")
        writer, raw_path, out_w, out_h = create_video_writer(raw_path, out_w, out_h, fps)

        cap = cv2.VideoCapture(str(target_video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = 0
        last_faces = []
        default_src = next(iter(source_cache.values()))

        try:
            while frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % max(detect_every_n, 1) == 0 or not last_faces:
                    faces = self.get_faces(frame)
                    last_faces = faces
                else:
                    faces = last_faces

                if faces:
                    if swap_all_with_first:
                        for tf in faces:
                            frame = self.swapper.get(frame, tf, default_src, paste_back=True)
                    else:
                        for t_idx, (src_path, s_idx) in source_map.items():
                            if 0 <= t_idx < len(faces):
                                src_face = source_cache[(str(src_path), s_idx)]
                                frame = self.swapper.get(frame, faces[t_idx], src_face, paste_back=True)

                if enhance and enhance_strength > 0:
                    frame = enhance_face_region(frame, strength=enhance_strength)
                if upscale > 1:
                    frame = upscale_image(frame, scale=upscale)
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                writer.write(frame)
                frame_idx += 1

                if progress_callback and (frame_idx % 3 == 0 or frame_idx == frames_to_process):
                    progress_callback(
                        min(frame_idx / frames_to_process * 0.85, 0.85),
                        f"Кадр {frame_idx}/{frames_to_process}",
                    )
        finally:
            cap.release()
            writer.release()

        if progress_callback:
            progress_callback(0.88, "Кодирование H.264...")

        # H.264
        video = make_browser_compatible(raw_path, crf=int(video_crf), preset="fast")

        # Звук
        if keep_audio:
            if progress_callback:
                progress_callback(0.94, "Добавление звука...")
            video = merge_audio(video, target_video_path, start_sec=start_sec, end_sec=end_sec)

        if progress_callback:
            progress_callback(1.0, "Готово")
        print(f"[FaceSwapper] Итог: {video}")
        return Path(video)


_swapper = None


def get_swapper(det_size=(640, 640)):
    global _swapper
    if _swapper is None:
        _swapper = FaceSwapper(det_size=det_size)
    return _swapper
