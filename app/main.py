#!/usr/bin/env python3
"""Local Face Swap """

from __future__ import annotations

import sys
import os
from pathlib import Path
import math

os.environ["HF_HOME"] = str(Path(__file__).resolve().parent.parent / ".cache" / "huggingface")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import cv2
import numpy as np
from PIL import Image

from app.swapper import get_swapper
from app.utils import OUTPUTS, unique_name, write_image, get_video_info, copy_to_uploads
from app.editor import EDITOR_MODELS, KLEIN_QUANTS, edit_image
from app.animator import (
    RESOLUTIONS as ANIM_RESOLUTIONS, DURATIONS as ANIM_DURATIONS,
    UPSCALES as ANIM_UPSCALES,
    DEFAULT_NEGATIVE as ANIM_NEGATIVE, animate_image,
)
from app.inpaint import inpaint_ai

MAX_FACES = 6

DET_SIZES = {
    "Быстрый (320)": (320, 320),
    "Баланс (480)": (480, 480),
    "Качественный (640)": (640, 640),
    "Максимум (800)": (800, 800),
}

PHOTO_QUALITY = {
    "Как есть (без изменения размера)": 0,
    "HD (1280)": 1280,
    "Full HD (1920)": 1920,
    "QHD / 2K (2560)": 2560,
    "4K UHD (3840)": 3840,
}

VIDEO_RESOLUTIONS = {
    "Как есть (оригинал)": 0,
    "SD (480p)": 480,
    "HD Ready (720p)": 720,
    "Full HD (1080p)": 1080,
    "2K / QHD (1440p)": 1440,
    "4K Ultra HD (2160p)": 2160,
    "8K Ultra HD (4320p)": 4320,
}

VIDEO_FPS = {
    "Как есть (оригинал)": 0,
    "24 fps (кино)": 24,
    "30 fps (стандарт)": 30,
    "60 fps (плавно)": 60,
}

ENHANCE_STRENGTH = {
    "Выкл": 0.0,
    "Слабое": 0.35,
    "Среднее": 0.55,
    "Сильное": 0.75,
}

VIDEO_QUALITY = {
    "Максимум (CRF 16)": 16,
    "Высокое (CRF 18)": 18,
    "Хорошее (CRF 20)": 20,
    "Среднее (CRF 23)": 23,
    "Быстрое (CRF 26)": 26,
}

INPAINT_MODELS = [
    "Realistic Vision (SD 1.5 inpaint)",
    "LUSTIFY (SDXL inpaint)",
    "TryOffAnyone (одежда)",
]


def _to_bgr(img):
    if img is None:
        return None
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_pil(bgr):
    if bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _save_tmp(img, prefix="tmp"):
    path = OUTPUTS / unique_name(prefix, ".jpg")
    write_image(path, _to_bgr(img))
    return path


def _apply_det(det_mode: str):
    size = DET_SIZES.get(det_mode, (640, 640))
    sw = get_swapper()
    sw.set_det_size(size)
    return sw


def _resize_long_side(img_bgr: np.ndarray, long_side: int) -> np.ndarray:
    if not long_side or long_side <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    cur = max(h, w)
    if cur == long_side:
        return img_bgr
    scale = long_side / cur
    nw, nh = int(w * scale), int(h * scale)
    nw, nh = max(2, nw // 2 * 2), max(2, nh // 2 * 2)
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img_bgr, (nw, nh), interpolation=interp)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _crop_face(img_bgr, face, pad: float = 0.35):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, face.bbox)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    return img_bgr[y1:y2, x1:x2].copy()


# ---------- PHOTO ----------

def detect_target_faces(target_img, det_mode):
    empty = [None] * MAX_FACES
    hidden = [gr.update(visible=False)] * MAX_FACES
    clear_prev = [None] * MAX_FACES
    clear_stat = [""] * MAX_FACES

    if target_img is None:
        return (
            "Загрузите целевое фото",
            *empty, *hidden, 0, *clear_prev, *clear_stat,
        )

    sw = _apply_det(det_mode)
    bgr = _to_bgr(target_img)
    faces = sw.get_faces(bgr)

    if not faces:
        return (
            "Лица не найдены на целевом фото",
            *empty, *hidden, 0, *clear_prev, *clear_stat,
        )

    crops = []
    vis = []
    for i in range(MAX_FACES):
        if i < len(faces):
            crop = _crop_face(bgr, faces[i])
            crops.append(_to_pil(crop))
            vis.append(gr.update(visible=True))
        else:
            crops.append(None)
            vis.append(gr.update(visible=False))

    return (
        f"Найдено лиц: **{len(faces)}** — загрузите замену для каждого",
        *crops,
        *vis,
        len(faces),
        *clear_prev,
        *clear_stat,
    )


def clear_target_and_slots(target_img):
    if target_img is not None:
        return (
            "Нажмите «Найти лица», чтобы обновить слоты",
            *[None] * MAX_FACES,
            *[gr.update(visible=False)] * MAX_FACES,
            0,
            *[None] * MAX_FACES,
            *[""] * MAX_FACES,
        )
    return (
        "",
        *[None] * MAX_FACES,
        *[gr.update(visible=False)] * MAX_FACES,
        0,
        *[None] * MAX_FACES,
        *[""] * MAX_FACES,
    )


def detect_source_face(src_img, det_mode):
    if src_img is None:
        return None, "Нет фото"
    sw = _apply_det(det_mode)
    bgr = _to_bgr(src_img)
    faces = sw.get_faces(bgr)
    if not faces:
        return None, "Лицо не найдено"
    crop = _crop_face(bgr, faces[0])
    return _to_pil(crop), "Лицо найдено"


def source_changed(src_img):
    return None, ""


def swap_photo_slots(
    target_img,
    face_count,
    det_mode,
    photo_quality,
    enhance_mode,
    out_format,
    *src_images,
    progress=gr.Progress(track_tqdm=False),
):
    if target_img is None:
        raise gr.Error("Загрузите целевое фото")

    n = int(face_count or 0)
    if n <= 0:
        raise gr.Error("Сначала нажмите «Найти лица» на целевом фото")

    loaded = [
        i for i in range(min(n, MAX_FACES))
        if i < len(src_images) and src_images[i] is not None
    ]
    if not loaded:
        raise gr.Error("Загрузите хотя бы одно фото для замены лица")

    progress(0.05, desc="Подготовка...")
    sw = _apply_det(det_mode)

    source_map = {}
    for i in loaded:
        src = src_images[i]
        path = _save_tmp(src, f"src{i}")
        faces = sw.get_faces(_to_bgr(src))
        if not faces:
            raise gr.Error(f"На фото замены для лица #{i + 1} лицо не найдено")
        source_map[i] = (path, 0)

    tgt_path = _save_tmp(target_img, "tgt")
    strength = ENHANCE_STRENGTH.get(enhance_mode, 0.55)
    long_side = PHOTO_QUALITY.get(photo_quality, 0)

    def cb(p, msg=""):
        progress(p, desc=msg or "Обработка...")

    result_path = sw.swap_image(
        source_map=source_map,
        target_path=tgt_path,
        swap_all_with_first=False,
        enhance=strength > 0,
        enhance_strength=strength,
        upscale=1,
        out_format=out_format.lower(),
        progress_callback=cb,
    )

    img = cv2.imread(str(result_path))
    if long_side:
        progress(0.92, desc="Масштабирование...")
        img = _resize_long_side(img, long_side)
        write_image(result_path, img, fmt=out_format.lower())

    progress(1.0, desc="Готово")
    return _to_pil(img), str(result_path)


# ---------- VIDEO ----------

def on_video_upload(video):
    if video is None:
        return (
            gr.update(maximum=60, value=0),
            gr.update(maximum=60, value=0),
            "Загрузите видео",
            "Отрезок: **—**",
            gr.update(choices=[], value=None),
        )
    try:
        try:
            video = str(copy_to_uploads(video, prefix="meta"))
        except Exception:
            pass
        _, _, fps, total, duration = get_video_info(video)
        dur = max(1.0, float(duration))
        return (
            gr.update(maximum=dur, value=0),
            gr.update(maximum=dur, value=min(dur, 15.0)),
            f"Длительность: **{_fmt_hms(dur)}** ({dur:.1f} сек) · {total} кадров · {fps:.1f} fps",
            f"Отрезок: **{_fmt_hms(0)} — {_fmt_hms(min(dur, 15.0))}**",
            gr.update(choices=[], value=None),
        )
    except Exception as e:
        return gr.update(), gr.update(), f"Ошибка: {e}", "Отрезок: **—**", gr.update()


def _update_range_text(start, end):
    try:
        s = float(start or 0)
        e = float(end or 0)
    except Exception:
        return "Отрезок: **—**"
    if e <= s:
        return f"Отрезок: **{_fmt_hms(s)} — {_fmt_hms(s)}** (конец должен быть больше начала)"
    return f"Отрезок: **{_fmt_hms(s)} — {_fmt_hms(e)}** (длительность {_fmt_hms(e - s)})"


def reset_vid_faces():
    return (
        "Отрезок изменён — нажмите «Найти лица в видео»",
        *[None] * MAX_FACES,
        *[gr.update(visible=False)] * MAX_FACES,
        0,
        gr.update(choices=[], value=None),
    )


def detect_video_faces(video_path, det_mode, start_sec, end_sec, progress=gr.Progress(track_tqdm=False)):
    if video_path is None:
        return "Загрузите видео", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    progress(0.05, desc="Сканирование видео...")
    try:
        video_path = str(copy_to_uploads(video_path, prefix="vid_scan"))
    except Exception:
        video_path = video_path

    sw = _apply_det(det_mode)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return "Ошибка чтения видео", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    start_frame = int(max(0, float(start_sec or 0)) * fps)
    end_frame = int(min(total_frames, float(end_sec or 0) * fps)) if end_sec and end_sec > start_sec else total_frames

    if end_frame <= start_frame:
        end_frame = total_frames

    frames_to_scan = end_frame - start_frame
    step = max(1, frames_to_scan // 100)

    unique_faces = []
    crops = []

    threshold = 0.5

    for f_idx in range(start_frame, end_frame, step):
        progress(min(0.05 + 0.8 * ((f_idx - start_frame)/frames_to_scan), 0.85), desc=f"Анализ кадра {f_idx}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: break

        faces = sw.get_faces(frame)
        for face in faces:
            is_new = True

            if face.embedding is not None:
                emb1 = face.embedding / np.linalg.norm(face.embedding)
                for uf in unique_faces:
                    if uf.embedding is not None:
                        emb2 = uf.embedding / np.linalg.norm(uf.embedding)
                        sim = np.dot(emb1, emb2)
                        if sim > threshold:
                            is_new = False
                            break
            else:
                cx, cy = (face.bbox[0]+face.bbox[2])/2, (face.bbox[1]+face.bbox[3])/2
                for uf in unique_faces:
                    ucx, ucy = (uf.bbox[0]+uf.bbox[2])/2, (uf.bbox[1]+uf.bbox[3])/2
                    if math.hypot(cx-ucx, cy-ucy) < frame.shape[1] * 0.15:
                        is_new = False
                        break

            if is_new:
                unique_faces.append(face)
                crops.append(_to_pil(_crop_face(frame, face)))
                if len(unique_faces) >= MAX_FACES:
                    break
        if len(unique_faces) >= MAX_FACES:
            break

    cap.release()

    if not unique_faces:
        return "Лица в видео не найдены", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    vis = [gr.update(visible=True)] * len(unique_faces) + [gr.update(visible=False)] * (MAX_FACES - len(unique_faces))
    crops_padded = crops + [None] * (MAX_FACES - len(crops))

    choices = [f"Лицо #{i+1}" for i in range(len(unique_faces))]

    return (
        f"Найдено уникальных лиц: **{len(unique_faces)}**",
        *crops_padded,
        *vis,
        len(unique_faces),
        gr.update(choices=choices, value=choices[0] if choices else None)
    )


def detect_src_faces(src_img, det_mode):
    if src_img is None:
        return "Загрузите фото", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    sw = _apply_det(det_mode)
    bgr = _to_bgr(src_img)
    faces = sw.get_faces(bgr)

    if not faces:
        return "Лица на фото не найдены", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    crops = []
    vis = []
    for i in range(MAX_FACES):
        if i < len(faces):
            crops.append(_to_pil(_crop_face(bgr, faces[i])))
            vis.append(gr.update(visible=True))
        else:
            crops.append(None)
            vis.append(gr.update(visible=False))

    choices = [f"Лицо #{i+1}" for i in range(len(faces))]

    return (
        f"Найдено лиц: **{len(faces)}**",
        *crops,
        *vis,
        len(faces),
        gr.update(choices=choices, value=choices[0] if choices else None)
    )


def swap_video_slots(
    src_img, target_video, start_sec, end_sec,
    det_mode, detect_every, enhance_mode, video_quality,
    video_res, video_fps, keep_audio,
    src_face_choice, vid_face_choice,
    src_video_count, tgt_video_count,
    progress=gr.Progress(track_tqdm=False),
):
    if target_video is None:
        raise gr.Error("Загрузите видео")
    if src_img is None:
        raise gr.Error("Загрузите фото лица-источника")
    if tgt_video_count == 0:
        raise gr.Error("Сначала нажмите «Найти лица в видео»")
    if src_video_count == 0:
        raise gr.Error("Сначала нажмите «Найти лица на фото»")

    if not src_face_choice or not vid_face_choice:
        raise gr.Error("Выберите лицо-источник и лицо в видео для замены")

    try:
        src_idx = int(src_face_choice.split("#")[1]) - 1
        vid_idx = int(vid_face_choice.split("#")[1]) - 1
    except Exception:
        raise gr.Error("Ошибка парсинга выбранных лиц")

    progress(0.02, desc="Подготовка...")
    try:
        video_path = str(copy_to_uploads(target_video, prefix="vid"))
    except Exception:
        video_path = target_video

    sw = _apply_det(det_mode)
    src_path = _save_tmp(src_img, "vsrc")

    source_map = {vid_idx: (src_path, src_idx)}

    strength = ENHANCE_STRENGTH.get(enhance_mode, 0.45)
    crf = VIDEO_QUALITY.get(video_quality, 20)

    tgt_res_val = VIDEO_RESOLUTIONS.get(video_res, 0)
    tgt_fps_val = VIDEO_FPS.get(video_fps, 0)

    def cb(p, msg=""):
        progress(p, desc=msg or "Обработка видео...")

    result_path = sw.swap_video(
        source_map=source_map,
        target_video_path=video_path,
        swap_all_with_first=False,
        start_sec=float(start_sec or 0),
        end_sec=float(end_sec or 0),
        detect_every_n=int(detect_every),
        enhance=strength > 0,
        enhance_strength=strength,
        upscale=1,
        video_crf=crf,
        keep_audio=bool(keep_audio),
        target_res=tgt_res_val,
        target_fps=tgt_fps_val,
        progress_callback=cb,
    )
    path = str(Path(result_path).resolve())
    return path, path


# ---------- INPAINT ----------

def run_inpaint(editor_data, method, radius,
                ai_prompt, ai_negative, ai_steps, ai_guidance, ai_strength, ai_seed,
                model_choice,
                progress=gr.Progress(track_tqdm=False)):
    if editor_data is None:
        raise gr.Error("Загрузите изображение и закрасьте область")

    progress(0.05, desc="Разбор маски...")

    def _arr_to_bgr(arr):
        arr = np.array(arr)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    bg = None
    mask = None

    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        composite = editor_data.get("composite")
        layers = editor_data.get("layers") or []
        if bg is None:
            bg = composite
        if bg is None:
            raise gr.Error("Нет изображения — сначала Upload")

        if layers:
            acc = None
            for layer in layers:
                arr = np.array(layer)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    a = arr[:, :, 3]
                elif arr.ndim == 3:
                    a = (np.any(arr[:, :, :3] > 8, axis=2)).astype(np.uint8) * 255
                else:
                    a = arr.astype(np.uint8)
                acc = a if acc is None else np.maximum(acc, a)
            mask = acc
        elif composite is not None and editor_data.get("background") is not None:
            c = np.array(composite)
            b = np.array(editor_data["background"])
            if c.shape[:2] != b.shape[:2]:
                c = cv2.resize(c, (b.shape[1], b.shape[0]))
            diff = np.abs(c.astype(np.int16)[..., :3] - b.astype(np.int16)[..., :3])
            mask = (np.any(diff > 15, axis=2)).astype(np.uint8) * 255
        image_bgr = _arr_to_bgr(bg)
    else:
        raise gr.Error("Используйте редактор: загрузите фото и закрасьте область кистью")

    if mask is None or int(np.asarray(mask).sum()) == 0:
        raise gr.Error("Маска пустая — выберите кисть и закрасьте нужную область")

    progress(0.15, desc="Маска создана. Освобождаю VRAM...")

    # --- Выгружаем ВСЕ модели, которые могут занимать VRAM ---
    try:
        import torch, gc

        # face-swap
        try:
            from app.swapper import _swapper
            if _swapper is not None and _swapper._initialized:
                _swapper.app = None
                _swapper.swapper = None
                _swapper._initialized = False
                print("[Inpaint] face-swap выгружен")
        except Exception as e:
            print(f"[Inpaint] face-swap: {e}")

        # редактор
        try:
            from app.editor import _klein, _ip2p
            if _klein is not None:
                _klein.unload()
                print("[Inpaint] FLUX.2 klein выгружен")
            if _ip2p is not None:
                _ip2p.unload()
                print("[Inpaint] InstructPix2Pix выгружен")
        except Exception as e:
            print(f"[Inpaint] editor: {e}")

        # аниматор
        try:
            from app.animator import _animator
            if _animator is not None:
                _animator.unload()
                print("[Inpaint] LTX-Video выгружен")
        except Exception as e:
            print(f"[Inpaint] animator: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        free, total = torch.cuda.mem_get_info()
        print(f"[Inpaint] VRAM после выгрузки: {free/1024**3:.2f} GB из {total/1024**3:.2f} GB")
    except Exception as e:
        print(f"[Inpaint] Очистка VRAM: {e}")

    progress(0.25, desc=f"Загрузка модели {model_choice}...")

    try:
        from app.inpaint import inpaint_ai, unload_inpaint

        path = inpaint_ai(
            image_bgr, np.asarray(mask),
            prompt=str(ai_prompt or "").strip() or "nude body, realistic skin, natural anatomy, photorealistic",
            negative=str(ai_negative or "").strip(),
            steps=int(ai_steps), guidance=float(ai_guidance),
            strength=float(ai_strength), seed=int(ai_seed),
            model_choice=model_choice,
        )

        progress(0.95, desc="Финальная сборка...")

        unload_inpaint()

        return _to_pil(cv2.imread(str(path))), str(path)
    except Exception as e:
        raise gr.Error(str(e))


# ---------- EDITOR / TRANSFORMATION ----------

EDITOR_EXAMPLES = [
    "Измени цвет волос на розовый",
    "Добавь солнцезащитные очки",
    "Сделай глаза немного больше и выразительнее",
    "Замени фон на закат на морском берегу",
    "Добавь снегопад и зимнюю атмосферу",
]

ANIMATE_EXAMPLES = [
    "Человек улыбается и слегка поворачивает голову, волосы чуть колышутся",
    "Собака виляет хвостом и моргает, камера медленно приближается",
    "Лёгкий ветер колышет листья и траву, облака медленно плывут",
    "Волны накатывают на берег, чайки пролетают в небе",
]


def _prog(progress):
    def cb(p, msg=""):
        progress(min(max(float(p), 0.0), 1.0), desc=msg or "Обработка...")
    return cb


def _friendly(e: Exception) -> str:
    from app.vram import is_oom
    text = str(e)
    low = text.lower()
    if is_oom(e):
        return ("Не хватило видеопамяти даже на минимальном разрешении. Закройте другие программы, использующие GPU, "
                "выберите меньший размер или более лёгкую модель.")
    if isinstance(e, FileNotFoundError):
        return text
    if any(k in low for k in ("connection", "timed out", "max retries", "name resolution", "offline", "401", "403", "gated")):
        return ("Не удалось скачать модель с Hugging Face. Проверьте интернет или заранее выполните "
                "python download_models.py --editor / --animate. Подробности: " + text[:300])
    return text[:600]


def editor_model_changed(label):
    key = EDITOR_MODELS.get(label, "klein")
    if key == "ip2p":
        return (
            gr.update(minimum=10, maximum=40, value=20, info="InstructPix2Pix: 15–25 шагов"),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    return (
        gr.update(minimum=2, maximum=8, value=4, info="FLUX.2 klein дистиллирована: 4 шага — оптимум"),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def run_editor(image, prompt, model_label, max_side, steps, image_guidance, seed, quant,
               translate, restore_size, progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Загрузите фото")
    if not (prompt or "").strip():
        raise gr.Error("Напишите, что нужно изменить на фото")
    try:
        try:
            import torch, gc
            from app.swapper import _swapper
            if _swapper is not None and _swapper._initialized:
                _swapper.app = None
                _swapper.swapper = None
                _swapper._initialized = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[Editor] Не удалось выгрузить face-swap: {e}")

        out, path, info = edit_image(
            image, prompt,
            model_key=EDITOR_MODELS.get(model_label, "klein"),
            max_side=int(max_side), steps=int(steps),
            seed=int(seed) if seed is not None else -1,
            quant=quant or "Q4_K_M",
            image_guidance=float(image_guidance),
            translate=bool(translate), restore_size=bool(restore_size),
            progress=_prog(progress),
        )
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(_friendly(e))
    progress(1.0, desc="Готово")
    return out, str(path), info


def run_animate(image, prompt, negative, resolution_key, duration_label, steps, guidance, seed,
                translate, add_quality, codec_prep, upscale,
                progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Загрузите фото")
    if not (prompt or "").strip():
        raise gr.Error("Опишите, что должно происходить в кадре")

    try:
        import torch, gc

        try:
            from app.swapper import _swapper
            if _swapper is not None and _swapper._initialized:
                _swapper.app = None
                _swapper.swapper = None
                _swapper._initialized = False
                print("[Animate] face-swap выгружен")
        except Exception as e:
            print(f"[Animate] face-swap: {e}")

        # редактор
        try:
            from app.editor import _klein, _ip2p
            if _klein is not None:
                _klein.unload()
                _klein._emb_cache.clear()   # очищаем кэш эмбеддингов
                print("[Animate] FLUX.2 klein выгружен")
            if _ip2p is not None:
                _ip2p.unload()
                print("[Animate] InstructPix2Pix выгружен")
        except Exception as e:
            print(f"[Animate] editor: {e}")

        # инпейнт
        try:
            from app.inpaint import unload_inpaint
            unload_inpaint()
            print("[Animate] inpaint выгружен")
        except Exception as e:
            print(f"[Animate] inpaint: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        free, total = torch.cuda.mem_get_info()
        print(f"[Animate] VRAM после выгрузки: {free/1024**3:.2f} GB из {total/1024**3:.2f} GB")
    except Exception as e:
        print(f"[Animate] Очистка VRAM: {e}")

    # --- Парсим длительность из строки вида "2 сек" ---
    try:
        duration_sec = float(duration_label.replace(" сек", "").strip())
    except Exception:
        duration_sec = 2.0  # безопасное значение по умолчанию

    try:
        path, info = animate_image(
            image, prompt, negative=negative or "",
            resolution_key=resolution_key, duration_sec=duration_sec,
            steps=int(steps), guidance=float(guidance),
            seed=int(seed) if seed is not None else -1,
            translate=bool(translate), add_quality=bool(add_quality),
            codec_prep=bool(codec_prep), upscale_key=upscale,
            progress=_prog(progress),
        )
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(_friendly(e))
    progress(1.0, desc="Готово")
    path = str(Path(path).resolve())
    return path, path, info



def build_ui():
    """Build the UI (kept as a compatibility entry point)."""
    from app.ui.layout import build_ui as _build_ui
    return _build_ui(sys.modules[__name__])

if __name__ == "__main__":
    print("=" * 60)
    print("  Local Face Swap")
    print("  http://127.0.0.1:7860")
    print("=" * 60)
    from app.gpu_utils import setup_device, get_device_name
    setup_device(verbose=True)
    try:
        get_swapper()
        print(f"[OK] FaceSwapper готов, устройство: {get_device_name()}")
    except Exception as e:
        print(f"[WARN] Модели: {e}")

    demo, theme, css = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=theme,
        css=css,
    )