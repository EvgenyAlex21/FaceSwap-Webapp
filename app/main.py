#!/usr/bin/env python3
"""Local Face Swap """

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple
import math

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import cv2
import numpy as np
from PIL import Image

from app.swapper import get_swapper
from app.utils import OUTPUTS, unique_name, write_image, get_video_info, copy_to_uploads
from app.inpaint import inpaint_region, inpaint_ai

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
        return gr.update(maximum=60, value=0), gr.update(maximum=60, value=0), "Загрузите видео"
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
            f"Длительность: **{dur:.1f}** сек · {total} кадров · {fps:.1f} fps",
        )
    except Exception as e:
        return gr.update(), gr.update(), f"Ошибка: {e}"


def detect_video_faces(video_path, det_mode, progress=gr.Progress(track_tqdm=False)):
    if video_path is None:
        return "Загрузите видео", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    progress(0.05, desc="Сканирование видео...")
    try:
        video_path = str(copy_to_uploads(video_path, prefix="vid_scan"))
    except Exception:
        video_path = video_path

    sw = _apply_det(det_mode)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return "Ошибка чтения видео", *[None]*MAX_FACES, *[gr.update(visible=False)]*MAX_FACES, 0, gr.update(choices=[], value=None)

    step = max(1, total_frames // 150)
    unique_faces = []
    crops = []

    for f_idx in range(0, total_frames, step):
        progress(min(0.05 + 0.8 * (f_idx/total_frames), 0.85), desc=f"Анализ...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: break
        
        faces = sw.get_faces(frame)
        for face in faces:
            is_new = True
            for uf in unique_faces:
                cx, cy = (face.bbox[0]+face.bbox[2])/2, (face.bbox[1]+face.bbox[3])/2
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

    progress(0.15, desc="Маска создана. Подготовка модели...")

    try:
        from app.inpaint import _pipe, _pipe_model_name, MODELS
        
        if "TryOffAnyone" in model_choice:
            local_model_file = MODELS / "tryOffAnyone.safetensors"
        elif "LUSTIFY" in model_choice:
            local_model_file = MODELS / "lustifySDXLNSFW_v20-inpainting.safetensors"
        else:
            local_model_file = MODELS / "realisticVisionV60B1_v50VAE-inpainting.safetensors"

        if _pipe is None or _pipe_model_name != str(local_model_file):
            progress(0.25, desc=f"Загрузка модели {model_choice} (может занять 1-2 минуты)...")
        else:
            progress(0.25, desc="Модель загружена. Генерация...")

        path = inpaint_ai(
            image_bgr, np.asarray(mask),
            prompt=str(ai_prompt or "").strip() or "nude body, realistic skin, natural anatomy, photorealistic",
            negative=str(ai_negative or "").strip(),
            steps=int(ai_steps), guidance=float(ai_guidance),
            strength=float(ai_strength), seed=int(ai_seed),
            model_choice=model_choice,
        )
        
        progress(0.95, desc="Финальная сборка...")
        return _to_pil(cv2.imread(str(path))), str(path)
    except Exception as e:
        raise gr.Error(str(e))


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

:root {
    --ink: #E8EEF7;
    --muted: #94A5BE;
    --line: #243B5C;
    --panel: #0F1D33;
    --panel-2: #0A1629;
    --accent: #3BB5D6;
    --grad: linear-gradient(120deg, #2D6BDB 0%, #1B9BC0 58%, #19A7A0 100%);
    --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
}

body {
    background:
        radial-gradient(1100px 560px at 92% -8%, rgba(27, 155, 192, 0.20), transparent 60%),
        radial-gradient(900px 520px at -8% 6%, rgba(45, 107, 219, 0.22), transparent 58%),
        linear-gradient(180deg, #091426 0%, #0B1B31 100%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    background: transparent !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    color: var(--ink);
}

/* ---------- Шапка ---------- */
.app-header {
    display: flex; align-items: center; justify-content: space-between; gap: 28px;
    margin: 28px 0 22px; padding: 30px 36px;
    border-radius: 14px;
    background: linear-gradient(115deg, #14335F 0%, #1B5A8E 52%, #167F88 100%);
    border: 1px solid rgba(255, 255, 255, 0.14);
}
.gradio-container .app-header h1 {
    font-family: var(--serif) !important; font-weight: 600 !important;
    font-size: 2rem !important; letter-spacing: -0.01em;
    color: #FFFFFF !important; margin: 0 0 4px 0 !important;
}
.gradio-container .app-header p { margin: 0 !important; color: rgba(255, 255, 255, 0.82) !important; font-size: 0.97rem !important; }
.gradio-container .app-header .header-note {
    max-width: 330px; padding-left: 24px; border-left: 1px solid rgba(255, 255, 255, 0.28);
    font-size: 0.83rem !important; line-height: 1.55; color: rgba(255, 255, 255, 0.78) !important;
}

/* ---------- Вкладки ---------- */
.tabs [role="tablist"], .tab-nav, .tab-wrapper { justify-content: center !important; }
.tabs button[role="tab"] {
    font-family: inherit !important; font-size: 0.97rem !important; font-weight: 500 !important;
    color: var(--muted) !important; background: transparent !important;
    padding: 10px 4px !important; margin: 0 20px !important;
    border: none !important; border-radius: 0 !important;
}
.tabs button[role="tab"]:hover { color: var(--ink) !important; }
.tabs button[role="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    background: transparent var(--grad) no-repeat bottom / 100% 2px !important;
}

/* ---------- Инструкция ---------- */
.gradio-container .guide {
    display: grid; grid-template-columns: 200px 1fr; gap: 8px 32px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 26px; margin: 18px 0 22px;
}
.gradio-container .guide h2 {
    font-family: var(--serif) !important; font-size: 1.15rem !important; font-weight: 600 !important;
    color: #FFFFFF !important; margin: 0 !important; border: none !important;
}
.gradio-container .guide p { margin: 0 0 10px; color: var(--ink); font-size: 0.93rem; line-height: 1.6; }
.gradio-container .guide ol { margin: 0 0 12px; padding-left: 20px; color: var(--ink); font-size: 0.93rem; line-height: 1.65; }
.gradio-container .guide ol li::marker { color: var(--accent); font-weight: 600; }
.gradio-container .guide strong { font-weight: 600; color: #FFFFFF; }
.gradio-container .guide .guide-note {
    margin: 0; padding-top: 12px; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 0.86rem; line-height: 1.6;
}

/* ---------- Заголовки разделов и подсказки ---------- */
.gradio-container .section-title h3 {
    font-family: var(--serif) !important; font-size: 1.25rem !important; font-weight: 600 !important;
    color: #FFFFFF !important; letter-spacing: -0.005em; margin: 26px 0 2px 0 !important;
}
.gradio-container .hint, .gradio-container .hint p {
    color: var(--muted) !important; font-size: 0.9rem !important; line-height: 1.55;
}
.gradio-container .hint strong { color: var(--ink); }
.status-text {
    min-height: 28px !important; display: flex !important; align-items: center !important;
    margin-top: 6px !important; color: #9FD8EA !important;
    font-size: 0.93rem !important; font-weight: 500 !important;
}

/* ---------- Убираем лишние скроллбары в Markdown и HTML ---------- */
.prose, .prose pre, .prose code, .md, .html-container {
    overflow: visible !important;
    overflow-x: visible !important;
    overflow-y: visible !important;
    height: auto !important;
    max-height: none !important;
}

/* ---------- Группы (слоты лиц) ---------- */
.gradio-container .gr-group { background: var(--panel) !important; border: 1px solid var(--line) !important; }
.gradio-container .gr-group .styler, .gradio-container .form { background: transparent !important; }
.gradio-container .gr-group .prose { padding: 10px 14px 2px !important; }

/* ---------- Кнопки ---------- */
.primary-btn {
    background: var(--grad) !important; border: none !important;
    color: #fff !important; font-weight: 500 !important; letter-spacing: 0.01em;
    border-radius: 8px !important; padding: 14px 28px !important;
}
.primary-btn:hover { filter: brightness(1.1); }
.secondary-btn {
    background: rgba(255, 255, 255, 0.04) !important; border: 1px solid #2F4A70 !important;
    color: var(--ink) !important; font-weight: 500 !important; border-radius: 8px !important;
}
.secondary-btn:hover { border-color: var(--accent) !important; color: #8FDBF0 !important; }
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible {
    outline: 2px solid var(--accent) !important; outline-offset: 2px;
}

/* ---------- Индикатор загрузки ---------- */
[data-testid="status-tracker"].wrap.default {
    background: rgba(9, 20, 38, 0.80) !important;
    backdrop-filter: blur(3px);
    border-radius: 10px;
}
[data-testid="status-tracker"] .eta-bar {
    background: var(--grad) !important; opacity: 0.16 !important;
}
[data-testid="status-tracker"] .margin { transform: scale(2.8); transform-origin: center; }
[data-testid="status-tracker"] .progress-text {
    position: absolute !important; left: 0 !important; right: 0 !important; bottom: 18px !important;
    text-align: center !important; padding: 0 16px !important;
    font-family: inherit !important; font-size: 0.95rem !important; color: #BFD9F2 !important;
}

/* ---------- Нижняя заметка ---------- */
.gradio-container .footnote {
    margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 0.82rem; line-height: 1.6;
}
.gradio-container .footnote code { background: #16294A; padding: 1px 6px; border-radius: 4px; color: #BFD9F2; }

footer { display: none !important; }

@media (max-width: 760px) {
    .gradio-container .guide { grid-template-columns: 1fr; }
    .app-header { flex-direction: column; align-items: flex-start; padding: 24px; }
    .gradio-container .app-header .header-note { padding-left: 0; border-left: none; }
    .tabs button[role="tab"] { margin: 0 10px !important; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
"""


def _guide(title_text, intro, steps, note):
    items = "".join(f"<li>{s}</li>" for s in steps)
    return (
        '<div class="guide">'
        f'<h2>{title_text}</h2>'
        '<div>'
        f'<p>{intro}</p>'
        f'<ol>{items}</ol>'
        f'<p class="guide-note">{note}</p>'
        '</div></div>'
    )


def build_ui():
    ink, muted, line = "#E8EEF7", "#94A5BE", "#243B5C"
    panel, field = "#0F1D33", "#0A1629"
    sapphire = gr.themes.Color(
        c50="#EAF3FB", c100="#CFE3F5", c200="#A3CBEB", c300="#6FB0DE", c400="#3BB5D6",
        c500="#2D8FD0", c600="#2D6BDB", c700="#24509F", c800="#1B3A73", c900="#142B55", c950="#0C1B38",
    )
    slate_blue = gr.themes.Color(
        c50="#F1F5FA", c100="#DCE5F1", c200="#B7C5DA", c300="#94A5BE", c400="#6F82A0",
        c500="#4F6482", c600="#3A4E6C", c700="#2A3E5D", c800="#1B2D49", c900="#0F1D33", c950="#0A1629",
    )
    theme = gr.themes.Base(
        primary_hue=sapphire, secondary_hue=slate_blue, neutral_hue=slate_blue,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0A1526", body_background_fill_dark="#0A1526",
        body_text_color=ink, body_text_color_dark=ink,
        body_text_color_subdued=muted, body_text_color_subdued_dark=muted,
        background_fill_primary=panel, background_fill_primary_dark=panel,
        background_fill_secondary=field, background_fill_secondary_dark=field,
        block_background_fill=panel, block_background_fill_dark=panel,
        block_border_color=line, block_border_color_dark=line,
        border_color_primary=line, border_color_primary_dark=line,
        border_color_accent="#3BB5D6", border_color_accent_dark="#3BB5D6",
        block_border_width="1px", block_radius="10px",
        block_shadow="none", block_shadow_dark="none",
        block_label_background_fill=panel, block_label_background_fill_dark=panel,
        block_label_text_color=muted, block_label_text_color_dark=muted,
        block_title_text_color=ink, block_title_text_color_dark=ink,
        block_info_text_color=muted, block_info_text_color_dark=muted,
        input_background_fill=field, input_background_fill_dark=field,
        input_border_color="#2A4268", input_border_color_dark="#2A4268",
        input_placeholder_color="#5F7392", input_placeholder_color_dark="#5F7392",
        input_radius="8px",
        loader_color="#3BB5D6", loader_color_dark="#3BB5D6",
        slider_color="#2D8FD0", slider_color_dark="#2D8FD0",
        checkbox_background_color=field, checkbox_background_color_dark=field,
        checkbox_background_color_selected="#2D6BDB", checkbox_background_color_selected_dark="#2D6BDB",
        checkbox_border_color="#2A4268", checkbox_border_color_dark="#2A4268",
        button_primary_background_fill="#2D6BDB", button_primary_background_fill_dark="#2D6BDB",
        button_primary_background_fill_hover="#2D6BDB", button_primary_background_fill_hover_dark="#2D6BDB",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill=field, button_secondary_background_fill_dark=field,
        button_secondary_text_color=ink, button_secondary_text_color_dark=ink,
        button_large_radius="8px", button_small_radius="8px",
    )

    with gr.Blocks(title="Local Face Swap") as demo:
        gr.HTML("""
        <div class="app-header">
            <div>
                <h1>Local Face Swap</h1>
                <p>Замена лиц на фото и видео, умное выделение</p>
            </div>
            <p class="header-note">Обработка выполняется на вашем компьютере. Исходные файлы не передаются во внешние сервисы.</p>
        </div>
        """)

        with gr.Tabs():
            # ===================== PHOTO =====================
            with gr.TabItem("Фото"):
                with gr.Column(elem_classes=["tab-container"]):
                    gr.HTML(_guide(
                        "Как это работает",
                        "Программа находит лица на целевом фото и заменяет каждое из них лицом с загруженного вами снимка.",
                        [
                            "Проверьте параметры обработки. Значения по умолчанию подходят для большинства фотографий.",
                            "Загрузите целевое фото и нажмите «Найти лица».",
                            "Для каждого найденного лица загрузите фото замены и нажмите «Найти лицо», чтобы проверить превью.",
                            "Нажмите «Заменить лица». Результат появится ниже и сохранится в папке <strong>outputs/</strong>.",
                        ],
                        "Для максимальной детализации выберите детекцию «Качественный (640)» или «Максимум (800)» и резкость «Среднее» или «Сильное». "
                        "Для быстрого результата — детекцию «Быстрый (320)» или «Баланс (480)». "
                        "Размер готового изображения приводится к 1920 px по длинной стороне.",
                    ))

                    photo_quality = gr.State("Full HD (1920)")

                    with gr.Accordion("Параметры обработки", open=True):
                        with gr.Row():
                            photo_det = gr.Dropdown(
                                list(DET_SIZES.keys()), value="Качественный (640)",
                                label="Детекция лиц",
                                info="Точность поиска лиц. Выше значение — точнее, но медленнее",
                            )
                            photo_enhance = gr.Dropdown(
                                list(ENHANCE_STRENGTH.keys()), value="Среднее",
                                label="Резкость",
                                info="Степень улучшения деталей лица",
                            )
                            photo_format = gr.Radio(
                                ["PNG", "JPG"], value="PNG", label="Формат файла",
                                info="PNG — без потерь, JPG — компактнее",
                            )

                    gr.Markdown("### 1. Целевое фото", elem_classes=["section-title"])
                    gr.Markdown("Фото, **на котором** нужно заменить лица.", elem_classes=["hint"])
                    tgt_photo = gr.Image(label="Целевое фото", type="pil", height=320)
                    btn_find_tgt = gr.Button("Найти лица", elem_classes=["secondary-btn"])
                    tgt_status = gr.Markdown("", elem_classes=["status-text"])
                    face_count = gr.State(0)

                    gr.Markdown("### 2. Замена для каждого найденного лица", elem_classes=["section-title"])
                    gr.Markdown(
                        "Слева — лицо с целевого фото. Справа загрузите фото человека, "
                        "чьё лицо нужно подставить. Кнопка «Найти лицо» покажет превью.",
                        elem_classes=["hint"],
                    )

                    slot_targets = []
                    slot_sources = []
                    slot_src_previews = []
                    slot_src_status = []
                    slot_find_btns = []
                    slot_groups = []

                    for i in range(MAX_FACES):
                        with gr.Group(visible=False) as grp:
                            gr.Markdown(f"**Лицо {i + 1}** (с целевого фото)")
                            with gr.Row():
                                t_crop = gr.Image(
                                    label=f"Найдено на цели #{i + 1}",
                                    height=140, interactive=False,
                                )
                                with gr.Column():
                                    s_img = gr.Image(
                                        label=f"Фото для замены #{i + 1}",
                                        type="pil", height=140,
                                    )
                                    s_btn = gr.Button(
                                        f"Найти лицо (замена #{i + 1})",
                                        elem_classes=["secondary-btn"], size="sm",
                                    )
                                    s_prev = gr.Image(
                                        label="Превью замены", height=100, interactive=False,
                                    )
                                    s_stat = gr.Markdown("", elem_classes=["status-text"])
                        slot_groups.append(grp)
                        slot_targets.append(t_crop)
                        slot_sources.append(s_img)
                        slot_src_previews.append(s_prev)
                        slot_src_status.append(s_stat)
                        slot_find_btns.append(s_btn)

                    tgt_photo.change(
                        clear_target_and_slots,
                        [tgt_photo],
                        [tgt_status, *slot_targets, *slot_groups, face_count,
                         *slot_src_previews, *slot_src_status],
                    )

                    btn_find_tgt.click(
                        lambda: "Поиск лиц...",
                        None, tgt_status,
                        show_progress="hidden",
                    ).then(
                        detect_target_faces,
                        [tgt_photo, photo_det],
                        [tgt_status, *slot_targets, *slot_groups, face_count,
                         *slot_src_previews, *slot_src_status],
                        show_progress="full",
                        show_progress_on=[tgt_photo],
                    )

                    for i in range(MAX_FACES):
                        slot_find_btns[i].click(
                            detect_source_face,
                            [slot_sources[i], photo_det],
                            [slot_src_previews[i], slot_src_status[i]],
                            show_progress="full",
                            show_progress_on=[slot_sources[i]],
                        )
                        slot_sources[i].change(
                            source_changed,
                            [slot_sources[i]],
                            [slot_src_previews[i], slot_src_status[i]],
                        )

                    gr.Markdown("### 3. Результат", elem_classes=["section-title"])
                    btn_swap_photo = gr.Button(
                        "Заменить лица", variant="primary",
                        elem_classes=["primary-btn"], size="lg",
                    )
                    out_photo = gr.Image(label="Результат", height=400)
                    out_path_p = gr.Textbox(label="Файл", interactive=False)

                    btn_swap_photo.click(
                        swap_photo_slots,
                        [
                            tgt_photo, face_count, photo_det, photo_quality,
                            photo_enhance, photo_format, *slot_sources,
                        ],
                        [out_photo, out_path_p],
                        show_progress="full",
                        show_progress_on=[out_photo],
                    )

            # ===================== VIDEO =====================
            with gr.TabItem("Видео"):
                with gr.Column(elem_classes=["tab-container"]):
                    gr.HTML(_guide(
                        "Как это работает",
                        "Программа заменяет выбранное лицо в видео лицом с вашего фото. Оригинальный звук можно сохранить.",
                        [
                            "Проверьте параметры обработки. Значения по умолчанию подходят для большинства роликов.",
                            "Загрузите фото с лицом-источником и нажмите «Найти лица на фото».",
                            "Загрузите видео, нажмите «Найти лица в видео» и задайте начало и конец обрабатываемого фрагмента.",
                            "В разделе «Сопоставление лиц» выберите, чьё лицо вставляем и кому.",
                            "Нажмите «Заменить лица в видео». Готовый файл сохранится в папке <strong>outputs/</strong>.",
                        ],
                        "Для лучшего результата: детекция «Качественный (640)» или «Максимум (800)», резкость «Среднее» или «Сильное», "
                        "кодирование CRF 18–20, детекция на каждом кадре. Для быстрого черновика: детекция 320–480, "
                        "каждые 2–3 кадра, CRF 23–26. Скорость зависит от видеокарты: без CUDA обработка занимает заметно больше времени.",
                    ))

                    with gr.Accordion("Параметры обработки", open=True):
                        with gr.Row():
                            vid_det = gr.Dropdown(
                                list(DET_SIZES.keys()), value="Качественный (640)",
                                label="Детекция лиц",
                                info="Точность поиска лиц. Выше значение — точнее, но медленнее",
                            )
                            vid_quality = gr.Dropdown(
                                list(VIDEO_QUALITY.keys()), value="Хорошее (CRF 20)",
                                label="Качество кодирования",
                                info="Чем меньше CRF, тем выше качество и больше размер файла",
                            )
                        with gr.Row():
                            vid_res = gr.Dropdown(
                                list(VIDEO_RESOLUTIONS.keys()), value="Как есть (оригинал)",
                                label="Разрешение",
                            )
                            vid_fps = gr.Dropdown(
                                list(VIDEO_FPS.keys()), value="Как есть (оригинал)",
                                label="FPS",
                            )
                        with gr.Row():
                            vid_enhance = gr.Dropdown(
                                list(ENHANCE_STRENGTH.keys()), value="Среднее",
                                label="Резкость",
                                info="Степень улучшения деталей лица",
                            )
                            detect_every = gr.Slider(
                                1, 5, value=1, step=1,
                                label="Детекция каждые N кадров",
                                info="1 — каждый кадр (точнее), 2–5 — быстрее",
                            )
                            keep_audio = gr.Checkbox(value=True, label="Сохранять звук")

                    # --- ИСТОЧНИК ---
                    gr.Markdown("### 1. Лицо-источник (с фото)", elem_classes=["section-title"])
                    gr.Markdown("Фото человека, чьё лицо нужно вставить в видео.", elem_classes=["hint"])
                    vsrc = gr.Image(label="Фото лица", type="pil", height=200)
                    btn_find_src = gr.Button("Найти лица на фото", elem_classes=["secondary-btn"])
                    src_status = gr.Markdown("", elem_classes=["status-text"])
                    src_count = gr.State(0)

                    src_face_slots = []
                    for i in range(MAX_FACES):
                        with gr.Group(visible=False) as src_grp:
                            with gr.Row():
                                src_crop = gr.Image(label=f"Источник #{i+1}", height=100, interactive=False)
                        src_face_slots.append(src_grp)
                        src_face_slots.append(src_crop)

                    src_grps_only = [src_face_slots[i] for i in range(0, len(src_face_slots), 2)]
                    src_crops_only = [src_face_slots[i] for i in range(1, len(src_face_slots), 2)]

                    # --- ВИДЕО ---
                    gr.Markdown("### 2. Видео", elem_classes=["section-title"])
                    gr.Markdown("Ролик, **в котором** нужно заменить лицо. Ползунками ниже задаётся обрабатываемый фрагмент.", elem_classes=["hint"])
                    tgt_video = gr.Video(label="Видео", height=220)
                    btn_find_vid = gr.Button("Найти лица в видео (быстро)", elem_classes=["secondary-btn"])
                    video_info = gr.Markdown("", elem_classes=["status-text"])
                    vid_status = gr.Markdown("", elem_classes=["status-text"])
                    vid_count = gr.State(0)

                    vid_face_slots = []
                    for i in range(MAX_FACES):
                        with gr.Group(visible=False) as vid_grp:
                            with gr.Row():
                                vid_crop = gr.Image(label=f"Лицо в видео #{i+1}", height=100, interactive=False)
                        vid_face_slots.append(vid_grp)
                        vid_face_slots.append(vid_crop)

                    vid_grps_only = [vid_face_slots[i] for i in range(0, len(vid_face_slots), 2)]
                    vid_crops_only = [vid_face_slots[i] for i in range(1, len(vid_face_slots), 2)]

                    with gr.Row():
                        start_sec = gr.Slider(0, 60, value=0, step=0.5, label="Начало (сек)")
                        end_sec = gr.Slider(0, 60, value=15, step=0.5, label="Конец (сек)")

                    # --- СОПОСТАВЛЕНИЕ ---
                    gr.Markdown("### 3. Сопоставление лиц", elem_classes=["section-title"])
                    with gr.Row():
                        src_choice = gr.Dropdown(choices=[], label="Лицо из фото (Кого вставляем?)")
                        vid_choice = gr.Dropdown(choices=[], label="Лицо в видео (Кому вставляем?)")

                    tgt_video.change(on_video_upload, [tgt_video], [start_sec, end_sec, video_info])

                    btn_find_src.click(
                        lambda: "Поиск...", None, src_status, show_progress="hidden"
                    ).then(
                        detect_src_faces,
                        [vsrc, vid_det],
                        [src_status, *src_crops_only, *src_grps_only, src_count, src_choice],
                        show_progress="full",
                        show_progress_on=[vsrc],
                    )

                    btn_find_vid.click(
                        lambda: "Сканирование видео...", None, vid_status, show_progress="hidden"
                    ).then(
                        detect_video_faces,
                        [tgt_video, vid_det],
                        [vid_status, *vid_crops_only, *vid_grps_only, vid_count, vid_choice],
                        show_progress="full",
                        show_progress_on=[tgt_video],
                    )

                    gr.Markdown("### 4. Результат", elem_classes=["section-title"])
                    btn_swap_v = gr.Button(
                        "Заменить лица в видео", variant="primary",
                        elem_classes=["primary-btn"], size="lg",
                    )
                    out_video = gr.Video(label="Результат")
                    out_path_v = gr.Textbox(label="Файл", interactive=False)

                    btn_swap_v.click(
                        swap_video_slots,
                        [vsrc, tgt_video, start_sec, end_sec,
                         vid_det, detect_every, vid_enhance, vid_quality,
                         vid_res, vid_fps, keep_audio,
                         src_choice, vid_choice,
                         src_count, vid_count],
                        [out_video, out_path_v],
                        show_progress="full",
                        show_progress_on=[out_video],
                    )

            # ===================== INPAINT =====================
            with gr.TabItem("Выделение"):
                with gr.Column(elem_classes=["tab-container"]):
                    gr.HTML(_guide(
                        "Как это работает",
                        "Программа перерисовывает выделенную кистью область с помощью нейросети (диффузии).",
                        [
                            "Проверьте параметры обработки. Значения по умолчанию подходят для большинства случаев.",
                            "Загрузите фото в редактор масок ниже.",
                            "Кистью закрасьте область, которую нужно перерисовать или убрать. Ластиком поправьте края.",
                            "Нажмите «Обработать выделенное». Результат сохранится в папке <strong>outputs/</strong>.",
                        ],
                        "Для базовых заливок можно использовать алгоритм OpenCV. AI-диффузия даёт реалистичную текстуру, "
                        "но требует скачивания модели (~4 GB) при первом запуске. Для видеокарт с 4 GB VRAM "
                        "рекомендуется модель «Realistic Vision (SD 1.5)».",
                    ))

                    with gr.Accordion("Параметры обработки", open=True):
                        inpaint_model = gr.Dropdown(
                            [
                                "Realistic Vision (SD 1.5)",
                                "TryOffAnyone (SD 1.5, без промпта)",
                                "LUSTIFY SDXL (нужно много VRAM)"
                            ],
                            value="Realistic Vision (SD 1.5)",
                            label="Модель для AI-диффузии",
                            info="TryOffAnyone не использует промпт. LUSTIFY тяжелая для 4GB VRAM."
                        )

                        with gr.Row():
                            inpaint_method = gr.Radio(
                                ["Telea (быстрее)", "Navier-Stokes (чище края)"],
                                value="Telea (быстрее)",
                                label="Алгоритм (OpenCV)",
                                info="Для базовых заливок без нейросети",
                            )
                            inpaint_radius = gr.Slider(3, 15, value=7, step=1, label="Радиус (OpenCV)")

                        with gr.Accordion("AI-настройки (для диффузии)", open=False):
                            ai_prompt = gr.Textbox(
                                label="Промпт",
                                value="nude body, realistic skin, natural anatomy, high detail, photorealistic",
                                lines=3,
                            )
                            ai_negative = gr.Textbox(
                                label="Negative prompt",
                                value=(
                                    "clothes, clothing, fabric, text, watermark, logo, deformed, "
                                    "extra limbs, bad anatomy, low quality, blurry"
                                ),
                                lines=3,
                            )
                            with gr.Row():
                                ai_steps = gr.Slider(15, 50, value=30, step=1, label="Шаги")
                                ai_guidance = gr.Slider(3, 12, value=7.5, step=0.5, label="Guidance")
                                ai_strength = gr.Slider(0.3, 1.0, value=1.0, step=0.05, label="Strength")
                                ai_seed = gr.Number(value=0, label="Seed", precision=0)

                    gr.Markdown("### 1. Загрузка и маска", elem_classes=["section-title"])
                    gr.Markdown(
                        "Загрузите фото и закрасьте область. Кисть и ластик находятся в верхней панели редактора.",
                        elem_classes=["hint"],
                    )

                    try:
                        inpaint_editor = gr.ImageEditor(
                            label="Рисование маски",
                            type="numpy",
                            height=420,
                            brush=gr.Brush(default_size=28, colors=["#3BB5D6"], default_color="#3BB5D6"),
                            eraser=gr.Eraser(default_size=28),
                            layers=False,
                            transforms=(),
                        )
                    except TypeError:
                        inpaint_editor = gr.ImageEditor(
                            label="Рисование маски",
                            type="numpy",
                            height=420,
                        )

                    gr.Markdown("### 2. Результат", elem_classes=["section-title"])
                    btn_inpaint = gr.Button(
                        "Обработать выделенное",
                        variant="primary",
                        elem_classes=["primary-btn"],
                        size="lg",
                    )
                    
                    inpaint_out = gr.Image(label="Результат", height=360)
                    inpaint_path = gr.Textbox(label="Файл", interactive=False)

                    btn_inpaint.click(
                        run_inpaint,
                        [
                            inpaint_editor, inpaint_method, inpaint_radius,
                            ai_prompt, ai_negative, ai_steps, ai_guidance, ai_strength, ai_seed,
                            inpaint_model, 
                        ],
                        [inpaint_out, inpaint_path],
                        show_progress="full",
                        show_progress_on=[inpaint_out],
                    )

        gr.HTML("""
        <div class="footnote">
            Результаты сохраняются в папке <code>outputs/</code>. Используйте только изображения и видео,
            на обработку которых у вас есть права и согласие изображённых на них людей.
        </div>
        """)

    return demo, theme, CUSTOM_CSS


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