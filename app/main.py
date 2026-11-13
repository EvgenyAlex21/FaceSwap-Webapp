#!/usr/bin/env python3
"""Local Face Swap — персональные настройки по вкладкам, слоты лиц на фото."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import cv2
import numpy as np
from PIL import Image

from app.swapper import get_swapper
from app.utils import OUTPUTS, unique_name, write_image, get_video_info, copy_to_uploads
from app.inpaint import inpaint_region


MAX_FACES = 6

DET_SIZES = {
    "Быстрый (320)": (320, 320),
    "Баланс (480)": (480, 480),
    "Качественный (640)": (640, 640),
    "Максимум (800)": (800, 800),
}

# Целевое разрешение длинной стороны (сохраняем аспект)
PHOTO_QUALITY = {
    "Как есть (без изменения размера)": 0,
    "HD (1280)": 1280,
    "Full HD (1920)": 1920,
    "QHD / 2K (2560)": 2560,
    "4K UHD (3840)": 3840,
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


# ---------- PHOTO: detect target faces → slots ----------

def detect_target_faces(target_img, det_mode):
    """
    Returns:
      status markdown,
      crops для слотов (6),
      visibility flags (6),
      face_count,
      cleared source previews (6),
      cleared source statuses (6),
    """
    empty = [None] * MAX_FACES
    hidden = [gr.update(visible=False)] * MAX_FACES
    # при пересчёте целевых лиц — сбрасываем превью/статус замены
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
            "❌ Лица не найдены на целевом фото",
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
        f"✅ Найдено лиц: **{len(faces)}** — загрузите замену для каждого",
        *crops,
        *vis,
        len(faces),
        *clear_prev,
        *clear_stat,
    )


def clear_target_and_slots(target_img):
    """Вызывается при изменении целевого фото.
    Если фото удалено — прячем слоты и чистим превью/статусы замены.
    Если загружено — просто сбрасываем найденные лица (нужно снова нажать «Найти лица»).
    """
    if target_img is not None:
        # новое фото — сбрасываем результаты поиска, но не трогаем замены
        return (
            "Нажмите «Найти лица», чтобы обновить слоты",
            *[None] * MAX_FACES,
            *[gr.update(visible=False)] * MAX_FACES,
            0,
            *[None] * MAX_FACES,
            *[""] * MAX_FACES,
        )
    # фото удалено — чистим всё
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
        return None, "❌ Лицо не найдено"
    crop = _crop_face(bgr, faces[0])
    return _to_pil(crop), "✅ Лицо найдено"


def source_changed(src_img):
    """При изменении/удалении заменяющего фото сбрасываем превью и статус."""
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
    # --- Проверки ---
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


def swap_video(
    src_img, target_video, start_sec, end_sec,
    det_mode, detect_every, enhance_mode, video_quality, keep_audio,
    progress=gr.Progress(track_tqdm=False),
):
    if target_video is None:
        raise gr.Error("Загрузите видео")
    if src_img is None:
        raise gr.Error("Загрузите фото лица-источника")

    progress(0.02, desc="Подготовка...")
    try:
        video_path = str(copy_to_uploads(target_video, prefix="vid"))
    except Exception:
        video_path = target_video

    sw = _apply_det(det_mode)
    src_path = _save_tmp(src_img, "vsrc")
    faces = sw.get_faces(_to_bgr(src_img))
    if not faces:
        raise gr.Error("На фото-источнике лицо не найдено")

    source_map = {0: (src_path, 0)}
    strength = ENHANCE_STRENGTH.get(enhance_mode, 0.45)
    crf = VIDEO_QUALITY.get(video_quality, 20)

    def cb(p, msg=""):
        progress(p, desc=msg or "Обработка видео...")

    result_path = sw.swap_video(
        source_map=source_map,
        target_video_path=video_path,
        swap_all_with_first=True,
        start_sec=float(start_sec or 0),
        end_sec=float(end_sec or 0),
        detect_every_n=int(detect_every),
        enhance=strength > 0,
        enhance_strength=strength,
        upscale=1,
        video_crf=crf,
        keep_audio=bool(keep_audio),
        progress_callback=cb,
    )
    path = str(Path(result_path).resolve())
    return path, path


# ---------- INPAINT / EDIT / ANIM ----------

def run_inpaint(editor_data, method, radius, progress=gr.Progress(track_tqdm=False)):
    if editor_data is None:
        raise gr.Error("Загрузите изображение и закрасьте область")
    progress(0.1, desc="Разбор маски...")

    def _arr_to_bgr(arr):
        arr = np.array(arr)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if not isinstance(editor_data, dict):
        raise gr.Error("Загрузите фото и закрасьте область кистью")

    bg = editor_data.get("background") or editor_data.get("composite")
    if bg is None:
        raise gr.Error("Нет изображения")
    layers = editor_data.get("layers") or []
    mask = None
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
    elif editor_data.get("composite") is not None and editor_data.get("background") is not None:
        c = np.array(editor_data["composite"])
        b = np.array(editor_data["background"])
        if c.shape[:2] != b.shape[:2]:
            c = cv2.resize(c, (b.shape[1], b.shape[0]))
        diff = np.abs(c.astype(np.int16)[..., :3] - b.astype(np.int16)[..., :3])
        mask = (np.any(diff > 15, axis=2)).astype(np.uint8) * 255

    if mask is None or int(np.asarray(mask).sum()) == 0:
        raise gr.Error("Маска пустая — закрасьте область кистью")

    progress(0.45, desc="Инпейнт...")
    path = inpaint_region(
        _arr_to_bgr(bg), np.asarray(mask),
        method="telea" if "Telea" in str(method) else "ns",
        radius=int(radius), refine=True,
    )
    progress(1.0, desc="Готово")
    return _to_pil(cv2.imread(str(path))), str(path)


def run_edit(image, prompt, steps, image_guidance, guidance, seed,
             progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Загрузите фото")
    if not prompt or not str(prompt).strip():
        raise gr.Error("Введите инструкцию (EN), например: add sunglasses")
    progress(0.1, desc="Обработка...")
    from app.editor import edit_image
    path = edit_image(
        _to_bgr(image), prompt=str(prompt), steps=int(steps),
        image_guidance=float(image_guidance), guidance_scale=float(guidance),
        seed=int(seed),
    )
    progress(1.0, desc="Готово")
    return _to_pil(cv2.imread(str(path))), str(path)


def run_animate(image, mode, prompt, seconds, svd_frames, motion, seed,
                progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Загрузите фото")
    bgr = _to_bgr(image)
    if prompt:
        print(f"[Animator] Промпт: {prompt}")

    def cb(p, msg=""):
        progress(p, desc=msg or "Анимация...")

    if mode == "Нейро (SVD) — нужен GPU":
        from app.animator import animate_svd
        path = animate_svd(
            bgr, num_frames=int(svd_frames), fps=7,
            motion_bucket_id=int(motion), seed=int(seed),
            progress_callback=cb,
        )
    else:
        from app.animator import animate_simple
        simple_mode = {
            "Zoom in": "zoom_in", "Zoom out": "zoom_out",
            "Pan right": "pan_right", "Pan left": "pan_left", "Pulse": "pulse",
        }.get(mode, "zoom_in")
        path = animate_simple(
            bgr, seconds=float(seconds), fps=24, mode=simple_mode,
            progress_callback=cb,
        )
    path = str(Path(path).resolve())
    return path, path


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; font-family: 'Inter', system-ui, sans-serif !important; }
.main-header {
    text-align: center; padding: 28px 24px 16px;
    background: linear-gradient(145deg, #0c0a1d 0%, #1a1440 40%, #0f172a 100%);
    border-radius: 18px; margin-bottom: 18px;
    border: 1px solid rgba(167,139,250,0.15);
}
.main-header h1 {
    font-size: 2rem !important; font-weight: 700 !important;
    background: linear-gradient(105deg, #c4b5fd, #67e8f9, #6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0 !important;
}
.main-header p { color: #94a3b8 !important; font-size: 0.9rem !important; margin: 0 !important; }
.primary-btn {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
    border-radius: 12px !important; padding: 14px 28px !important;
}
.secondary-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important; border-radius: 10px !important;
}
.face-slot {
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px;
    padding: 8px;
    background: rgba(15,23,42,0.5);
}
.status-text { font-size: 0.9rem !important; color: #a5b4fc !important; }
footer { display: none !important; }
"""


def build_ui():
    theme = gr.themes.Base(
        primary_hue="violet", secondary_hue="slate", neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#080b14", body_background_fill_dark="#080b14",
        block_background_fill="#0f1320", block_background_fill_dark="#0f1320",
        border_color_primary="#1e293b", block_border_width="1px", block_radius="14px",
        button_primary_background_fill="linear-gradient(135deg, #7c3aed, #4f46e5)",
        button_primary_text_color="white", input_background_fill="#0a0e1a",
    )

    with gr.Blocks(title="Local Face Swap") as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>Local Face Swap</h1>
            <p>Замена лиц · Видео со звуком · Инпейнт · Редактор · Анимация</p>
        </div>
        """)

        with gr.Tabs():
            # ===================== PHOTO =====================
            with gr.TabItem("🖼  Фото"):
                with gr.Accordion("⚙️ Настройки", open=True):
                    with gr.Row():
                        photo_det = gr.Dropdown(
                            list(DET_SIZES.keys()), value="Качественный (640)",
                            label="Детекция лиц",
                        )
                        photo_quality = gr.Dropdown(
                            list(PHOTO_QUALITY.keys()),
                            value="Full HD (1920)",
                            label="Качество результата",
                            info="Длинная сторона кадра",
                        )
                    with gr.Row():
                        photo_enhance = gr.Dropdown(
                            list(ENHANCE_STRENGTH.keys()), value="Среднее",
                            label="Резкость",
                        )
                        photo_format = gr.Radio(
                            ["PNG", "JPG"], value="PNG", label="Формат файла",
                        )

                gr.Markdown("### 1. Целевое фото")
                gr.Markdown("Фото, **на котором** нужно заменить лица", elem_classes=["status-text"])
                tgt_photo = gr.Image(label="Целевое фото", type="pil", height=320)
                btn_find_tgt = gr.Button("🔍 Найти лица", elem_classes=["secondary-btn"])
                tgt_status = gr.Markdown("", elem_classes=["status-text"])
                face_count = gr.State(0)

                gr.Markdown("### 2. Замена для каждого найденного лица")
                gr.Markdown(
                    "Сверху — лицо с целевого фото. Ниже загрузите фото человека, "
                    "чьё лицо нужно подставить. «Найти лицо» покажет превью.",
                    elem_classes=["status-text"],
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

                # --- события ---

                # изменение целевого фото: сброс слотов, если фото удалено
                tgt_photo.change(
                    clear_target_and_slots,
                    [tgt_photo],
                    [tgt_status, *slot_targets, *slot_groups, face_count,
                     *slot_src_previews, *slot_src_status],
                )

                # «Найти лица» — статус «⏳» + спиннер + результат
                btn_find_tgt.click(
                    lambda: "⏳ Поиск лиц...",
                    None, tgt_status,
                    show_progress="hidden",
                ).then(
                    detect_target_faces,
                    [tgt_photo, photo_det],
                    [tgt_status, *slot_targets, *slot_groups, face_count,
                     *slot_src_previews, *slot_src_status],
                    show_progress="full",
                )

                # для каждого слота: найти лицо + сброс при смене/удалении фото замены
                for i in range(MAX_FACES):
                    slot_find_btns[i].click(
                        detect_source_face,
                        [slot_sources[i], photo_det],
                        [slot_src_previews[i], slot_src_status[i]],
                        show_progress="full",
                    )
                    slot_sources[i].change(
                        source_changed,
                        [slot_sources[i]],
                        [slot_src_previews[i], slot_src_status[i]],
                    )

                btn_swap_photo = gr.Button(
                    "✨ Заменить лица", variant="primary",
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
                )

            # ===================== VIDEO =====================
            with gr.TabItem("🎬  Видео"):
                with gr.Accordion("⚙️ Настройки видео", open=True):
                    with gr.Row():
                        vid_det = gr.Dropdown(
                            list(DET_SIZES.keys()), value="Качественный (640)",
                            label="Детекция",
                        )
                        vid_quality = gr.Dropdown(
                            list(VIDEO_QUALITY.keys()), value="Хорошее (CRF 20)",
                            label="Качество кодирования",
                        )
                    with gr.Row():
                        vid_enhance = gr.Dropdown(
                            list(ENHANCE_STRENGTH.keys()), value="Среднее",
                            label="Резкость",
                        )
                        detect_every = gr.Slider(
                            1, 5, value=1, step=1,
                            label="Детекция каждые N кадров",
                        )
                        keep_audio = gr.Checkbox(value=True, label="Сохранять звук")

                gr.Markdown("### Лицо-источник")
                vsrc = gr.Image(label="Фото лица", type="pil", height=200)
                gr.Markdown("### Видео")
                tgt_video = gr.Video(label="Видео", height=220)
                video_info = gr.Markdown("", elem_classes=["status-text"])
                with gr.Row():
                    start_sec = gr.Slider(0, 60, value=0, step=0.5, label="Начало (сек)")
                    end_sec = gr.Slider(0, 60, value=15, step=0.5, label="Конец (сек)")
                tgt_video.change(on_video_upload, [tgt_video], [start_sec, end_sec, video_info])
                btn_swap_v = gr.Button(
                    "✨ Заменить лица в видео", variant="primary",
                    elem_classes=["primary-btn"], size="lg",
                )
                out_video = gr.Video(label="Результат")
                out_path_v = gr.Textbox(label="Файл", interactive=False)
                btn_swap_v.click(
                    swap_video,
                    [vsrc, tgt_video, start_sec, end_sec,
                     vid_det, detect_every, vid_enhance, vid_quality, keep_audio],
                    [out_video, out_path_v],
                )

            # ===================== INPAINT =====================
            with gr.TabItem("🖌  Выделение"):
                with gr.Accordion("⚙️ Настройки инпейнта", open=False):
                    inpaint_method = gr.Radio(
                        ["Telea (быстрее)", "Navier-Stokes (чище края)"],
                        value="Telea (быстрее)", label="Алгоритм",
                    )
                    inpaint_radius = gr.Slider(3, 15, value=7, step=1, label="Радиус")
                try:
                    inpaint_editor = gr.ImageEditor(
                        label="Кисть / ластик", type="numpy", height=420,
                        brush=gr.Brush(default_size=28, colors=["#ff2d55"], default_color="#ff2d55"),
                        eraser=gr.Eraser(default_size=28),
                        layers=False, transforms=(),
                    )
                except TypeError:
                    inpaint_editor = gr.ImageEditor(label="Кисть / ластик", type="numpy", height=420)
                btn_inpaint = gr.Button(
                    "✨ Обработать", variant="primary",
                    elem_classes=["primary-btn"], size="lg",
                )
                inpaint_out = gr.Image(label="Результат", height=360)
                inpaint_path = gr.Textbox(label="Файл", interactive=False)
                btn_inpaint.click(
                    run_inpaint, [inpaint_editor, inpaint_method, inpaint_radius],
                    [inpaint_out, inpaint_path],
                )

            # ===================== EDITOR =====================
            with gr.TabItem("✏️  Редактор"):
                with gr.Accordion("⚙️ Настройки редактора", open=False):
                    with gr.Row():
                        edit_steps = gr.Slider(10, 40, value=20, step=1, label="Шаги")
                        edit_img_g = gr.Slider(1.0, 2.5, value=1.5, step=0.1, label="Сохранить оригинал")
                        edit_cfg = gr.Slider(3, 12, value=7.5, step=0.5, label="Сила промпта")
                        edit_seed = gr.Number(value=0, label="Seed", precision=0)
                gr.Markdown("Промпт на английском: `add sunglasses`, `make smile`")
                with gr.Row():
                    edit_in = gr.Image(label="Фото", type="pil", height=250)
                    edit_out = gr.Image(label="Результат", height=250)
                edit_prompt = gr.Textbox(label="Инструкция", placeholder="add sunglasses")
                btn_edit = gr.Button("✏️ Применить", variant="primary", elem_classes=["primary-btn"], size="lg")
                edit_path = gr.Textbox(label="Файл", interactive=False)
                btn_edit.click(
                    run_edit,
                    [edit_in, edit_prompt, edit_steps, edit_img_g, edit_cfg, edit_seed],
                    [edit_out, edit_path],
                )

            # ===================== ANIMATE =====================
            with gr.TabItem("🎥  Оживить"):
                with gr.Accordion("⚙️ Настройки анимации", open=True):
                    anim_mode = gr.Radio(
                        ["Zoom in", "Zoom out", "Pan right", "Pan left", "Pulse",
                         "Нейро (SVD) — нужен GPU"],
                        value="Zoom in", label="Режим",
                    )
                    with gr.Row():
                        anim_sec = gr.Slider(1, 8, value=3, step=0.5, label="Сек (простая)")
                        anim_frames = gr.Slider(10, 25, value=14, step=1, label="Кадров SVD")
                        anim_motion = gr.Slider(20, 180, value=80, step=5, label="Движение SVD")
                        anim_seed = gr.Number(value=0, label="Seed", precision=0)
                with gr.Row():
                    anim_in = gr.Image(label="Фото", type="pil", height=230)
                    anim_out = gr.Video(label="Результат", height=230)
                anim_prompt = gr.Textbox(label="Описание (опц.)", lines=2)
                btn_anim = gr.Button("🎥 Оживить", variant="primary", elem_classes=["primary-btn"], size="lg")
                anim_path = gr.Textbox(label="Файл", interactive=False)
                btn_anim.click(
                    run_animate,
                    [anim_in, anim_mode, anim_prompt, anim_sec, anim_frames, anim_motion, anim_seed],
                    [anim_out, anim_path],
                )

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