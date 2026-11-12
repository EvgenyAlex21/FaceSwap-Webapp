#!/usr/bin/env python3
"""Local Face Swap + Editor + Animation"""

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
from app.utils import OUTPUTS, unique_name, write_image, get_video_info


DET_SIZES = {
    "Быстрый (320)": (320, 320),
    "Баланс (480)": (480, 480),
    "Качественный (640)": (640, 640),
}
UPSCALE_OPTS = {
    "Без апскейла (1x)": 1,
    "HD (2x)": 2,
    "Full HD / 4x": 4,
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
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _save_tmp(img, prefix="tmp"):
    path = OUTPUTS / unique_name(prefix, ".jpg")
    write_image(path, _to_bgr(img))
    return path


def _apply_speed(speed_mode: str):
    size = DET_SIZES.get(speed_mode, (480, 480))
    sw = get_swapper()
    sw.set_det_size(size)
    return sw


def detect_faces(img, speed_mode):
    if img is None:
        return None, "Загрузите фото"
    sw = _apply_speed(speed_mode)
    preview, count = sw.detect_faces_preview(_to_bgr(img))
    if count == 0:
        msg = "Лица не найдены"
    elif count == 1:
        msg = "Найдено **1** лицо"
    else:
        msg = f"Найдено лиц: **{count}** (индексы 0…{count-1})"
    return _to_pil(preview), msg


def _build_source_map(src1, src2, src3, idx1, idx2, idx3, map1, map2, map3, single_mode):
    sources = []
    for img, s_idx in ((src1, idx1), (src2, idx2), (src3, idx3)):
        if img is not None:
            path = _save_tmp(img, "src")
            sources.append((path, int(s_idx) if str(s_idx).isdigit() else 0))
    if not sources:
        raise gr.Error("Загрузите хотя бы одно фото лица-источника")
    if single_mode:
        return {0: sources[0]}, True
    mapping = {}
    pairs = [
        (map1, sources[0] if len(sources) > 0 else None),
        (map2, sources[1] if len(sources) > 1 else None),
        (map3, sources[2] if len(sources) > 2 else None),
    ]
    for t_idx_str, src in pairs:
        if src is None:
            continue
        try:
            t_idx = int(str(t_idx_str).strip())
        except Exception:
            t_idx = 0
        mapping[t_idx] = src
    if not mapping:
        mapping[0] = sources[0]
    return mapping, False


def swap_photo(src1, src2, src3, idx1, idx2, idx3, map1, map2, map3,
               target_img, single_mode, speed_mode, enhance, upscale_mode, out_format,
               progress=gr.Progress(track_tqdm=False)):
    if target_img is None:
        raise gr.Error("Загрузите целевое фото")

    progress(0.05, desc="Подготовка...")
    sw = _apply_speed(speed_mode)
    source_map, swap_all = _build_source_map(
        src1, src2, src3, idx1, idx2, idx3, map1, map2, map3, single_mode
    )
    tgt_path = _save_tmp(target_img, "tgt")
    upscale = UPSCALE_OPTS.get(upscale_mode, 1)

    def cb(p, msg=""):
        progress(p, desc=msg or "Обработка...")

    try:
        result_path = sw.swap_image(
            source_map=source_map, target_path=tgt_path,
            swap_all_with_first=swap_all, enhance=enhance,
            upscale=upscale, out_format=out_format.lower(),
            progress_callback=cb,
        )
        progress(1.0, desc="Готово")
        return _to_pil(cv2.imread(str(result_path))), str(result_path)
    except Exception as e:
        raise gr.Error(str(e))


def on_video_upload(video):
    if video is None:
        return gr.update(maximum=60, value=0), gr.update(maximum=60, value=0), "Загрузите видео"
    try:
        _, _, fps, total, duration = get_video_info(video)
        dur = max(1.0, float(duration))
        return (
            gr.update(maximum=dur, value=0),
            gr.update(maximum=dur, value=min(dur, 15.0)),
            f"Длительность: **{dur:.1f}** сек · {total} кадров · {fps:.1f} fps",
        )
    except Exception as e:
        return gr.update(), gr.update(), f"Ошибка: {e}"


def swap_video(src1, src2, src3, idx1, idx2, idx3, map1, map2, map3,
               target_video, single_mode, start_sec, end_sec,
               speed_mode, detect_every, enhance, upscale_mode,
               progress=gr.Progress(track_tqdm=False)):
    if target_video is None:
        raise gr.Error("Загрузите видео")

    progress(0.02, desc="Подготовка...")
    sw = _apply_speed(speed_mode)
    source_map, swap_all = _build_source_map(
        src1, src2, src3, idx1, idx2, idx3, map1, map2, map3, single_mode
    )
    upscale = UPSCALE_OPTS.get(upscale_mode, 1)

    def cb(p, msg=""):
        progress(p, desc=msg or "Обработка видео...")

    try:
        result_path = sw.swap_video(
            source_map=source_map, target_video_path=target_video,
            swap_all_with_first=swap_all,
            start_sec=float(start_sec or 0), end_sec=float(end_sec or 0),
            detect_every_n=int(detect_every), enhance=enhance, upscale=upscale,
            progress_callback=cb,
        )
        progress(1.0, desc="Готово")
        return str(result_path), str(result_path)
    except Exception as e:
        raise gr.Error(str(e))


def run_edit(image, prompt, steps, image_guidance, guidance, seed,
             progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Загрузите фото")
    if not prompt or not str(prompt).strip():
        raise gr.Error("Введите инструкцию, например: add sunglasses")
    progress(0.1, desc="Загрузка модели / обработка...")
    try:
        from app.editor import edit_image
        path = edit_image(
            _to_bgr(image), prompt=str(prompt), steps=int(steps),
            image_guidance=float(image_guidance), guidance_scale=float(guidance),
            seed=int(seed),
        )
        progress(1.0, desc="Готово")
        return _to_pil(cv2.imread(str(path))), str(path)
    except Exception as e:
        raise gr.Error(str(e))


def run_animate(image, mode, prompt, seconds, svd_frames, motion, seed,
                progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Загрузите фото")
    bgr = _to_bgr(image)
    if prompt:
        print(f"[Animator] Промпт: {prompt}")

    def cb(p, msg=""):
        progress(p, desc=msg or "Анимация...")

    try:
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
        # Gradio лучше подхватывает абсолютный путь
        path = str(Path(path).resolve())
        return path, path
    except Exception as e:
        raise gr.Error(str(e))


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; font-family: 'Inter', system-ui, sans-serif !important; }
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
.tips {
    background: rgba(15,23,42,0.75); border: 1px solid rgba(167,139,250,0.12);
    border-radius: 14px; padding: 14px 18px; margin-top: 16px;
    color: #94a3b8; font-size: 0.86rem; line-height: 1.5;
}
.tips strong { color: #c4b5fd; }
.tips code { background: rgba(124,58,237,0.15); padding: 2px 6px; border-radius: 6px; color: #a5b4fc; }
.status-text { font-size: 0.86rem !important; color: #a5b4fc !important; }
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
            <p>Замена лиц · Редактор · Анимация · Индикатор прогресса</p>
        </div>
        """)

        with gr.Accordion("⚡ Скорость и качество", open=False):
            with gr.Row():
                speed_mode = gr.Radio(list(DET_SIZES.keys()), value="Баланс (480)", label="Детекция")
                upscale_mode = gr.Radio(list(UPSCALE_OPTS.keys()), value="Без апскейла (1x)", label="Апскейл")
                enhance = gr.Checkbox(value=True, label="Улучшить резкость")

        with gr.Tabs():
            with gr.TabItem("🖼  Фото"):
                gr.Markdown("#### Исходные лица (до 3)")
                with gr.Row():
                    with gr.Column():
                        src1 = gr.Image(label="Источник A", type="pil", height=160)
                        idx1 = gr.Number(value=0, label="Лицо на A", precision=0)
                        map1 = gr.Number(value=0, label="→ цель индекс", precision=0)
                    with gr.Column():
                        src2 = gr.Image(label="Источник B", type="pil", height=160)
                        idx2 = gr.Number(value=0, label="Лицо на B", precision=0)
                        map2 = gr.Number(value=1, label="→ цель индекс", precision=0)
                    with gr.Column():
                        src3 = gr.Image(label="Источник C", type="pil", height=160)
                        idx3 = gr.Number(value=0, label="Лицо на C", precision=0)
                        map3 = gr.Number(value=2, label="→ цель индекс", precision=0)
                single_mode_p = gr.Checkbox(value=True, label="Один источник на все лица цели")
                tgt_photo = gr.Image(label="Целевое фото", type="pil", height=220)
                with gr.Row():
                    btn_det_tgt = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                    out_format = gr.Radio(["JPG", "PNG"], value="JPG", label="Формат")
                tgt_preview = gr.Image(label="Лица на цели", height=130, interactive=False)
                tgt_status = gr.Markdown("", elem_classes=["status-text"])
                btn_swap_p = gr.Button("✨ Заменить лица", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_photo = gr.Image(label="Результат", height=360)
                out_path_p = gr.Textbox(label="Файл", interactive=False)
                btn_det_tgt.click(detect_faces, [tgt_photo, speed_mode], [tgt_preview, tgt_status])
                btn_swap_p.click(
                    swap_photo,
                    [src1, src2, src3, idx1, idx2, idx3, map1, map2, map3,
                     tgt_photo, single_mode_p, speed_mode, enhance, upscale_mode, out_format],
                    [out_photo, out_path_p],
                )

            with gr.TabItem("🎬  Видео"):
                gr.Markdown("#### Исходные лица (до 3)")
                with gr.Row():
                    with gr.Column():
                        vsrc1 = gr.Image(label="Источник A", type="pil", height=140)
                        vidx1 = gr.Number(value=0, label="Лицо на A", precision=0)
                        vmap1 = gr.Number(value=0, label="→ видео индекс", precision=0)
                    with gr.Column():
                        vsrc2 = gr.Image(label="Источник B", type="pil", height=140)
                        vidx2 = gr.Number(value=0, label="Лицо на B", precision=0)
                        vmap2 = gr.Number(value=1, label="→ видео индекс", precision=0)
                    with gr.Column():
                        vsrc3 = gr.Image(label="Источник C", type="pil", height=140)
                        vidx3 = gr.Number(value=0, label="Лицо на C", precision=0)
                        vmap3 = gr.Number(value=2, label="→ видео индекс", precision=0)
                single_mode_v = gr.Checkbox(value=True, label="Один источник на все лица")
                tgt_video = gr.Video(label="Видео", height=200)
                video_info = gr.Markdown("Загрузите видео", elem_classes=["status-text"])
                with gr.Row():
                    start_sec = gr.Slider(0, 60, value=0, step=0.5, label="Начало (сек)")
                    end_sec = gr.Slider(0, 60, value=15, step=0.5, label="Конец (сек)")
                detect_every = gr.Slider(1, 5, value=2, step=1, label="Детекция каждые N кадров")
                tgt_video.change(on_video_upload, [tgt_video], [start_sec, end_sec, video_info])
                btn_swap_v = gr.Button("✨ Заменить лица в видео", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_video = gr.Video(label="Результат")
                out_path_v = gr.Textbox(label="Файл", interactive=False)
                btn_swap_v.click(
                    swap_video,
                    [vsrc1, vsrc2, vsrc3, vidx1, vidx2, vidx3, vmap1, vmap2, vmap3,
                     tgt_video, single_mode_v, start_sec, end_sec,
                     speed_mode, detect_every, enhance, upscale_mode],
                    [out_video, out_path_v],
                )

            with gr.TabItem("✏️  Редактор"):
                gr.Markdown("Промпты на английском: `add sunglasses`, `make the person smile`")
                with gr.Row():
                    edit_in = gr.Image(label="Фото", type="pil", height=260)
                    edit_out = gr.Image(label="Результат", height=260)
                edit_prompt = gr.Textbox(label="Инструкция", placeholder="add sunglasses")
                with gr.Row():
                    edit_steps = gr.Slider(10, 40, value=20, step=1, label="Шаги")
                    edit_img_g = gr.Slider(1.0, 2.5, value=1.5, step=0.1, label="Сохранить оригинал")
                    edit_cfg = gr.Slider(3, 12, value=7.5, step=0.5, label="Сила промпта")
                    edit_seed = gr.Number(value=0, label="Seed", precision=0)
                btn_edit = gr.Button("✏️ Применить", variant="primary", elem_classes=["primary-btn"], size="lg")
                edit_path = gr.Textbox(label="Файл", interactive=False)
                btn_edit.click(run_edit, [edit_in, edit_prompt, edit_steps, edit_img_g, edit_cfg, edit_seed], [edit_out, edit_path])

            with gr.TabItem("🎥  Оживить фото"):
                gr.Markdown("**Zoom/Pan** — сразу на CPU. **SVD** — только с рабочим GPU.")
                with gr.Row():
                    anim_in = gr.Image(label="Фото", type="pil", height=240)
                    anim_out = gr.Video(label="Результат", height=240)
                anim_mode = gr.Radio(
                    ["Zoom in", "Zoom out", "Pan right", "Pan left", "Pulse", "Нейро (SVD) — нужен GPU"],
                    value="Zoom in", label="Режим",
                )
                anim_prompt = gr.Textbox(
                    label="Описание движения (опционально)",
                    placeholder="например: slow zoom to face, camera pans right...",
                    lines=2,
                )
                with gr.Row():
                    anim_sec = gr.Slider(1, 8, value=3, step=0.5, label="Сек (простая)")
                    anim_frames = gr.Slider(10, 25, value=14, step=1, label="Кадров SVD")
                    anim_motion = gr.Slider(20, 180, value=80, step=5, label="Движение SVD")
                    anim_seed = gr.Number(value=0, label="Seed", precision=0)
                btn_anim = gr.Button("🎥 Оживить", variant="primary", elem_classes=["primary-btn"], size="lg")
                anim_path = gr.Textbox(label="Файл", interactive=False)
                btn_anim.click(run_animate, [anim_in, anim_mode, anim_prompt, anim_sec, anim_frames, anim_motion, anim_seed], [anim_out, anim_path])

        gr.HTML("""
        <div class="tips">
            <strong>Важно</strong><br>
            • Пока CUDA DLL не грузится — face-swap идёт на <strong>CPU</strong> (медленно).<br>
            • Для видео используйте короткий отрезок и детекцию каждые 2–3 кадра.<br>
            • Прогресс показывается полоской над интерфейсом во время обработки.<br>
            • SVD на CPU недоступен — берите Zoom in / Pan.
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
