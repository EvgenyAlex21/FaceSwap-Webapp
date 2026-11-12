#!/usr/bin/env python3
"""
Local Face Swap — современный GPU-интерфейс.
Автозамена лиц + улучшение качества + апскейл.
"""

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
from app.utils import OUTPUTS, unique_name, write_image


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


def _parse_indices(text: str) -> list:
    if not text or not str(text).strip():
        return []
    return [int(p) for p in str(text).replace(" ", "").split(",") if p.isdigit()]


def _apply_speed(speed_mode: str):
    size = DET_SIZES.get(speed_mode, (480, 480))
    swapper = get_swapper()
    swapper.set_det_size(size)
    return swapper


def detect_faces(img, speed_mode):
    if img is None:
        return None, "Загрузите фото"
    swapper = _apply_speed(speed_mode)
    preview, count = swapper.detect_faces_preview(_to_bgr(img))
    if count == 0:
        msg = "Лица не найдены"
    elif count == 1:
        msg = "Найдено **1** лицо"
    else:
        msg = f"Найдено лиц: **{count}** — все будут заменены автоматически"
    return _to_pil(preview), msg


def swap_photo(source_img, target_img, source_idx_str, swap_all, target_idx_str,
               speed_mode, enhance, upscale_mode, out_format):
    if source_img is None or target_img is None:
        raise gr.Error("Загрузите оба изображения")

    swapper = _apply_speed(speed_mode)
    src_path = OUTPUTS / unique_name("tmp_src", ".jpg")
    tgt_path = OUTPUTS / unique_name("tmp_tgt", ".jpg")
    write_image(src_path, _to_bgr(source_img))
    write_image(tgt_path, _to_bgr(target_img))

    source_idx = (_parse_indices(source_idx_str) or [0])[0]
    target_indices = None if swap_all else (_parse_indices(target_idx_str) or [0])
    upscale = UPSCALE_OPTS.get(upscale_mode, 1)

    try:
        result_path = swapper.swap_image(
            source_path=src_path,
            target_path=tgt_path,
            source_face_index=source_idx,
            target_face_indices=target_indices,
            swap_all=swap_all,
            enhance=enhance,
            upscale=upscale,
            out_format=out_format.lower(),
        )
        return _to_pil(cv2.imread(str(result_path))), str(result_path)
    except Exception as e:
        raise gr.Error(str(e))


def swap_video(source_img, target_video, source_idx_str, swap_all, target_idx_str,
               max_seconds, speed_mode, detect_every, enhance, upscale_mode,
               progress=gr.Progress()):
    if source_img is None or target_video is None:
        raise gr.Error("Загрузите фото лица и видео")

    swapper = _apply_speed(speed_mode)
    src_path = OUTPUTS / unique_name("tmp_src", ".jpg")
    write_image(src_path, _to_bgr(source_img))

    source_idx = (_parse_indices(source_idx_str) or [0])[0]
    target_indices = None if swap_all else (_parse_indices(target_idx_str) or [0])
    upscale = UPSCALE_OPTS.get(upscale_mode, 1)

    try:
        cap = cv2.VideoCapture(target_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_frames = min(int(max_seconds * fps), total) if max_seconds and max_seconds > 0 else None

        def prog(p):
            progress(p, desc="Обработка кадров...")

        result_path = swapper.swap_video(
            source_path=src_path,
            target_video_path=target_video,
            source_face_index=source_idx,
            target_face_indices=target_indices,
            swap_all=swap_all,
            max_frames=max_frames,
            detect_every_n=int(detect_every),
            enhance=enhance,
            upscale=upscale,
            progress_callback=prog,
        )
        return str(result_path)
    except Exception as e:
        raise gr.Error(str(e))


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 36px 28px 22px;
    background: linear-gradient(145deg, #0c0a1d 0%, #1a1440 40%, #0f172a 100%);
    border-radius: 20px;
    margin-bottom: 24px;
    border: 1px solid rgba(167, 139, 250, 0.15);
    box-shadow: 0 16px 48px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}
.main-header h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(105deg, #c4b5fd 0%, #67e8f9 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.03em;
}
.main-header p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

.primary-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(124, 58, 237, 0.5) !important;
}

.secondary-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

.tips {
    background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,27,75,0.4));
    border: 1px solid rgba(167, 139, 250, 0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 28px;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
}
.tips strong { color: #c4b5fd; }
.tips code {
    background: rgba(124, 58, 237, 0.15);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.84em;
    color: #a5b4fc;
}

.status-text { font-size: 0.88rem !important; color: #a5b4fc !important; }
footer { display: none !important; }

.settings-box {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 14px;
    padding: 4px 8px;
}
"""


def build_ui():
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#080b14",
        body_background_fill_dark="#080b14",
        block_background_fill="#0f1320",
        block_background_fill_dark="#0f1320",
        border_color_primary="#1e293b",
        block_border_width="1px",
        block_radius="14px",
        button_primary_background_fill="linear-gradient(135deg, #7c3aed, #4f46e5)",
        button_primary_background_fill_hover="linear-gradient(135deg, #6d28d9, #4338ca)",
        button_primary_text_color="white",
        input_background_fill="#0a0e1a",
    )

    with gr.Blocks(title="Local Face Swap") as demo:

        gr.HTML("""
        <div class="main-header">
            <h1>Local Face Swap</h1>
            <p>Автозамена лиц · Улучшение качества · Апскейл до Full HD · Полностью локально</p>
        </div>
        """)

        with gr.Accordion("⚡ Скорость и качество", open=True, elem_classes=["settings-box"]):
            with gr.Row():
                speed_mode = gr.Radio(
                    choices=list(DET_SIZES.keys()),
                    value="Баланс (480)",
                    label="Детекция лиц",
                    info="Меньше = быстрее",
                )
                upscale_mode = gr.Radio(
                    choices=list(UPSCALE_OPTS.keys()),
                    value="Без апскейла (1x)",
                    label="Апскейл результата",
                    info="2x ≈ HD, 4x ≈ Full HD+",
                )
                enhance = gr.Checkbox(
                    value=True,
                    label="Улучшить резкость и контраст",
                    info="Рекомендуется включать",
                )

        with gr.Tabs():

            # ========== PHOTO ==========
            with gr.TabItem("🖼  Фото"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Лицо-источник")
                        src_photo = gr.Image(label="", type="pil", height=250, sources=["upload", "clipboard"])
                        btn_detect_src = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        src_preview = gr.Image(label="Обнаруженные лица", height=160, interactive=False)
                        src_status = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        src_idx = gr.Textbox(value="0", label="Индекс лица-источника", placeholder="0")

                    with gr.Column():
                        gr.Markdown("#### Целевое фото")
                        tgt_photo = gr.Image(label="", type="pil", height=250, sources=["upload", "clipboard"])
                        btn_detect_tgt = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        tgt_preview = gr.Image(label="Обнаруженные лица", height=160, interactive=False)
                        tgt_status = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        swap_all_photo = gr.Checkbox(value=True, label="Заменить все лица автоматически")
                        tgt_idx = gr.Textbox(value="", label="Или конкретные индексы", placeholder="0,2")
                        out_format = gr.Radio(["JPG", "PNG"], value="JPG", label="Формат сохранения")

                btn_swap_photo = gr.Button("✨  Заменить лицо", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_photo = gr.Image(label="Результат", height=420)
                out_path_photo = gr.Textbox(label="Путь к файлу", interactive=False)

                btn_detect_src.click(fn=detect_faces, inputs=[src_photo, speed_mode], outputs=[src_preview, src_status])
                btn_detect_tgt.click(fn=detect_faces, inputs=[tgt_photo, speed_mode], outputs=[tgt_preview, tgt_status])
                btn_swap_photo.click(
                    fn=swap_photo,
                    inputs=[src_photo, tgt_photo, src_idx, swap_all_photo, tgt_idx,
                            speed_mode, enhance, upscale_mode, out_format],
                    outputs=[out_photo, out_path_photo],
                )

            # ========== VIDEO ==========
            with gr.TabItem("🎬  Видео"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Лицо-источник")
                        src_face_v = gr.Image(label="", type="pil", height=230, sources=["upload", "clipboard"])
                        btn_detect_src_v = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        src_preview_v = gr.Image(label="Обнаруженные лица", height=140, interactive=False)
                        src_status_v = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        src_idx_v = gr.Textbox(value="0", label="Индекс лица-источника", placeholder="0")

                    with gr.Column():
                        gr.Markdown("#### Целевое видео")
                        tgt_video = gr.Video(label="", height=230)
                        swap_all_video = gr.Checkbox(value=True, label="Заменить все лица автоматически")
                        tgt_idx_v = gr.Textbox(value="", label="Или конкретные индексы", placeholder="0,1")
                        max_sec = gr.Slider(0, 180, value=15, step=1, label="Макс. длина (сек), 0 = всё")
                        detect_every = gr.Slider(1, 5, value=2, step=1, label="Детекция каждые N кадров")

                btn_swap_video = gr.Button("✨  Заменить лицо в видео", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_video = gr.Video(label="Результат")

                btn_detect_src_v.click(fn=detect_faces, inputs=[src_face_v, speed_mode], outputs=[src_preview_v, src_status_v])
                btn_swap_video.click(
                    fn=swap_video,
                    inputs=[src_face_v, tgt_video, src_idx_v, swap_all_video, tgt_idx_v,
                            max_sec, speed_mode, detect_every, enhance, upscale_mode],
                    outputs=[out_video],
                )

        gr.HTML("""
        <div class="tips">
            <strong>Возможности</strong><br>
            • <strong>Автозамена всех лиц</strong> — система сама находит и меняет каждое лицо.<br>
            • <strong>Улучшение качества</strong> — резкость + контраст после замены.<br>
            • <strong>Апскейл</strong> — 2x (HD) или 4x (Full HD+).<br>
            • Фото сохраняются в <code>JPG</code> или <code>PNG</code>, видео — в <code>MP4</code>.<br>
            • Все файлы лежат в папке <code>outputs/</code>.
        </div>
        """)

    return demo, theme, CUSTOM_CSS


if __name__ == "__main__":
    print("=" * 50)
    print("  Local Face Swap")
    print("  http://127.0.0.1:7860")
    print("=" * 50)

    try:
        get_swapper()
    except Exception as e:
        print(f"[warn] Модели пока не загружены: {e}")

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
