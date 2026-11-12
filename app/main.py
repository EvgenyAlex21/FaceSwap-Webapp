#!/usr/bin/env python3
"""
Local Face Swap
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bgr(img) -> np.ndarray:
    if img is None:
        return None
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _parse_indices(text: str) -> list:
    if not text or not str(text).strip():
        return []
    parts = str(text).replace(" ", "").split(",")
    indices = []
    for p in parts:
        if p.isdigit():
            indices.append(int(p))
    return indices


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_faces(img):
    if img is None:
        return None, "Загрузите фото"
    bgr = _to_bgr(img)
    swapper = get_swapper()
    preview, count = swapper.detect_faces_preview(bgr)
    if count == 0:
        msg = "Лица не найдены"
    elif count == 1:
        msg = "Найдено **1** лицо"
    else:
        msg = f"Найдено лиц: **{count}** — все будут заменены автоматически"
    return _to_pil(preview), msg


# ---------------------------------------------------------------------------
# Photo
# ---------------------------------------------------------------------------

def swap_photo(source_img, target_img, source_idx_str, swap_all, target_idx_str):
    if source_img is None or target_img is None:
        raise gr.Error("Загрузите оба изображения")

    source_bgr = _to_bgr(source_img)
    target_bgr = _to_bgr(target_img)

    src_path = OUTPUTS / unique_name("tmp_src", ".jpg")
    tgt_path = OUTPUTS / unique_name("tmp_tgt", ".jpg")
    write_image(src_path, source_bgr)
    write_image(tgt_path, target_bgr)

    source_indices = _parse_indices(source_idx_str)
    source_idx = source_indices[0] if source_indices else 0

    target_indices = None if swap_all else _parse_indices(target_idx_str)
    if not swap_all and not target_indices:
        target_indices = [0]

    try:
        swapper = get_swapper()
        result_path = swapper.swap_image(
            source_path=src_path,
            target_path=tgt_path,
            source_face_index=source_idx,
            target_face_indices=target_indices,
            swap_all=swap_all,
        )
        result_bgr = cv2.imread(str(result_path))
        return _to_pil(result_bgr)
    except Exception as e:
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def swap_video(source_img, target_video, source_idx_str, swap_all, target_idx_str, max_seconds, progress=gr.Progress()):
    if source_img is None or target_video is None:
        raise gr.Error("Загрузите фото лица и видео")

    source_bgr = _to_bgr(source_img)
    src_path = OUTPUTS / unique_name("tmp_src", ".jpg")
    write_image(src_path, source_bgr)

    source_indices = _parse_indices(source_idx_str)
    source_idx = source_indices[0] if source_indices else 0

    target_indices = None if swap_all else _parse_indices(target_idx_str)
    if not swap_all and not target_indices:
        target_indices = [0]

    try:
        swapper = get_swapper()

        import cv2 as cv
        cap = cv.VideoCapture(target_video)
        fps = cap.get(cv.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_frames = None
        if max_seconds and max_seconds > 0:
            max_frames = min(int(max_seconds * fps), total)

        def prog(p):
            progress(p, desc="Обработка кадров...")

        result_path = swapper.swap_video(
            source_path=src_path,
            target_video_path=target_video,
            source_face_index=source_idx,
            target_face_indices=target_indices,
            swap_all=swap_all,
            max_frames=max_frames,
            progress_callback=prog,
        )
        return str(result_path)
    except Exception as e:
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 32px 24px 18px 24px;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
    border-radius: 18px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.main-header h1 {
    font-size: 2.15rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94a3b8 !important;
    font-size: 0.98rem !important;
    margin: 0 !important;
}

.primary-btn {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.25) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(124, 58, 237, 0.4) !important;
}

.secondary-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

.tips {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 24px;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.55;
}
.tips strong { color: #c4b5fd; }
.tips code {
    background: rgba(255,255,255,0.08);
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.85em;
    color: #a5b4fc;
}

.status-text {
    font-size: 0.9rem !important;
    color: #a5b4fc !important;
}

footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(
        title="Local Face Swap",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0b0f19",
            body_background_fill_dark="#0b0f19",
            block_background_fill="#131722",
            block_background_fill_dark="#131722",
            border_color_primary="#1e293b",
            block_border_width="1px",
            block_radius="12px",
            button_primary_background_fill="linear-gradient(135deg, #7c3aed, #4f46e5)",
            button_primary_background_fill_hover="linear-gradient(135deg, #6d28d9, #4338ca)",
            button_primary_text_color="white",
            input_background_fill="#0f172a",
        ),
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML("""
        <div class="main-header">
            <h1>Local Face Swap</h1>
            <p>Автоматическая замена лиц • Фото и видео • Полностью локально</p>
        </div>
        """)

        with gr.Tabs():

            # ==================== PHOTO ====================
            with gr.TabItem("🖼  Фото"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### Лицо-источник")
                        gr.Markdown("Чьё лицо берём", elem_classes=["status-text"])
                        src_photo = gr.Image(label="", type="pil", height=270, sources=["upload", "clipboard"])
                        btn_detect_src = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        src_preview = gr.Image(label="Обнаруженные лица", height=180, interactive=False)
                        src_status = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        src_idx = gr.Textbox(
                            value="0",
                            label="Индекс лица-источника (если лиц несколько)",
                            placeholder="0",
                            info="По умолчанию берётся самое крупное лицо (0)",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Целевое фото")
                        gr.Markdown("Куда вставляем", elem_classes=["status-text"])
                        tgt_photo = gr.Image(label="", type="pil", height=270, sources=["upload", "clipboard"])
                        btn_detect_tgt = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        tgt_preview = gr.Image(label="Обнаруженные лица", height=180, interactive=False)
                        tgt_status = gr.Markdown("Загрузите фото", elem_classes=["status-text"])

                        swap_all_photo = gr.Checkbox(
                            value=True,
                            label="Заменить все лица автоматически",
                            info="Включено — система сама найдёт и заменит все лица на фото",
                        )
                        tgt_idx = gr.Textbox(
                            value="",
                            label="Или укажите конкретные индексы",
                            placeholder="Например: 0,2",
                            info="Работает только если галочка выше снята",
                            visible=True,
                        )

                btn_swap_photo = gr.Button(
                    "✨  Заменить лицо",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    size="lg",
                )
                out_photo = gr.Image(label="Результат", height=400)

                btn_detect_src.click(fn=detect_faces, inputs=[src_photo], outputs=[src_preview, src_status])
                btn_detect_tgt.click(fn=detect_faces, inputs=[tgt_photo], outputs=[tgt_preview, tgt_status])
                btn_swap_photo.click(
                    fn=swap_photo,
                    inputs=[src_photo, tgt_photo, src_idx, swap_all_photo, tgt_idx],
                    outputs=[out_photo],
                )

            # ==================== VIDEO ====================
            with gr.TabItem("🎬  Видео"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### Лицо-источник")
                        src_face_v = gr.Image(label="", type="pil", height=250, sources=["upload", "clipboard"])
                        btn_detect_src_v = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        src_preview_v = gr.Image(label="Обнаруженные лица", height=160, interactive=False)
                        src_status_v = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        src_idx_v = gr.Textbox(
                            value="0",
                            label="Индекс лица-источника",
                            placeholder="0",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Целевое видео")
                        tgt_video = gr.Video(label="", height=250)

                        swap_all_video = gr.Checkbox(
                            value=True,
                            label="Заменить все лица автоматически",
                            info="На каждом кадре система сама найдёт и заменит все лица",
                        )
                        tgt_idx_v = gr.Textbox(
                            value="",
                            label="Или укажите конкретные индексы",
                            placeholder="Например: 0,1",
                            info="Работает только если галочка выше снята",
                        )
                        max_sec = gr.Slider(
                            minimum=0,
                            maximum=180,
                            value=20,
                            step=1,
                            label="Максимальная длина обработки (сек)",
                            info="0 = обработать всё видео",
                        )

                btn_swap_video = gr.Button(
                    "✨  Заменить лицо в видео",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    size="lg",
                )
                out_video = gr.Video(label="Результат")

                btn_detect_src_v.click(fn=detect_faces, inputs=[src_face_v], outputs=[src_preview_v, src_status_v])
                btn_swap_video.click(
                    fn=swap_video,
                    inputs=[src_face_v, tgt_video, src_idx_v, swap_all_video, tgt_idx_v, max_sec],
                    outputs=[out_video],
                )

        gr.HTML("""
        <div class="tips">
            <strong>Как это работает</strong><br>
            • По умолчанию включена <strong>автоматическая замена всех лиц</strong> — система сама определяет, сколько лиц на фото/видео, и заменяет каждое.<br>
            • Если нужно заменить только конкретные лица — снимите галочку и укажите индексы (например <code>0,2</code>).<br>
            • Кнопка «Найти лица» показывает номера лиц на фото.<br>
            • Лучшее качество дают чёткие фронтальные фото.<br>
            • Результаты сохраняются в папку <code>outputs/</code>.
        </div>
        """)

    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("  Local Face Swap")
    print("  http://127.0.0.1:7860")
    print("=" * 50)

    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
