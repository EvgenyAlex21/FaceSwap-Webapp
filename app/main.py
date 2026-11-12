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

DET_SIZES = {
    "Быстрый (320)": (320, 320),
    "Баланс (480)": (480, 480),
    "Качественный (640)": (640, 640),
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
    parts = str(text).replace(" ", "").split(",")
    return [int(p) for p in parts if p.isdigit()]


def _apply_speed(speed_mode: str):
    """Применяет размер детектора в зависимости от режима скорости."""
    size = DET_SIZES.get(speed_mode, (480, 480))
    swapper = get_swapper()
    swapper.set_det_size(size)
    return swapper


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_faces(img, speed_mode):
    if img is None:
        return None, "Загрузите фото"
    swapper = _apply_speed(speed_mode)
    bgr = _to_bgr(img)
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

def swap_photo(source_img, target_img, source_idx_str, swap_all, target_idx_str, speed_mode):
    if source_img is None or target_img is None:
        raise gr.Error("Загрузите оба изображения")

    swapper = _apply_speed(speed_mode)

    src_path = OUTPUTS / unique_name("tmp_src", ".jpg")
    tgt_path = OUTPUTS / unique_name("tmp_tgt", ".jpg")
    write_image(src_path, _to_bgr(source_img))
    write_image(tgt_path, _to_bgr(target_img))

    source_idx = (_parse_indices(source_idx_str) or [0])[0]
    target_indices = None if swap_all else (_parse_indices(target_idx_str) or [0])

    try:
        result_path = swapper.swap_image(
            source_path=src_path,
            target_path=tgt_path,
            source_face_index=source_idx,
            target_face_indices=target_indices,
            swap_all=swap_all,
        )
        return _to_pil(cv2.imread(str(result_path)))
    except Exception as e:
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def swap_video(source_img, target_video, source_idx_str, swap_all, target_idx_str,
               max_seconds, speed_mode, detect_every, progress=gr.Progress()):
    if source_img is None or target_video is None:
        raise gr.Error("Загрузите фото лица и видео")

    swapper = _apply_speed(speed_mode)

    src_path = OUTPUTS / unique_name("tmp_src", ".jpg")
    write_image(src_path, _to_bgr(source_img))

    source_idx = (_parse_indices(source_idx_str) or [0])[0]
    target_indices = None if swap_all else (_parse_indices(target_idx_str) or [0])

    try:
        cap = cv2.VideoCapture(target_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            detect_every_n=int(detect_every),
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
.status-text { font-size: 0.9rem !important; color: #a5b4fc !important; }
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
            <p>GPU-ускорение • Автозамена всех лиц • Фото и видео</p>
        </div>
        """)

        # ---- Общие настройки скорости ----
        with gr.Accordion("⚡ Настройки скорости (GPU)", open=True):
            speed_mode = gr.Radio(
                choices=list(DET_SIZES.keys()),
                value="Баланс (480)",
                label="Качество детекции лиц",
                info="Меньше разрешение = заметно быстрее. На NVIDIA разница очень ощутима.",
            )

        with gr.Tabs():

            # ==================== PHOTO ====================
            with gr.TabItem("🖼  Фото"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        gr.Markdown("### Лицо-источник")
                        src_photo = gr.Image(label="", type="pil", height=260, sources=["upload", "clipboard"])
                        btn_detect_src = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        src_preview = gr.Image(label="Обнаруженные лица", height=170, interactive=False)
                        src_status = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        src_idx = gr.Textbox(value="0", label="Индекс лица-источника", placeholder="0")

                    with gr.Column():
                        gr.Markdown("### Целевое фото")
                        tgt_photo = gr.Image(label="", type="pil", height=260, sources=["upload", "clipboard"])
                        btn_detect_tgt = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        tgt_preview = gr.Image(label="Обнаруженные лица", height=170, interactive=False)
                        tgt_status = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        swap_all_photo = gr.Checkbox(value=True, label="Заменить все лица автоматически")
                        tgt_idx = gr.Textbox(value="", label="Или конкретные индексы", placeholder="0,2")

                btn_swap_photo = gr.Button("✨  Заменить лицо", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_photo = gr.Image(label="Результат", height=400)

                btn_detect_src.click(fn=detect_faces, inputs=[src_photo, speed_mode], outputs=[src_preview, src_status])
                btn_detect_tgt.click(fn=detect_faces, inputs=[tgt_photo, speed_mode], outputs=[tgt_preview, tgt_status])
                btn_swap_photo.click(
                    fn=swap_photo,
                    inputs=[src_photo, tgt_photo, src_idx, swap_all_photo, tgt_idx, speed_mode],
                    outputs=[out_photo],
                )

            # ==================== VIDEO ====================
            with gr.TabItem("🎬  Видео"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        gr.Markdown("### Лицо-источник")
                        src_face_v = gr.Image(label="", type="pil", height=240, sources=["upload", "clipboard"])
                        btn_detect_src_v = gr.Button("Найти лица", elem_classes=["secondary-btn"], size="sm")
                        src_preview_v = gr.Image(label="Обнаруженные лица", height=150, interactive=False)
                        src_status_v = gr.Markdown("Загрузите фото", elem_classes=["status-text"])
                        src_idx_v = gr.Textbox(value="0", label="Индекс лица-источника", placeholder="0")

                    with gr.Column():
                        gr.Markdown("### Целевое видео")
                        tgt_video = gr.Video(label="", height=240)
                        swap_all_video = gr.Checkbox(value=True, label="Заменить все лица автоматически")
                        tgt_idx_v = gr.Textbox(value="", label="Или конкретные индексы", placeholder="0,1")
                        max_sec = gr.Slider(0, 180, value=20, step=1, label="Макс. длина (сек), 0 = всё")
                        detect_every = gr.Slider(
                            1, 5, value=2, step=1,
                            label="Детектировать лица каждые N кадров",
                            info="2–3 = хороший баланс скорости и качества. 1 = максимум точности",
                        )

                btn_swap_video = gr.Button("✨  Заменить лицо в видео", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_video = gr.Video(label="Результат")

                btn_detect_src_v.click(fn=detect_faces, inputs=[src_face_v, speed_mode], outputs=[src_preview_v, src_status_v])
                btn_swap_video.click(
                    fn=swap_video,
                    inputs=[src_face_v, tgt_video, src_idx_v, swap_all_video, tgt_idx_v, max_sec, speed_mode, detect_every],
                    outputs=[out_video],
                )

        gr.HTML("""
        <div class="tips">
            <strong>Ускорение под NVIDIA</strong><br>
            • Режим <strong>Быстрый (320)</strong> — максимальная скорость детекции.<br>
            • Режим <strong>Баланс (480)</strong> — рекомендуется для большинства случаев.<br>
            • Для видео ставьте «детектировать каждые 2–3 кадра» — это даёт самый большой прирост скорости.<br>
            • Убедитесь, что установлен <code>onnxruntime-gpu</code> (не обычный onnxruntime).
        </div>
        """)

    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("  Local Face Swap (GPU)")
    print("  http://127.0.0.1:7860")
    print("=" * 50)

    # Прогрев при старте
    try:
        get_swapper()
    except Exception as e:
        print(f"[warn] Модели пока не загружены: {e}")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
