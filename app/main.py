#!/usr/bin/env python3
"""
Local Face Swap — мульти-лица, отрезок видео, апскейл.
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
    """
    Собирает {target_idx: (path, source_face_idx)}.
    single_mode: один источник на все лица цели.
    """
    sources = []
    for img, s_idx in ((src1, idx1), (src2, idx2), (src3, idx3)):
        if img is not None:
            path = _save_tmp(img, "src")
            sources.append((path, int(s_idx) if str(s_idx).isdigit() else 0))

    if not sources:
        raise gr.Error("Загрузите хотя бы одно фото лица-источника")

    if single_mode:
        # Все целевые лица → первый источник
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
        # fallback: первый источник → лицо 0
        mapping[0] = sources[0]

    return mapping, False


def swap_photo(src1, src2, src3, idx1, idx2, idx3, map1, map2, map3,
               target_img, single_mode, speed_mode, enhance, upscale_mode, out_format):
    if target_img is None:
        raise gr.Error("Загрузите целевое фото")

    sw = _apply_speed(speed_mode)
    source_map, swap_all = _build_source_map(
        src1, src2, src3, idx1, idx2, idx3, map1, map2, map3, single_mode
    )
    tgt_path = _save_tmp(target_img, "tgt")
    upscale = UPSCALE_OPTS.get(upscale_mode, 1)

    try:
        result_path = sw.swap_image(
            source_map=source_map,
            target_path=tgt_path,
            swap_all_with_first=swap_all,
            enhance=enhance,
            upscale=upscale,
            out_format=out_format.lower(),
        )
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
        return gr.update(), gr.update(), f"Ошибка чтения видео: {e}"


def swap_video(src1, src2, src3, idx1, idx2, idx3, map1, map2, map3,
               target_video, single_mode, start_sec, end_sec,
               speed_mode, detect_every, enhance, upscale_mode,
               progress=gr.Progress()):
    if target_video is None:
        raise gr.Error("Загрузите видео")

    sw = _apply_speed(speed_mode)
    source_map, swap_all = _build_source_map(
        src1, src2, src3, idx1, idx2, idx3, map1, map2, map3, single_mode
    )
    upscale = UPSCALE_OPTS.get(upscale_mode, 1)

    try:
        def prog(p):
            progress(p, desc="Обработка кадров...")

        result_path = sw.swap_video(
            source_map=source_map,
            target_video_path=target_video,
            swap_all_with_first=swap_all,
            start_sec=float(start_sec or 0),
            end_sec=float(end_sec or 0),
            detect_every_n=int(detect_every),
            enhance=enhance,
            upscale=upscale,
            progress_callback=prog,
        )
        # Возвращаем путь из outputs — Gradio сможет открыть
        return str(result_path)
    except Exception as e:
        raise gr.Error(str(e))


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; font-family: 'Inter', system-ui, sans-serif !important; }
.main-header {
    text-align: center; padding: 32px 24px 18px;
    background: linear-gradient(145deg, #0c0a1d 0%, #1a1440 40%, #0f172a 100%);
    border-radius: 18px; margin-bottom: 20px;
    border: 1px solid rgba(167,139,250,0.15);
    box-shadow: 0 16px 48px rgba(0,0,0,0.4);
}
.main-header h1 {
    font-size: 2.1rem !important; font-weight: 700 !important;
    background: linear-gradient(105deg, #c4b5fd, #67e8f9, #6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0 !important;
}
.main-header p { color: #94a3b8 !important; font-size: 0.92rem !important; margin: 0 !important; }
.primary-btn {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
    border-radius: 12px !important; padding: 14px 28px !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.35) !important;
}
.primary-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 32px rgba(124,58,237,0.5) !important; }
.secondary-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important; border-radius: 10px !important;
}
.tips {
    background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,27,75,0.35));
    border: 1px solid rgba(167,139,250,0.12); border-radius: 14px;
    padding: 16px 20px; margin-top: 20px; color: #94a3b8; font-size: 0.88rem; line-height: 1.55;
}
.tips strong { color: #c4b5fd; }
.tips code { background: rgba(124,58,237,0.15); padding: 2px 7px; border-radius: 6px; color: #a5b4fc; font-size: 0.84em; }
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
            <p>Несколько лиц · Отрезок видео · Апскейл · Полностью локально</p>
        </div>
        """)

        with gr.Accordion("⚡ Скорость и качество", open=True):
            with gr.Row():
                speed_mode = gr.Radio(list(DET_SIZES.keys()), value="Баланс (480)", label="Детекция")
                upscale_mode = gr.Radio(list(UPSCALE_OPTS.keys()), value="Без апскейла (1x)", label="Апскейл")
                enhance = gr.Checkbox(value=True, label="Улучшить резкость")

        with gr.Tabs():
            # ================= PHOTO =================
            with gr.TabItem("🖼  Фото"):
                gr.Markdown("#### Исходные лица (до 3 фото)")
                gr.Markdown("Загрузите разные фото, если нужно подставить **разные** лица на цель.", elem_classes=["status-text"])
                with gr.Row():
                    with gr.Column():
                        src1 = gr.Image(label="Источник A", type="pil", height=180)
                        idx1 = gr.Number(value=0, label="Лицо на A", precision=0)
                        map1 = gr.Number(value=0, label="→ на цель индекс", precision=0)
                    with gr.Column():
                        src2 = gr.Image(label="Источник B (опц.)", type="pil", height=180)
                        idx2 = gr.Number(value=0, label="Лицо на B", precision=0)
                        map2 = gr.Number(value=1, label="→ на цель индекс", precision=0)
                    with gr.Column():
                        src3 = gr.Image(label="Источник C (опц.)", type="pil", height=180)
                        idx3 = gr.Number(value=0, label="Лицо на C", precision=0)
                        map3 = gr.Number(value=2, label="→ на цель индекс", precision=0)

                single_mode_p = gr.Checkbox(
                    value=True,
                    label="Один источник на все лица цели (если нужен только один человек)",
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Целевое фото")
                        tgt_photo = gr.Image(label="", type="pil", height=240)
                        btn_det_tgt = gr.Button("Найти лица на цели", elem_classes=["secondary-btn"], size="sm")
                        tgt_preview = gr.Image(label="Лица на цели", height=160, interactive=False)
                        tgt_status = gr.Markdown("", elem_classes=["status-text"])
                        out_format = gr.Radio(["JPG", "PNG"], value="JPG", label="Формат")

                btn_swap_p = gr.Button("✨ Заменить лица", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_photo = gr.Image(label="Результат", height=400)
                out_path_p = gr.Textbox(label="Файл", interactive=False)

                btn_det_tgt.click(detect_faces, [tgt_photo, speed_mode], [tgt_preview, tgt_status])
                btn_swap_p.click(
                    swap_photo,
                    [src1, src2, src3, idx1, idx2, idx3, map1, map2, map3,
                     tgt_photo, single_mode_p, speed_mode, enhance, upscale_mode, out_format],
                    [out_photo, out_path_p],
                )

            # ================= VIDEO =================
            with gr.TabItem("🎬  Видео"):
                gr.Markdown("#### Исходные лица (до 3 фото)")
                with gr.Row():
                    with gr.Column():
                        vsrc1 = gr.Image(label="Источник A", type="pil", height=160)
                        vidx1 = gr.Number(value=0, label="Лицо на A", precision=0)
                        vmap1 = gr.Number(value=0, label="→ на видео индекс", precision=0)
                    with gr.Column():
                        vsrc2 = gr.Image(label="Источник B (опц.)", type="pil", height=160)
                        vidx2 = gr.Number(value=0, label="Лицо на B", precision=0)
                        vmap2 = gr.Number(value=1, label="→ на видео индекс", precision=0)
                    with gr.Column():
                        vsrc3 = gr.Image(label="Источник C (опц.)", type="pil", height=160)
                        vidx3 = gr.Number(value=0, label="Лицо на C", precision=0)
                        vmap3 = gr.Number(value=2, label="→ на видео индекс", precision=0)

                single_mode_v = gr.Checkbox(value=True, label="Один источник на все лица в кадре")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Видео")
                        tgt_video = gr.Video(label="", height=220)
                        video_info = gr.Markdown("Загрузите видео", elem_classes=["status-text"])
                        with gr.Row():
                            start_sec = gr.Slider(0, 60, value=0, step=0.5, label="Начало (сек)")
                            end_sec = gr.Slider(0, 60, value=15, step=0.5, label="Конец (сек)")
                        detect_every = gr.Slider(1, 5, value=2, step=1,
                            label="Детекция каждые N кадров (2–3 быстрее, 1 точнее)")

                tgt_video.change(on_video_upload, [tgt_video], [start_sec, end_sec, video_info])

                btn_swap_v = gr.Button("✨ Заменить лица в видео", variant="primary", elem_classes=["primary-btn"], size="lg")
                out_video = gr.Video(label="Результат")
                out_path_v = gr.Textbox(label="Файл сохранён в", interactive=False)

                def _swap_v_wrap(*args):
                    path = swap_video(*args)
                    return path, path

                btn_swap_v.click(
                    _swap_v_wrap,
                    [vsrc1, vsrc2, vsrc3, vidx1, vidx2, vidx3, vmap1, vmap2, vmap3,
                     tgt_video, single_mode_v, start_sec, end_sec,
                     speed_mode, detect_every, enhance, upscale_mode],
                    [out_video, out_path_v],
                )

            # ================= SOON =================
            with gr.TabItem("✨  Редактор / Анимация"):
                gr.Markdown("""
### Скоро

Здесь будет:
- изменение атрибутов (очки, волосы, телосложение) по текстовому описанию
- оживление фото (image-to-video) по промпту

Это **отдельные тяжёлые модели** (Stable Diffusion / AnimateDiff и т.п.),  
их нужно ставить дополнительно (~5–15 GB).  
Могу добавить в следующем обновлении, если нужно.
                """)

        gr.HTML("""
        <div class="tips">
            <strong>Как назначить разные лица</strong><br>
            1. Снимите галочку «Один источник на все лица».<br>
            2. Загрузите фото A, B, C.<br>
            3. Укажите: «→ на цель индекс» = номер лица на целевом фото/видео (смотрите «Найти лица»).<br>
            4. Пример: A→0, B→1 — лицо 0 заменится на A, лицо 1 на B.<br><br>
            <strong>Видео</strong> — выберите отрезок слайдерами «Начало / Конец». Результат пишется в <code>outputs/</code>.
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
        print(f"[warn] {e}")

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
