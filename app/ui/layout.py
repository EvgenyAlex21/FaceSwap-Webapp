"""Gradio presentation layer: layout, components, theme and visual styles.

Business logic and event handlers live in :mod:`app.main`; this module only
assembles the interface and wires UI events to those handlers.
"""
from pathlib import Path

import gradio as gr

_UI_DIR = Path(__file__).resolve().parent
CUSTOM_CSS = (_UI_DIR / "styles" / "app.css").read_text(encoding="utf-8")


def _const(value):
    return lambda: value


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


def build_ui(handlers):
    """Assemble the UI using callbacks/constants from the running app module."""
    globals().update({key: value for key, value in vars(handlers).items() if not key.startswith("__")})
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
                <p>Замена лиц, редактор фото по промпту, оживление снимков и умный инпейнт</p>
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
                        with gr.Row():
                            photo_quality = gr.Dropdown(
                                list(PHOTO_QUALITY.keys()),
                                value="Full HD (1920)",
                                label="Качество изображения",
                                info="Размер по длинной стороне. «Как есть» — без изменения размера",
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
                            "Загрузите видео, задайте начало и конец обрабатываемого фрагмента и нажмите «Найти лица в видео».",
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
                        start_sec = gr.Slider(0, 86400, value=0, step=0.1, label="Начало (сек)")
                        end_sec = gr.Slider(0, 86400, value=15, step=0.1, label="Конец (сек)")
                    range_display = gr.Markdown("Отрезок: **00:00:00 — 00:00:15**", elem_classes=["status-text"])

                    # --- СОПОСТАВЛЕНИЕ ---
                    gr.Markdown("### 3. Сопоставление лиц", elem_classes=["section-title"])
                    with gr.Row():
                        src_choice = gr.Dropdown(choices=[], label="Лицо из фото (Кого вставляем?)")
                        vid_choice = gr.Dropdown(choices=[], label="Лицо в видео (Кому вставляем?)")

                    tgt_video.change(
                        on_video_upload,
                        [tgt_video],
                        [start_sec, end_sec, video_info, range_display, vid_choice]
                    )

                    start_sec.change(_update_range_text, [start_sec, end_sec], range_display)
                    end_sec.change(_update_range_text, [start_sec, end_sec], range_display)

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
                        [tgt_video, vid_det, start_sec, end_sec],
                        [vid_status, *vid_crops_only, *vid_grps_only, vid_count, vid_choice],
                        show_progress="full",
                        show_progress_on=[tgt_video],
                    )

                    for slider in [start_sec, end_sec]:
                        slider.change(
                            reset_vid_faces,
                            None,
                            [vid_status, *vid_crops_only, *vid_grps_only, vid_count, vid_choice],
                            show_progress="hidden",
                            trigger_mode="always_last",
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

            # ===================== РЕДАКТОР =====================
            with gr.TabItem("Редактор"):
                with gr.Column(elem_classes=["tab-container"]):
                    gr.HTML(_guide(
                        "Как это работает",
                        "Нейросеть меняет фото по вашему текстовому описанию: цвет волос, очки, глаза, фон, погода, стиль.",
                        [
                            "Загрузите фото — человек, животное, природа, что угодно.",
                            "Опишите правку одной фразой на русском или английском, например «добавь очки».",
                            "Нажмите «Применить правки». Результат сохранится в папке <strong>outputs/</strong>.",
                            "Не устроило — измените формулировку или seed. Кнопка «Использовать результат как исходное фото» "
                            "позволяет вносить правки по цепочке.",
                        ],
                        "Основная модель — FLUX.2 klein 4B (Apache 2.0): при первом запуске скачивается ≈12 ГБ. "
                        "На видеокарте с 4 ГБ работает через квантование GGUF и выгрузку на CPU: ожидайте 1–3 минуты на правку. "
                        "Лучше всего модель понимает конкретные короткие команды. Если видеопамяти не хватает, размер "
                        "автоматически уменьшается.",
                    ))

                    with gr.Accordion("Параметры обработки", open=True):
                        with gr.Row():
                            ed_model = gr.Dropdown(
                                list(EDITOR_MODELS.keys()), value=list(EDITOR_MODELS.keys())[0],
                                label="Модель",
                                info="FLUX.2 klein — точнее и качественнее; InstructPix2Pix — быстрее и легче",
                            )
                            ed_size = gr.Dropdown(
                                [512, 640, 768, 896, 1024], value=640,
                                label="Рабочий размер (px)",
                                info="По длинной стороне. Для 4 ГБ VRAM оптимально 512–768",
                            )
                        with gr.Row():
                            ed_steps = gr.Slider(2, 8, value=4, step=1, label="Шаги",
                                                 info="FLUX.2 klein дистиллирована: 4 шага — оптимум")
                            ed_imgguid = gr.Slider(1.0, 2.5, value=1.5, step=0.1, visible=False,
                                                   label="Сохранение оригинала",
                                                   info="Выше — результат ближе к исходному фото")
                            ed_seed = gr.Number(value=-1, precision=0, label="Seed", info="-1 — случайный")
                        with gr.Row():
                            ed_quant = gr.Dropdown(
                                KLEIN_QUANTS, value="Q4_K_M", label="Квантование FLUX",
                                info="Q3 — меньше VRAM, Q5 — чуть точнее",
                            )
                            ed_translate = gr.Checkbox(value=True, label="Переводить русский промпт на английский")
                            ed_restore = gr.Checkbox(value=False, label="Растянуть результат до размера оригинала")

                    gr.Markdown("### 1. Исходное фото", elem_classes=["section-title"])
                    ed_in = gr.Image(label="Исходное фото", type="pil", height=360)

                    gr.Markdown("### 2. Что изменить", elem_classes=["section-title"])
                    gr.Markdown("Короткая и конкретная команда работает лучше длинного описания.", elem_classes=["hint"])
                    ed_prompt = gr.Textbox(
                        label="Промпт", lines=2,
                        placeholder="Например: измени цвет волос на рыжий и добавь очки",
                    )
                    with gr.Row():
                        ed_ex_btns = [gr.Button(ex, size="sm", elem_classes=["secondary-btn"]) for ex in EDITOR_EXAMPLES]

                    gr.Markdown("### 3. Результат", elem_classes=["section-title"])
                    ed_btn = gr.Button("Применить правки", variant="primary",
                                       elem_classes=["primary-btn"], size="lg")
                    ed_out = gr.Image(label="Результат", height=420)
                    ed_info = gr.Markdown("", elem_classes=["status-text"])
                    ed_path = gr.Textbox(label="Файл", interactive=False)
                    ed_chain = gr.Button("Использовать результат как исходное фото", elem_classes=["secondary-btn"])

                    ed_model.change(editor_model_changed, [ed_model], [ed_steps, ed_imgguid, ed_size])

                    for btn, ex in zip(ed_ex_btns, EDITOR_EXAMPLES):
                        btn.click(_const(ex), None, ed_prompt)

                    ed_btn.click(
                        run_editor,
                        [ed_in, ed_prompt, ed_model, ed_size, ed_steps, ed_imgguid,
                         ed_seed, ed_quant, ed_translate, ed_restore],
                        [ed_out, ed_path, ed_info],
                        show_progress="full",
                        show_progress_on=[ed_out],
                    )
                    ed_chain.click(lambda img: img, [ed_out], [ed_in])

            # ===================== ПРЕОБРАЖЕНИЕ =====================
            with gr.TabItem("Преображение"):
                with gr.Column(elem_classes=["tab-container"]):
                    gr.HTML(_guide(
                        "Как это работает",
                        "Нейросеть оживляет статичное фото: добавляет движение, мимику, движение камеры — по текстовому описанию.",
                        [
                            "Загрузите фото — портрет, животное, пейзаж.",
                            "Опишите сцену: что происходит, как двигается камера.",
                            "Выберите разрешение и длительность клипа.",
                            "Нажмите «Оживить фото». Результат сохранится в папке <strong>outputs/</strong>.",
                        ],
                        "Модель — LTX-Video 0.9.5 (2B параметров). При первом запуске скачивается 15–25 ГБ. "
                        "На 4 ГБ VRAM работает с выгрузкой слоёв на CPU: ожидайте 3–10 минут на клип. "
                        "Разрешение и длина клипа подстраиваются под свободную видеопамять автоматически.",
                    ))

                    with gr.Accordion("Параметры обработки", open=True):
                        with gr.Row():
                            an_resolution = gr.Dropdown(
                                list(ANIM_RESOLUTIONS.keys()), value="512p (512 px)",
                                label="Качество (высота)",
                                info="Чем выше, тем детальнее и медленнее. На 4 ГБ VRAM оптимум — 512",
                            )
                            an_duration = gr.Dropdown(
                                [f"{d} сек" for d in ANIM_DURATIONS], value="2 сек",
                                label="Длина клипа",
                                info="На 4 ГБ VRAM реалистично 2–4 секунды",
                            )
                        with gr.Row():
                            an_steps = gr.Slider(10, 50, value=30, step=1, label="Шаги")
                            an_guidance = gr.Slider(1.0, 7.0, value=3.0, step=0.1, label="Guidance")
                            an_seed = gr.Number(value=-1, precision=0, label="Seed", info="-1 — случайный")
                        with gr.Row():
                            an_upscale = gr.Dropdown(
                                list(ANIM_UPSCALES.keys()), value="Как есть",
                                label="Увеличение результата",
                                info="Простое масштабирование готового видео (Lanczos)",
                            )
                        with gr.Row():
                            an_translate = gr.Checkbox(value=True, label="Переводить промпт на английский")
                            an_quality = gr.Checkbox(value=True, label="Добавлять quality-суффикс")
                            an_codec = gr.Checkbox(value=True, label="JPEG-подготовка кадра (рекомендуется)")

                    gr.Markdown("### 1. Исходное фото", elem_classes=["section-title"])
                    an_in = gr.Image(label="Фото", type="pil", height=360)

                    gr.Markdown("### 2. Что должно происходить", elem_classes=["section-title"])
                    an_prompt = gr.Textbox(
                        label="Описание сцены", lines=2,
                        placeholder="Например: человек улыбается, волосы колышутся на ветру",
                    )
                    with gr.Row():
                        an_ex_btns = [gr.Button(ex, size="sm", elem_classes=["secondary-btn"]) for ex in ANIMATE_EXAMPLES]

                    with gr.Accordion("Negative prompt", open=False):
                        an_negative = gr.Textbox(
                            label="Negative prompt", value=ANIM_NEGATIVE, lines=3,
                        )

                    gr.Markdown("### 3. Результат", elem_classes=["section-title"])
                    an_btn = gr.Button("Оживить фото", variant="primary",
                                       elem_classes=["primary-btn"], size="lg")
                    an_out = gr.Video(label="Результат")
                    an_info = gr.Markdown("", elem_classes=["status-text"])
                    an_path = gr.Textbox(label="Файл", interactive=False)

                    for btn, ex in zip(an_ex_btns, ANIMATE_EXAMPLES):
                        btn.click(_const(ex), None, an_prompt)

                    an_btn.click(
                        run_animate,
                        [an_in, an_prompt, an_negative, an_resolution, an_duration,
                         an_steps, an_guidance, an_seed,
                         an_translate, an_quality, an_codec, an_upscale],
                        [an_out, an_path, an_info],
                        show_progress="full", show_progress_on=[an_out],
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

