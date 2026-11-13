# Local Face Swap

<p align="center">
  <b>Полностью локальная замена лиц на фото и видео</b><br>
  Слоты лиц · Сохранение звука · Пресеты качества · Выделение
</p>

---

## Возможности

| Функция | Описание |
|---------|----------|
| **Замена лиц на фото** | Автоматический поиск лиц на целевом фото и замена каждого из них лицом с отдельного снимка |
| **Замена лиц на видео** | Быстрое сканирование видео, выбор лица-источника и лица-цели, обработка с сохранением звука |
| **AI-инпейнт («Выделение»)** | Перерисовка области, закрашенной кистью, с помощью диффузионных моделей |
| **Звук** | Оригинальная звуковая дорожка видео сохраняется |
| **Настройки качества** | Детекция, резкость, разрешение выхода, FPS, CRF для видео |
| **GPU / CPU** | Автоматическая проверка NVIDIA CUDA для `onnxruntime` с подробным логированием; fallback на CPU |

Вся обработка выполняется **офлайн** после загрузки моделей. Нет облачных API, водяных знаков и лимитов использования.

---

## Быстрый старт

```bash
git clone <your-repo-url> FaceSwap-Webapp
cd FaceSwap-Webapp

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
python download_models.py
python app/main.py
```

Откройте **http://127.0.0.1:7860**

---

## Вкладка «Выделение» (AI-инпейнт)

Для работы диффузии требуются дополнительные библиотеки и скачивание моделей (~4–6 ГБ):

```bash
pip install diffusers transformers accelerate torch
```

Скрипт `download_models.py` автоматически скачает три модели в папку `models/`:

- **Realistic Vision (SD 1.5)** — универсальная модель по умолчанию (влезает в 4 GB VRAM).
- **TryOffAnyone (SD 1.5)** — специализированная модель без использования текстового промпта.
- **LUSTIFY SDXL** — тяжёлая модель для высокого разрешения (требует много VRAM).

---

## Рекомендуемые настройки качества

Для лучшего визуального результата (медленнее):

| Параметр | Значение |
|----------|----------|
| Детекция лиц | **Качественный (640)** или **Максимум (800)** |
| Резкость | **Среднее** или **Сильное** |
| Разрешение видео | **Как есть** или **Full HD (1080p)** |
| Качество видео | **Высокое (CRF 18)** или **Хорошее (CRF 20)** |
| Детекция кадров | **1** (каждый кадр) |

Для быстрого черновика:

- Детекция **320–480**
- Каждые **2–3** кадра
- CRF **23–26**

---

## Настройка NVIDIA GPU (Windows)

Замена лиц (`onnxruntime-gpu`) и диффузия (`torch`) требуют CUDA 12 runtime + cuDNN 9:

```bash
pip install onnxruntime-gpu==1.20.2
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

Для диффузии (PyTorch) обязательно нужна версия с CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Проверка:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import torch; print(torch.cuda.is_available())"
```

В консоли при запуске должно появиться:

```text
[GPU] CUDA: OK — face-swap на видеокарте
```

### Типичные проблемы

**1. `LoadLibrary failed with error 126` / CUDA переключается на CPU**
- Отсутствуют cuDNN или CUDA DLL. Переустановите `nvidia-cudnn-cu12` (см. выше). **Не удаляйте** этот пакет, если нужен CUDA для ORT.

**2. Torch: `Error loading cudnn_cnn64_9.dll` (WinError 127)**

```bash
pip uninstall torch torchvision torchaudio nvidia-cudnn-cu12 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-cudnn-cu12
```

**3. В видео нет звука**
- Оставляйте включённым чекбокс **«Сохранять звук»**. Используется `imageio-ffmpeg`.

**4. Превью в браузере пустое**
- Браузеру нужен H.264 (`yuv420p`). Выход перекодируется через ffmpeg. Файлы всегда лежат в `outputs/`.

**5. «Выделение» падает с ошибкой `OutOfMemoryError`**
- Модель LUSTIFY SDXL требует много VRAM. Для видеокарт с 4 GB используйте Realistic Vision (SD 1.5).

---

## Структура проекта

```text
FaceSwap-Webapp/
├── app/
│   ├── main.py          # Gradio UI (Фото, Видео, Выделение)
│   ├── swapper.py       # Ядро замены лиц (детекция, замена, видео)
│   ├── inpaint.py       # AI-инпейнт
│   ├── gpu_utils.py     # Детекция CUDA и логирование
│   └── utils.py         # I/O, склейка звука, H.264, апскейл
├── models/              # Веса моделей (в .gitignore)
├── outputs/             # Результаты (в .gitignore)
├── download_models.py   # Скрипт загрузки buffalo_l, inswapper, LUSTIFY, TryOffAnyone и др.
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Как работает сопоставление лиц в видео

1. Загрузите фото с лицом-источником и нажмите **«Найти лица на фото»**.
2. Загрузите видео и нажмите **«Найти лица в видео (быстро)»** (сканирует кадры и находит уникальных людей).
3. В разделе **«Сопоставление лиц»** выберите, чьё лицо вставляем (из фото) и кому (в видео).
4. Задайте начало и конец фрагмента ползунками и нажмите **«Заменить лица в видео»**.

---

## Лицензия и модели

- Код приложения свободен для личных и исследовательских проектов.
- **InsightFace / InSwapper**, **Realistic Vision**, **LUSTIFY**, **TryOffAnyone** — имеют собственные лицензии на Hugging Face. Проверяйте перед коммерческим использованием.

---

## Чек-лист диагностики

```bash
# Провайдеры ONNX
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

# Torch CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Видна ли GPU?
nvidia-smi
```

Если CUDA в списке есть, но не активна — смотрите блок `[GPU] ---- ОТЛАДКА ----` в логах при запуске.

---

Сделано для **локальной**, **приватной** и **неограниченной** замены лиц с практичным качеством и поддержкой звука.