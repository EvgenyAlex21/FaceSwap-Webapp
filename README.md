# Local Face Swap

<p align="center">
  <b>Полностью локальная замена лиц на фото и видео</b><br>
  Несколько лиц · Сохранение звука · Пресеты качества · Опциональный AI-редактор и анимация
</p>

---

## Возможности

| Функция | Описание |
|--------|-------------|
| **Замена лиц на фото** | Одно или несколько исходных лиц на целевые |
| **Замена лиц на видео** | Покадровая замена с указанием диапазона кадров |
| **Звук** | Оригинальная звуковая дорожка сохраняется после обработки |
| **Настройки качества** | Размер детекции, сила улучшения, апскейл, CRF видео |
| **GPU / CPU** | Использует NVIDIA CUDA при наличии; иначе переключается на CPU с понятными логами |
| **Редактор** | Текстовые инструкции через InstructPix2Pix *(опционально)* |
| **Анимация** | Ken Burns (зум/панорамирование) или Stable Video Diffusion *(опционально)* |

Вся обработка выполняется **офлайн** после загрузки моделей. Нет облачных API, водяных знаков и лимитов использования.

---

## Быстрый старт

```bash
git clone <your-repo-url> local-faceswap
cd local-faceswap

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

## Рекомендуемые настройки качества

Для лучшего визуального результата (медленнее):

| Параметр | Значение |
|--------|--------|
| Детекция лиц | **Качественный (640)** или **Максимум (800)** |
| Резкость | **Среднее** или **Сильное** |
| Апскейл | **HD 2x**, если исходник маленький |
| Качество видео | **Высокое (CRF 18)** или **Хорошее (CRF 20)** |
| Детектировать каждые N кадров | **1** (каждый кадр) |
| Сохранять звук | **Вкл** |

Для скорости:

- Детекция **480** или **320**
- Детектировать каждые **2–3** кадра  
- CRF **23–26**

---

## Настройка NVIDIA GPU (Windows)

Замена лиц использует **onnxruntime-gpu**. Нужны CUDA 12 runtime + cuDNN 9:

```bash
pip install onnxruntime-gpu==1.20.2
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

Проверка:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python app/main.py
```

В консоли должно появиться:

```text
[GPU] CUDA: OK — face-swap на видеокарте
```

### Типичные проблемы с GPU (из реальных запусков)

**1. `LoadLibrary failed with error 126` / CUDA переключается на CPU**  
- Отсутствуют cuDNN или CUDA DLL.  
- Решение: переустановите `nvidia-cudnn-cu12` и связанные пакеты (см. выше).  
- **Не удаляйте** `nvidia-cudnn-cu12`, если нужен CUDA для ORT.

**2. `CUDA: OK`, но после переустановки torch CUDA снова ломается**  
- Часто конфликт порядка PATH / DLL между pip-пакетами `nvidia-*` и PyTorch.  
- Проект использует `os.add_dll_directory` и не засоряет `PATH`, чтобы ORT и Torch лучше уживались вместе.

**3. Torch: `Error loading cudnn_cnn64_9.dll` (WinError 127)**  
```bash
pip uninstall torch torchvision torchaudio nvidia-cudnn-cu12 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-cudnn-cu12
```
Затем проверьте:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**4. В видео нет звука**  
- Исправлено в текущей версии: звук копируется из исходника через ffmpeg (`imageio-ffmpeg`).  
- Оставляйте включённым чекбокс **«Сохранять звук»**.

**5. Превью в браузере пустое, файл скачивается нормально**  
- Браузеру нужен H.264 (`yuv420p`). Выход перекодируется для превью.  
- Файлы всегда лежат в `outputs/`.

**6. Ошибки OpenH264 / VideoWriter**  
- Writer использует `mp4v`, затем ffmpeg перепаковывает в H.264 — это обходит проблемы с OpenH264 DLL на Windows.

**7. 15-секундное видео обрабатывается ~40 минут**  
- Это был режим CPU. С рабочей CUDA должно быть значительно быстрее.  
- Для черновых прогонов используйте детекцию каждые 2–3 кадра.

**8. SVD / Редактор требуют PyTorch CUDA**  
- Это отдельно от onnxruntime. Установите Torch с CUDA (колесо cu124 выше).  
- На RTX 3050 4GB может не хватить VRAM для полного SVD — используйте режимы Zoom/Pan.

---

## Опционально: Редактор и нейросетевая анимация

```bash
pip install -r requirements-extra.txt
```

- **Редактор** — InstructPix2Pix (`add sunglasses`, `make the person smile`, …)  
- **Анимация → SVD** — нужна GPU + достаточно VRAM  

Простая анимация Zoom/Pan работает **без** дополнительных пакетов.

---

## Структура проекта

```text
local-faceswap/
├── app/
│   ├── main.py          # Gradio UI
│   ├── swapper.py       # Ядро замены лиц
│   ├── gpu_utils.py     # Детекция CUDA и fallback
│   ├── editor.py        # InstructPix2Pix (опционально)
│   ├── animator.py      # SVD / простая анимация
│   └── utils.py         # I/O, склейка звука, H.264
├── models/              # Скачанные веса (в .gitignore)
├── outputs/             # Результаты (в .gitignore)
├── download_models.py
├── requirements.txt
├── requirements-extra.txt
├── .gitignore
└── README.md
```

---

## Как работает сопоставление нескольких лиц

1. Снимите галочку **«Один источник на все лица»**  
2. Загрузите исходные фото A / B / C  
3. Нажмите **Найти лица** на целевом изображении, чтобы увидеть индексы  
4. Сопоставьте, например: A→0, B→1  

---

## Лицензия и модели

- Код приложения: свободно для личных и исследовательских проектов.  
- **InsightFace / InSwapper** и другие веса имеют собственные лицензии — проверяйте перед коммерческим использованием.  
- InstructPix2Pix и SVD: смотрите карточки моделей на Hugging Face.

---

## Чек-лист диагностики

```bash
# Провайдеры
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

# Torch CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Видна ли GPU?
nvidia-smi
```

Если CUDA в списке есть, но не активна — смотрите блок `[GPU] ---- ОТЛАДКА ----` при запуске: там объясняется причина сбоя.

---

Сделано для **локальной**, **приватной** и **неограниченной** замены лиц с практичным качеством и поддержкой звука.
