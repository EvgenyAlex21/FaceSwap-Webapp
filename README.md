# Local Face Swap

Полностью локальная замена лиц на фото и видео.

- Автоматическое определение всех лиц
- Улучшение резкости и контраста
- Апскейл до HD / Full HD
- Сохранение фото в JPG или PNG
- Работает на NVIDIA GPU и CPU

## Быстрый старт

```bash
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

Откроется: **http://127.0.0.1:7860**

## Возможности

| Функция | Описание |
|---------|----------|
| Автозамена лиц | Система сама находит все лица и заменяет их |
| Улучшение качества | Резкость + контраст после замены |
| Апскейл | 2x ≈ HD, 4x ≈ Full HD+ |
| Формат фото | JPG или PNG |
| Скорость | Режимы детекции 320 / 480 / 640 |

## GPU (NVIDIA)

Рекомендуемая связка:

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.20.1
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

Проверка:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Должен быть `CUDAExecutionProvider`.

Если CUDA не поднимается — программа продолжит работать на CPU.

## Структура

```
local-faceswap/
├── app/
│   ├── main.py       # Интерфейс
│   ├── swapper.py    # Ядро замены
│   └── utils.py
├── models/           # Модели (скачиваются)
├── outputs/          # Результаты
├── download_models.py
└── requirements.txt
```

Результаты сохраняются в `outputs/`.
