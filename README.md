# Local Face Swap

Современный полностью локальный face-swap для **фото и видео**.

- Без лимитов и облака
- Автоматическое определение количества лиц
- Замена всех лиц одной кнопкой

## Главная особенность

Система **сама** определяет, сколько лиц на фото или в кадре видео, и заменяет их все автоматически.  
Ручной выбор индексов — опциональный.

## Быстрый старт

```bash
cd local-faceswap

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
python download_models.py         # один раз, ~350 МБ
python app/main.py
```

Откроется: **http://127.0.0.1:7860**

## Как пользоваться

1. Загрузите фото лица-источника
2. Загрузите целевое фото или видео
3. Нажмите **Заменить лицо**

По умолчанию заменяются **все** найденные лица.  
Если нужно только определённые — снимите галочку «Заменить все лица» и укажите индексы.

## Требования

- Python 3.10–3.12
- Рекомендуется NVIDIA GPU (от 6 GB VRAM)
- Можно на CPU (медленнее)

Если нет GPU:
```bash
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

## Структура

```
local-faceswap/
├── app/
│   ├── main.py
│   ├── swapper.py
│   └── utils.py
├── models/
├── outputs/
├── download_models.py
├── requirements.txt
└── README.md
```

Результаты сохраняются в `outputs/`.
