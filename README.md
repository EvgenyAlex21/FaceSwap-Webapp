# Local Face Swap

Локальная замена лиц на фото и видео.

## Возможности

- **Несколько исходных лиц** — лицо A → цель 0, лицо B → цель 1 и т.д.
- **Отрезок видео** — слайдеры «Начало / Конец» (сек)
- Автозамена всех лиц одним источником (галочка)
- Улучшение резкости + апскейл HD / Full HD
- Фото: JPG или PNG
- Результат всегда в папке `outputs/`

## Быстрый старт

```bash
cd local-faceswap
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
python download_models.py
python app/main.py
```

http://127.0.0.1:7860

## Несколько лиц

1. Снимите «Один источник на все лица»
2. Загрузите фото A / B / C
3. Укажите индекс лица на цели (кнопка «Найти лица»)
4. Пример: A→0, B→1

## Видео

- Выберите отрезок слайдерами
- «Детекция каждые N кадров»: 2–3 = быстрее, 1 = точнее
- Результат сохраняется в `outputs/` и показывается в превью

## GPU

Сейчас часто падает загрузка CUDA DLL (error 126) → работа на CPU (медленно).

Проверка:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Нужен рабочий `CUDAExecutionProvider` без error 126 при запуске.
При CPU 15 сек видео может занимать десятки минут — это нормально без GPU.

## Структура

```
local-faceswap/
├── app/
├── models/
├── outputs/
├── download_models.py
└── requirements.txt
```
