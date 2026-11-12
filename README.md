# Local Face Swap (GPU)

Локальный face-swap с ускорением на NVIDIA.

- Автоматическая замена всех лиц
- Настройки скорости под GPU
- Фото + Видео

## Быстрый старт

```bash
cd local-faceswap
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python download_models.py
python app/main.py
```

Откроется http://127.0.0.1:7860

## Ускорение

В интерфейсе есть блок **«Настройки скорости (GPU)»**:

| Режим              | Скорость | Качество |
|--------------------|----------|----------|
| Быстрый (320)      | ★★★★★    | ★★★      |
| Баланс (480)       | ★★★★     | ★★★★     |
| Качественный (640) | ★★★      | ★★★★★    |

Для **видео** дополнительно:
- «Детектировать каждые N кадров» → поставь **2 или 3** — сильно ускоряет

## Важно для NVIDIA

Должен быть установлен именно:
```bash
pip install onnxruntime-gpu
```
Не обычный `onnxruntime`.

Проверить:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Должен быть `CUDAExecutionProvider`.

## Структура

```
local-faceswap/
├── app/
├── models/
├── outputs/
├── download_models.py
└── requirements.txt
```
