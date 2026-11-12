# Local Face Swap

Локальная замена лиц + редактор по тексту + оживление фото.

## Возможности

- Замена лиц на фото и видео (несколько источников)
- Выбор отрезка видео
- Редактор по текстовой инструкции (InstructPix2Pix)
- Оживление фото: простая анимация или нейро (SVD)
- Апскейл, улучшение резкости

## Установка

```bash
cd local-faceswap
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python download_models.py
python app/main.py
```

### Редактор и нейро-анимация (опционально)

```bash
pip install -r requirements-extra.txt
```

При первом использовании модели скачаются сами (~5–10 GB).

## Вкладки

| Вкладка | Что делает |
|---------|------------|
| Фото | Замена лиц, до 3 источников |
| Видео | Замена на отрезке, превью в outputs/ |
| Редактор | `add sunglasses`, `make smile` и т.п. |
| Оживить фото | Zoom/Pan сразу; SVD при наличии GPU |

## GPU

Для скорости face-swap и для SVD нужна рабочая CUDA.
Если `CUDAExecutionProvider` не грузится (error 126) — всё идёт на CPU.

## Структура

```
local-faceswap/
├── app/
│   ├── main.py
│   ├── swapper.py
│   ├── editor.py
│   ├── animator.py
│   └── utils.py
├── models/
├── outputs/
├── requirements.txt
├── requirements-extra.txt
└── download_models.py
```
