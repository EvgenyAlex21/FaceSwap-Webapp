# Local Face Swap

<p align="center">
  <b>Fully local face swap for photos &amp; videos</b><br>
  Multi-face · Audio preserved · Quality presets · Optional AI editor &amp; animation
</p>

---

## Features

| Feature | Description |
|--------|-------------|
| **Photo face swap** | One or multiple source faces mapped to target faces |
| **Video face swap** | Frame-by-frame swap with start/end range |
| **Audio** | Original soundtrack is merged back after processing |
| **Quality controls** | Detection size, enhance strength, upscale, video CRF |
| **GPU / CPU** | Uses NVIDIA CUDA when available; falls back to CPU with clear logs |
| **Editor** | Text instructions via InstructPix2Pix *(optional)* |
| **Animate** | Ken Burns zoom/pan or Stable Video Diffusion *(optional)* |

All processing runs **offline** after models are downloaded. No cloud API, no watermarks, no usage limits.

---

## Quick start

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

Open **http://127.0.0.1:7860**

---

## Recommended quality settings

For best visual results (slower):

| Setting | Value |
|--------|--------|
| Face detection | **Качественный (640)** or **Максимум (800)** |
| Sharpening | **Среднее** or **Сильное** |
| Upscale | **HD 2x** if source is small |
| Video quality | **Высокое (CRF 18)** or **Хорошее (CRF 20)** |
| Detect every N frames | **1** (every frame) |
| Keep audio | **On** |

For speed:

- Detection **480** or **320**
- Detect every **2–3** frames  
- CRF **23–26**

---

## NVIDIA GPU setup (Windows)

Face-swap uses **onnxruntime-gpu**. You need CUDA 12 runtime + cuDNN 9:

```bash
pip install onnxruntime-gpu==1.20.2
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

Check:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python app/main.py
```

Console should show:

```text
[GPU] CUDA: OK — face-swap на видеокарте
```

### Common GPU issues (from real runs)

**1. `LoadLibrary failed with error 126` / CUDA falls back to CPU**  
- Missing cuDNN or CUDA DLLs.  
- Fix: reinstall `nvidia-cudnn-cu12` and related packages (see above).  
- Do **not** remove `nvidia-cudnn-cu12` if you need ORT CUDA.

**2. `CUDA: OK` then after reinstalling torch → CUDA broken again**  
- Often PATH / DLL order conflict between pip `nvidia-*` and PyTorch.  
- This project uses `os.add_dll_directory` and avoids polluting `PATH` so ORT and Torch can coexist better.

**3. Torch: `Error loading cudnn_cnn64_9.dll` (WinError 127)**  
```bash
pip uninstall torch torchvision torchaudio nvidia-cudnn-cu12 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-cudnn-cu12
```
Then verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**4. Video has no sound**  
- Fixed in current version: audio is copied from the source with ffmpeg (`imageio-ffmpeg`).  
- Keep the **«Сохранять звук»** checkbox enabled.

**5. Preview empty in browser, file downloads fine**  
- Browser needs H.264 (`yuv420p`). Output is re-encoded for preview.  
- Files are always under `outputs/`.

**6. OpenH264 / VideoWriter errors**  
- Writer uses `mp4v`, then ffmpeg remuxes to H.264 — avoids OpenH264 DLL issues on Windows.

**7. 15s video takes ~40 minutes**  
- That was CPU mode. With working CUDA it should be much faster.  
- Also use detect every 2–3 frames for draft runs.

**8. SVD / Editor need PyTorch CUDA**  
- Separate from onnxruntime. Install Torch with CUDA (cu124 wheel above).  
- RTX 3050 4GB may OOM on full SVD — use Zoom/Pan modes instead.

---

## Optional: Editor & neural animation

```bash
pip install -r requirements-extra.txt
```

- **Editor** — InstructPix2Pix (`add sunglasses`, `make the person smile`, …)  
- **Animate → SVD** — needs GPU + enough VRAM  

Simple Zoom/Pan animation works **without** extra packages.

---

## Project layout

```text
local-faceswap/
├── app/
│   ├── main.py          # Gradio UI
│   ├── swapper.py       # Face swap core
│   ├── gpu_utils.py     # CUDA detection & fallback
│   ├── editor.py        # InstructPix2Pix (optional)
│   ├── animator.py      # SVD / simple motion
│   └── utils.py         # I/O, audio merge, H.264
├── models/              # Downloaded weights (gitignored)
├── outputs/             # Results (gitignored)
├── download_models.py
├── requirements.txt
├── requirements-extra.txt
├── .gitignore
└── README.md
```

---

## How multi-face mapping works

1. Uncheck **«Один источник на все лица»**  
2. Upload source photos A / B / C  
3. Click **Найти лица** on the target to see indices  
4. Map e.g. A→0, B→1  

---

## License & models

- App code: use freely for personal / research projects.  
- **InsightFace / InSwapper** and other weights have their own licenses — check before commercial use.  
- InstructPix2Pix & SVD: see model cards on Hugging Face.

---

## Troubleshooting checklist

```bash
# Providers
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

# Torch CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# GPU visible?
nvidia-smi
```

If CUDA is listed but inactive, read the `[GPU] ---- ОТЛАДКА ----` block printed at startup — it explains the failure reason.

---

Built for **local**, **private**, **unlimited** face swapping with practical quality and audio support.
