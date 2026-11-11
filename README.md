# BIPOD-RTX

**Batch image processor for NVIDIA RTX GPUs**

~60–80 images per second on an RTX 4060M (8GB) with RF-DETR Large + TensorRT.
Storage must be on WSL2's native Linux filesystem (not `/mnt/`) for full speed.

Part of [AI BIPOD](https://github.com/baboeska/bipod) — see the hub for the
OpenVINO, RKNN, and Raspberry Pi variants.

## Performance

| Platform | Model | Throughput |
|---|---|---|
| RTX 4060M (8GB) | RF-DETR Large + TensorRT | **60–80 img/s** |

## Requirements

- NVIDIA GPU with CUDA 12.x support
- WSL2 (Ubuntu/Debian) or native Linux
- Python 3.10+
- `nvidia-smi` should show your GPU before you start

## Setup

# System packages

sudo apt update && sudo apt upgrade -y

sudo apt install python3-pycuda

# Python environment

cd ~

python3 -m venv rfdetr-env

source rfdetr-env/bin/activate

pip install --upgrade pip

# Core dependencies

pip install tqdm opencv-python

pip install "inference-models[trt10]"

pip install "rfdetr[tensorrt]"

pip install opencv-tools


# Pin torch to CUDA 12.1 (avoids the CUDA 13 segfault)
pip uninstall -y torch torchvision
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121


Verify the stack before continuing:
bash

  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

  python -c "import tensorrt as trt; trt.Builder(trt.Logger(trt.Logger.WARNING)); print('TRT OK')"

  python -c "import pycuda.autoinit; print('pycuda OK')"

Expected: 2.5.1+cu121 True, TRT OK, pycuda OK.


Build the TensorRT engine (one-time)

The engine is compiled for your specific GPU. Rebuild it if you change GPUs.



cd ~

python - <<'EOF'

from rfdetr import RFDETRLarge

model = RFDETRLarge()

engine_path = model.export(format="trt", output_dir="rfdetr_trt_model")

print("Engine:", engine_path)

EOF



For a faster (slightly less accurate) model, substitute RFDETRMedium or RFDETRSmall.


Usage
bash

# Preview routing decisions, no files copied
python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted --dry-run

# Actually sort
python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted

# Point at a specific engine
python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted \
    -m ~/rfdetr_trt_model/rfdetr-medium_fp16.trt

At the end of a run you get a class summary and a failures.csv log in the
output directory.
Known pitfalls

    Storage location matters enormously. Keep the image library on WSL2's
    native filesystem, not a Windows-mounted drive (/mnt/c/, /mnt/d/, etc.).
    On a USB drive mounted through Windows, throughput drops 10–16×.

    Rebuild the .engine on the machine that will run it. TensorRT engines
    are GPU-specific and TensorRT-version-specific.

    Pin torch to 2.5.1+cu121. Letting pip resolve inference-models[trt10]
    pulls CUDA 13 wheels, which segfault against CUDA 12 system libraries.

    Dry-run first. Especially on large libraries.

License

Apache 2.0

No guarantee, work in progress.

Made with help from DeepSeek, ChatGPT, and Google Search AI. :)
