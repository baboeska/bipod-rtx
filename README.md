# BIPOD-RTX

**Batch image processor with Object Detection for NVIDIA RTX GPUs**

~60–80 images per second on an RTX 4060M (8GB) with RF-DETR Large + TensorRT.
Storage must be on WSL2's native Linux filesystem (not `/mnt/`) for full speed.

Part of [AI BIPOD](https://github.com/baboeska/bipod) — see the hub for the
OpenVINO, RKNN, and Raspberry Pi variants.

Optional resize and tile and both, with custom threshold potential.

Tiling 1000px+ image gives 1.45x+ accuracy, at 4x+ increase in time. May be worth it for security applications.
Default RF-DETR Large model works best on 704x704. Script has line for fine tuning for other models ie S, M, XL.

Example:

    python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted \
        -m ~/rfdetr_trt_model/rfdetr-large_fp16.trt --conf 0.40
    --mode resize (baseline), --mode tile --tile-threshold 0 (all tiled), --mode both --max-edge 4096


## Performance

| Platform | Model | Mode | Throughput |
|---|---|---|---|
| RTX 4060M (8GB) | RF-DETR Large + TensorRT | Resize | **60–80 img/s** |
| RTX 4060M (8GB) | RF-DETR Large + TensorRT | Tile | ~15-25 img/s |

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
    
    pip install opencv-python-headless

    pip install supervision

    


# Pin torch to CUDA 12.1 (avoids the CUDA 13 segfault)
    pip uninstall -y torch torchvision
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121


Verify the stack before continuing:

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



For a faster (slightly less accurate) model, substitute RFDETRMedium or RFDETRSmall, and edit script model line.


Usage

# Activate virtual environment; 

    source rfdetr-env/bin/activate

# Preview routing decisions, no files copied
    python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted --dry-run

# Actually sort
    python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted

# Point at a specific engine
    python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted \
    -m ~/rfdetr_trt_model/rfdetr-medium_fp16.trt

At the end of a run you get a class summary and a failures.csv log in the
output directory.



##Known pitfalls


Storage location matters enormously. Keep the image library on WSL2's
native filesystem, not a Windows-mounted drive (/mnt/c/, /mnt/d/, etc.).
On a USB drive mounted through Windows, throughput drops 10–16×.

Rebuild the .engine on the machine that will run it. TensorRT engines
are GPU-specific and TensorRT-version-specific.

Pin torch to 2.5.1+cu121. Letting pip resolve inference-models[trt10]
pulls CUDA 13 wheels, which segfault against CUDA 12 system libraries.

RF-DETR expects RGB, not BGR. The script converts with
cv2.cvtColor(img, cv2.COLOR_BGR2RGB) before every predict() call.
Skipping this quietly degrades detection quality.

Dry-run first. Especially on large libraries.

License

Apache 2.0, Do good things, share and share alike.

No guarantee, work in progress.

Made with help from DeepSeek, ChatGPT, and Google Search AI. :)
