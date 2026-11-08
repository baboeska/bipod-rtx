# nVidiabip
#Batch image processor for Python and nVidia RTX gpus
#For GPU's with 8gb of VRAM+
#Apache 2.0 license
#No guarantee, currently work in progress scratchpad.

#Log into WSL or Ubuntu or Debian

sudo apt update && sudo apt ugprade -y

sudo apt install python3-pycuda

#Activate your virtual environment

cd ~

python3 -m venv rfdetr-env

source rfdetr-env/bin/activate

pip install --upgrade pip

pip install "inference-models[trt10]

pip install "rfdetr[tensorrt]"

.

pip uninstall -y torch torchvision
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

OR


from rfdetr import RFDETRLarge

# Load the COCO-pretrained Large variant
model = RFDETRLarge()

# Export to TensorRT
engine_path = model.export(format="trt", output_dir="rfdetr_trt_model")

print(f"Engine saved to: {engine_path}")

python - <<'EOF'

engine_path = model.export(format="trt", output_dir="rfdetr_trt_model")
print("Engine:", engine_path)
EOF
