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

conda install -c conda-forge pycuda

OR


from rfdetr import RFDETRLarge

# Load the COCO-pretrained Large variant
model = RFDETRLarge()

# Export to TensorRT (FP16 is the default and fastest for your GPU)
engine_path = model.export(format="trt", output_dir="rfdetr_trt_model")

print(f"Engine saved to: {engine_path}")


