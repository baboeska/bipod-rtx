# nVidiabip
#Batch image processor for Python and nVidia RTX gpus
#For GPU's with 8gb of VRAM+
#Apache 2.0 license
#No guarantee, currently work in progress scratchpad.

#Log into WSL or Ubuntu or Debian
sudo apt update && sudo apt ugprade -y

#Activate your virtual environment

pip install "rfdetr[tensorrt]"

OR

conda install -c conda-forge pycuda

pip install "inference-models[trt10]"

from rfdetr import RFDETRLarge

# Load the COCO-pretrained Large variant
model = RFDETRLarge()

# Export to TensorRT (FP16 is the default and fastest for your GPU)
engine_path = model.export(format="trt", output_dir="rfdetr_trt_model")

print(f"Engine saved to: {engine_path}")


