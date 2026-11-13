#!/bin/bash
#
# Script Name: install_rfdetr_trt.sh
# Description: Sets up a specific environment for RFDETR, managing
#              PyCUDA, TensorRT, and specific PyTorch/CUDA versions.
#
# Note: This script assumes a Debian/Ubuntu-based Linux distribution.

set -e

# --- Configuration ---
VENV_NAME="rfdetr-env"
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
CUDA_URL="https://download.pytorch.org/whl/cu121"
MODEL_OUTPUT_DIR="rfdetr_trt_model"

echo "================================================="
echo "STARTING RFDETR/TENSORRT INSTALLATION"
echo "================================================="

# --- Step 1: System Update and PyCUDA Installation ---
echo "STEP 1: System Update and Install PyCUDA dependency..."
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pycuda python3-venv -y

# --- Step 2: Environment Setup ---
echo "STEP 2: Creating and activating Virtual Environment (${VENV_NAME})..."
cd ~

# Create the virtual environment (allow system packages for pycuda)
python3 -m venv --system-site-packages "${VENV_NAME}"
source "${VENV_NAME}/bin/activate"

# Upgrade pip
pip install --upgrade pip

# --- Step 3: Install Core Libraries and Dependencies ---
echo "STEP 3: Installing core packages and TensorRT dependencies..."
# Install standard packages
pip install tqdm opencv-python supervision

# Install specialized inference/tensorrt packages
pip install "inference-models[trt10]"
pip install "rfdetr[tensorrt]"

# --- Step 4: PyTorch/CUDA Version Management (Critical Step) ---
echo "STEP 4: Installing specific PyTorch/TorchVision versions for CUDA 12.1..."
# Uninstall potentially conflicting versions
pip uninstall -y torch torchvision

# Reinstall specific versions linked to CUDA 12.1
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url "${CUDA_URL}"
echo "PyTorch and TorchVision installed successfully."

# --- Step 5: Verification Checks ---
echo "STEP 5: Running system verification checks..."

echo "--- PyTorch and CUDA Check ---"
python -c "import torch; print(f'torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo "--- TensorRT Check ---"
python -c "import tensorrt as trt; print('TRT OK')"

echo "--- PyCUDA Check ---"
python -c "import pycuda.autoinit; print('pycuda OK')"
echo "Verification successful."

# --- Step 6: Model Export using RFDETR ---
echo "================================================="
echo "STEP 6: Exporting RFDETR Model to TensorRT format"
echo "================================================="

python - <<'EOF'
import os
from rfdetr import RFDETRLarge

print("Initializing RFDETRLarge model...")
model = RFDETRLarge()

# Export the model to the desired TensorRT format
print("Exporting model to TRT format...")
# Note: Hardcoded output dir to avoid Bash variable issues
engine_path = model.export(format="trt", output_dir="rfdetr_trt_model")

print("\n=================================================")
print("✅ FINAL SUCCESS: Engine Path:")
print(engine_path)
print("=================================================")
EOF

echo "================================================="
echo "INSTALLATION AND EXPORT COMPLETE!"
echo "================================================="
echo "The TensorRT engine is located in the directory: ${MODEL_OUTPUT_DIR}"
echo ""
echo "To start sorting images, run:"
echo "  source ~/${VENV_NAME}/bin/activate"
echo "  python bipod-rtx.py -i ~/Pictures -o ~/Pictures_Sorted -m ~/${MODEL_OUTPUT_DIR}/rfdetr-large_fp16.trt"
