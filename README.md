# PCAL-Vision: A Physics- and Compute-Aware Lightweight Network for Real-Time Underwater Object Detection and Tracking 🌊

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-orange.svg)
![YOLOv8](https://img.shields.io/badge/Based_on-YOLOv8-yellow.svg)

This repository contains the official PyTorch implementation of **PCAL-Vision**, a customized and lightweight object detection model designed for Unmanned Underwater Vehicles (UUVs) / Autonomous Underwater Vehicles (AUVs) visual servoing and closed-loop control tasks. 

By modifying the underlying network structure, we optimized the model to balance detection accuracy in visually degraded underwater environments and real-time inference performance on compute-constrained edge devices (e.g., NVIDIA Jetson).

## 🎥 Real-world Underwater Experiments
Watch the demonstration of our PCAL-Vision system deployed on a custom AUV for real-time visual servoing (Fixed-point Hovering & Dynamic Tracking):

[![AUV Real-world Tracking](https://img.youtube.com/vi/IJa1e5b3HrI/0.jpg)](https://www.youtube.com/watch?v=IJa1e5b3HrI)
*(Click the image above to watch the video on YouTube)*

## 🌟 Key Highlights
- **Compute-Aware Lightweight Architecture (`C2f_Faster`)**: Deeply compressed specifically for underwater edge computing platforms. We replaced standard convolutions with Partial Convolutions (PConv) to reduce memory access costs (MAC) and overcome the memory wall on Jetson platforms, ensuring high FPS.
- **Physics-Aware Differentiable Enhancement (`DPE`)**: Injected a lightweight, differentiable physical imaging model (Jaffe-McGlamery) directly into the network front-end to achieve task-driven adaptive dehazing and color correction without heavy GAN pre-processing.
- **End-to-End Pipeline**: Provides a plug-and-play toolkit including scripts for training, ONNX/TensorRT exporting, and real-time inference tracking.

## 📝 Citation / Paper
*(Our related paper detailing the specific lightweight mechanisms and UUV control integration is currently under review for IEEE Transactions. The link and citation format will be updated here upon publication.)*

## 🛠️ Installation

⚠️ **Important Notice**: Since this project heavily modifies the underlying network structure of YOLOv8 to inject custom underwater modules (like DPE and FasterBlock), please **DO NOT** install the official `ultralytics` package via pip directly. 

Instead, clone this repository and install it locally:

```bash
# 1. Clone the repository
git clone https://github.com/bigflower296/PC-YOLO-Lightweight-Object-Detection-for-UUV-Underwater-Control.git
cd PC-YOLO-Lightweight-Object-Detection-for-UUV-Underwater-Control

# 2. Create a clean conda environment (Recommended)
conda create -n pcyolo python=3.9 -y
conda activate pcyolo

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the modified ultralytics package in editable mode
pip install -e .
