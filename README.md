# PC-YOLO: Lightweight Object Detection for UUV Underwater Control 🌊

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)](https://github.com/ultralytics/ultralytics)

This repository contains **PC-YOLO**, a customized and lightweight object detection model based on the YOLOv8 architecture, specifically designed for **Unmanned Underwater Vehicles (UUVs)** and **underwater control tasks**.

By modifying the underlying network structure, we optimized the model to balance detection accuracy and real-time performance on compute-constrained underwater edge devices.

## 🌟 Key Highlights
- **Lightweight Architecture for UUVs:** Deeply compressed and optimized specifically for underwater PC/edge computing platforms with limited computational resources.
- **Customized Network Modules:** Modified the core `ultralytics/nn/modules` to integrate custom feature extraction mechanisms tailored for underwater environments (addressing issues like low visibility and color distortion).
- **End-to-End Pipeline:** Provides a plug-and-play toolkit including scripts for training, ONNX exporting, and real-time inference tracking.

## 📝 Citation / Paper
*(Our related paper detailing the specific lightweight mechanisms and UUV control integration is currently under review/in press. The link and citation format will be updated here upon publication.)*

## 🛠️ Installation

**⚠️ Important Notice:**
Since this project modifies the underlying network structure of YOLOv8 to inject custom underwater modules, **please DO NOT install the official `ultralytics` package via pip directly**.

Instead, clone this repository and install it locally:

```bash
# 1. Clone the repository
git clone https://github.com/bigflower296/PC-YOLO.git
cd PC-YOLO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the modified ultralytics package in editable mode
pip install -e .