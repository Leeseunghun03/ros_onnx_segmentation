# ROS 2 Real-Time Image Segmentation with ONNX Runtime

## Overview
This ROS 2 package performs real-time image segmentation using the [ONNX Runtime](https://onnxruntime.ai/). Designed for robotics applications, it processes camera input and generates segmented output in real-time.

---

## Features
- Real-time image segmentation using ONNX Runtime.
- Compatible with ROS 2 (tested with Humble).
- Supports Yolo ONNX segmentation model.
- Configurable parameters for input, output topics, and model path.

---

## Requirements
- ROS 2 Humble
- ONNX Runtime (`onnxruntime` Python package)
- OpenCV (`opencv-python` package)

To install dependencies:
```bash
pip install onnxruntime opencv-python
