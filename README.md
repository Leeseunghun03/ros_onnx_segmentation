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
- ONNX Runtime (Auto Installed)
- OpenCV 4.5 or later

## Usage

1. Clone the Repository into ROS 2 Workspace Navigate to your ROS 2 workspace's src directory and clone the repository:

```
git clone https://github.com/Leeseunghun03/ros_onnx_segmentation.git
```
   Build the workspace:
```
colcon build --symlink-install
```

2. Customize Parameters Adjust parameters such as class, input_topic, output_topic, and other settings in the provided configuration file or via command-line arguments.
3. Run the Camera Package Start your camera package to publish raw image data to a topic
   (e.g., /camera/image_raw).
4. Launch the Segmentation Package Run the segmentation package using the launch file:
```
ros2 launch ros_onnx_segmentation segmentation_launch.py
```

## Example (Custom model using Yolov8n-seg / without GPU)

![seg_s-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/e95fa76d-9802-4f28-9685-b906d511643d)

## Reference
https://github.com/winxos/yolov8_segment_onnx_in_cpp.git

https://github.com/ms-iot/ros_msft_onnx.git
