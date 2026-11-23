# ros_onnx_segmentation

[![build](https://github.com/Leeseunghun03/ros_onnx_segmentation/actions/workflows/build.yml/badge.svg)](https://github.com/Leeseunghun03/ros_onnx_segmentation/actions/workflows/build.yml)

Instance segmentation for ROS 2 in a single C++ node. Runs YOLOv8 / YOLO11 segmentation models exported to ONNX, on CPU, CUDA or TensorRT.

- MIT licensed. At runtime the node uses only ONNX Runtime (MIT) and OpenCV, so no copyleft code ends up on the robot.
- Masks are published for other nodes to use: `vision_msgs/Detection2DArray` plus a per-pixel instance ID image, aligned with the input image.
- Runs as a standalone node or as a component. Loaded next to the camera driver with intra-process communication, images are handed over without a copy.
- One package for x64 and aarch64. ONNX Runtime is fetched at build time; CUDA and TensorRT are a CMake option away, and an existing install (e.g. on Jetson) can be used instead.
- About 700 lines of C++. Easy to read, easy to change for your own model.

![demo](docs/demo.gif)

yolo11s-seg with the CUDA provider. Video: [Pro dog walker, Tokyo](https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm) by Nesnad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The GIF is released under the same license.

Tested on ROS 2 Humble (x64, CPU / CUDA / TensorRT). CI also builds Humble aarch64, the GPU variant and Jazzy.

## Performance

Time from publishing a 960x540 image to receiving `detections`, including pre- and post-processing and mask decoding. Mean over 100 frames.

| Model | CPU (Ryzen 9 5950X, 4 threads) | CUDA (RTX 3090) | TensorRT FP16 (RTX 3090) |
|---|---|---|---|
| yolo11n-seg | 62 ms | 15 ms | 8.3 ms |
| yolo11s-seg | 134 ms | 18 ms | 12.4 ms |

With TensorRT the first start builds an engine, which takes a few minutes depending on the model. It is cached in `trt_cache_dir` and reused afterwards.

## Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/Leeseunghun03/ros_onnx_segmentation.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build
```

This downloads the CPU build of ONNX Runtime 1.26.0 for x64 or aarch64.

### GPU

```bash
colcon build --cmake-args -DONNXRUNTIME_GPU=ON
```

Uses the CUDA build of ONNX Runtime (Linux x64). Requires CUDA 12 and cuDNN 9 at runtime, plus TensorRT 10 for the `tensorrt` provider. Select the provider with the `execution_provider` parameter. If it cannot be loaded, the node logs a warning and falls back to CPU.

### Jetson or a custom ONNX Runtime

Point the build at an ONNX Runtime install that has the providers you need:

```bash
colcon build --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

The directory needs `include/` (with `onnxruntime_cxx_api.h`) and `lib/libonnxruntime.so`.

## Model

Export an Ultralytics segmentation model once, on any machine:

```bash
pip install ultralytics
yolo export model=yolo11n-seg.pt format=onnx imgsz=640
```

Class names, input size and mask layout are read from the model, so no extra config is needed. Note that the pretrained Ultralytics weights are AGPL-3.0; the license of this package does not change that.

## Run

```bash
ros2 launch ros_onnx_segmentation segmentation_launch.py model_path:=/path/to/yolo11n-seg.onnx
```

Defaults are in [config/params.yaml](ros_onnx_segmentation/config/params.yaml). Pass your own with `params_file:=...`.

### As a component

The plugin is `ros_onnx_segmentation::YoloProcessor`. To load it into the container that runs your camera driver:

```bash
ros2 launch ros_onnx_segmentation composable_launch.py model_path:=/path/to/model.onnx container:=/camera_container
```

Without `container:=` a new container is started. The launch file enables `use_intra_process_comms`; the camera driver must enable it too for zero-copy transport.

### Topics

| Topic | Type | Description |
|---|---|---|
| `image_topic` (sub) | `sensor_msgs/Image` | Input image, any encoding cv_bridge can convert to BGR8 |
| `detections` | `vision_msgs/Detection2DArray` | Boxes in input image pixels. `results[0].hypothesis` has the class name and score |
| `instance_mask` | `sensor_msgs/Image` (mono16) | Same size as the input. 0 is background, pixel value `k` belongs to the detection with `id == "k"` |
| `image_debug_topic` | `sensor_msgs/Image` | Overlay for visual checks, only when `debug` is true |

All outputs carry the input header, so they can be synchronized with the image or depth by timestamp.

Getting the mask of one detection in Python:

```python
ids = bridge.imgmsg_to_cv2(mask_msg)
for det in detections_msg.detections:
    mask = ids == int(det.id)
```

### Parameters

| Name | Default | Description |
|---|---|---|
| `onnx_model_path` | | Path to the `.onnx` file |
| `image_topic` | `camera/image_raw` | Input topic |
| `image_debug_topic` | `image_debug_raw` | Overlay topic |
| `execution_provider` | `cpu` | `cpu`, `cuda` or `tensorrt` |
| `device_id` | `0` | GPU index |
| `trt_cache_dir` | `~/.ros/trt_cache` | TensorRT engine cache. The first run builds the engine and takes a while |
| `num_threads` | `1` | CPU threads for inference |
| `image_processing` | `letterbox` | `letterbox` or `resize` |
| `tensor_width`, `tensor_height` | `640` | Only used if the model has a dynamic input shape |
| `conf_thresh` | `0.35` | Minimum class score |
| `iou_thresh` | `0.6` | NMS threshold, applied per class |
| `mask_thresh` | `0.3` | Mask probability cutoff |
| `class_names` | from model | Overrides the names stored in the model |
| `debug` | `false` | Publish the overlay image |

## License

MIT. `onnxruntime_vendor` and the original node structure come from [ms-iot/ros_msft_onnx](https://github.com/ms-iot/ros_msft_onnx) (MIT, Microsoft).
