# ros_onnx_segmentation

[![build](https://github.com/Leeseunghun03/ros_onnx_segmentation/actions/workflows/build.yml/badge.svg)](https://github.com/Leeseunghun03/ros_onnx_segmentation/actions/workflows/build.yml)

ROS 2 C++ node for YOLOv8 / YOLO11 instance segmentation with ONNX Runtime.

- MIT license. Runtime dependencies are ONNX Runtime (MIT) and OpenCV.
- Publishes `vision_msgs/Detection2DArray` and a per-pixel instance mask.
- Runs as a node or as a component (zero-copy with intra-process communication).
- CPU on x64 and aarch64, CUDA and TensorRT on x64. A custom ONNX Runtime build can be used instead (e.g. Jetson).

![demo](docs/demo.gif)

yolo11s-seg, CUDA. Video: [Pro dog walker, Tokyo](https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm) by Nesnad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The GIF is under the same license.

Tested on Humble x64 (CPU, CUDA, TensorRT). CI builds Humble x64/aarch64/GPU and Jazzy x64.

## Performance

960x540 input, from image publish to `detections` received:

- Ryzen 9 5950X, `num_threads: 4`: yolo11n-seg at about 16 fps
- RTX 3090, TensorRT: about 10 ms for yolo11n-seg and yolo11s-seg

## Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/Leeseunghun03/ros_onnx_segmentation.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build
```

The build downloads ONNX Runtime 1.26.0 (CPU).

GPU (Linux x64, needs CUDA 12 and cuDNN 9, plus TensorRT 10 for `tensorrt`):

```bash
colcon build --cmake-args -DONNXRUNTIME_GPU=ON
```

Existing ONNX Runtime install with `include/` and `lib/libonnxruntime.so`:

```bash
colcon build --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

## Model

```bash
pip install ultralytics
yolo export model=yolo11n-seg.pt format=onnx imgsz=640
```

Class names and shapes are read from the model. Pretrained Ultralytics weights are AGPL-3.0.

## Run

```bash
ros2 launch ros_onnx_segmentation segmentation_launch.py model_path:=/path/to/yolo11n-seg.onnx
```

Default parameters: [config/params.yaml](ros_onnx_segmentation/config/params.yaml). Override with `params_file:=...`.

As a component (`ros_onnx_segmentation::YoloProcessor`), in a new container or an existing one:

```bash
ros2 launch ros_onnx_segmentation composable_launch.py model_path:=/path/to/model.onnx container:=/camera_container
```

Zero-copy needs `use_intra_process_comms` on the camera driver as well.

### Topics

| Topic | Type | Description |
|---|---|---|
| `image_topic` (sub) | `sensor_msgs/Image` | Input image |
| `detections` | `vision_msgs/Detection2DArray` | Boxes in input pixels, class name and score in `results[0].hypothesis` |
| `instance_mask` | `sensor_msgs/Image` (mono16) | Input size. 0 is background, `k` is the detection with `id == "k"` |
| `image_debug_topic` | `sensor_msgs/Image` | Overlay, when `debug` is true |

Outputs keep the input header.

```python
ids = bridge.imgmsg_to_cv2(mask_msg)
for det in detections_msg.detections:
    mask = ids == int(det.id)
```

### Parameters

| Name | Default | Description |
|---|---|---|
| `onnx_model_path` | | `.onnx` file |
| `image_topic` | `camera/image_raw` | Input topic |
| `image_debug_topic` | `image_debug_raw` | Overlay topic |
| `execution_provider` | `cpu` | `cpu`, `cuda` or `tensorrt`. Falls back to `cpu` if the provider library cannot be loaded |
| `device_id` | `0` | GPU index |
| `trt_cache_dir` | `~/.ros/trt_cache` | TensorRT engine cache. The first run builds the engine (a few minutes) |
| `num_threads` | `1` | CPU inference threads |
| `image_processing` | `letterbox` | `letterbox` or `resize` |
| `tensor_width`, `tensor_height` | `640` | Used if the model input size is dynamic |
| `conf_thresh` | `0.35` | Minimum class score |
| `iou_thresh` | `0.6` | NMS threshold, per class |
| `mask_thresh` | `0.3` | Mask probability threshold |
| `class_names` | from model | Overrides model class names |
| `debug` | `false` | Publish the overlay |

## License

MIT. `onnxruntime_vendor` and the original node structure are from [ms-iot/ros_msft_onnx](https://github.com/ms-iot/ros_msft_onnx) (MIT).
