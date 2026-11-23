import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory('ros_onnx_segmentation')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.join(share_dir, 'models', 'yolov8n-seg.onnx')),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share_dir, 'config', 'params.yaml')),
        Node(
            package='ros_onnx_segmentation',
            executable='ros_onnx_segmentation',
            name='ros_onnx_segmentation',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'onnx_model_path': LaunchConfiguration('model_path')},
            ]),
    ])
