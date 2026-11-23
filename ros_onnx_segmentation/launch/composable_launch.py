import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    share_dir = get_package_share_directory('ros_onnx_segmentation')

    segmentation = ComposableNode(
        package='ros_onnx_segmentation',
        plugin='ros_onnx_segmentation::YoloProcessor',
        name='ros_onnx_segmentation',
        parameters=[
            LaunchConfiguration('params_file'),
            {'onnx_model_path': LaunchConfiguration('model_path')},
        ],
        extra_arguments=[{'use_intra_process_comms': True}])

    # container:=<name> loads into an existing container, e.g. the one running the camera driver.
    no_container = PythonExpression(["'", LaunchConfiguration('container'), "' == ''"])

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.join(share_dir, 'models', 'yolov8n-seg.onnx')),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share_dir, 'config', 'params.yaml')),
        DeclareLaunchArgument('container', default_value=''),
        ComposableNodeContainer(
            name='segmentation_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[segmentation],
            output='screen',
            condition=IfCondition(no_container)),
        LoadComposableNodes(
            target_container=LaunchConfiguration('container'),
            composable_node_descriptions=[segmentation],
            condition=UnlessCondition(no_container)),
    ])

