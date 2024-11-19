import os
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch import conditions
import platform 

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Use share directory to locate model path within the package
    share_dir = get_package_share_directory('ros_onnx_segmentation')
    ModelPath = os.path.join(
        share_dir,
        'model',
        'run15.onnx')
    
    os_name = platform.system()
    os_flag = "false" if os_name == 'Windows' else "true"  # default to Linux

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'onnx_model_path_arg', 
            default_value=ModelPath,
            description="Onnx model path"),
        launch_ros.actions.Node(
            package='ros_onnx_segmentation', executable='ros_onnx_segmentation', output='screen',
            name='ros_onnx_segmentation',
            parameters=[
                {'onnx_model_path': launch.substitutions.LaunchConfiguration('onnx_model_path_arg')},
                {'confidence': 0.5},
                {'tensor_width': 640},
                {'tensor_height': 640},
                {'debug': True}
            ]),
    ])
