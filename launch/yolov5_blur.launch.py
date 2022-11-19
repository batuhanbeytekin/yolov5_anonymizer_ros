import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('yolov5_anonymizer_ros'),
        'config',
        'params.yaml'
        )
        
    node=Node(
        package = 'yolov5_anonymizer_ros',
        name = 'yolov5_blur',
        executable = 'yolov5_blur',
        parameters = [config]
    )

    ld.add_action(node)
    return ld