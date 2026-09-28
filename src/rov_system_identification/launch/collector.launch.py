from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rov_system_identification",
            executable="collect_physical_data",
            name="rov_system_id_collector",
            output="screen",
        )
    ])
