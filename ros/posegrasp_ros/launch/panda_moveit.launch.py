"""MoveIt 2 move_group with a Franka Panda on ros2_control mock hardware (no RViz).

Mirrors the demo of moveit_resources_panda_moveit_config without the GUI, so it runs headless.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description() -> LaunchDescription:
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    controllers = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )

    def spawner(name: str) -> Node:
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager"],
            output="screen",
        )

    return LaunchDescription(
        [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["--frame-id", "world", "--child-frame-id", "panda_link0"],
                output="log",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[moveit_config.robot_description],
                output="log",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                parameters=[moveit_config.to_dict()],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[controllers],
                remappings=[("/controller_manager/robot_description", "/robot_description")],
                output="screen",
            ),
            spawner("joint_state_broadcaster"),
            spawner("panda_arm_controller"),
            spawner("panda_hand_controller"),
        ]
    )
