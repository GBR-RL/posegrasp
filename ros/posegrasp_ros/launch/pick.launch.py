"""The whole stack: perception (scene player, segmentation, pose, grasps) and a MoveIt 2 pick
with a Franka Panda on mock hardware."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PERCEPTION_ARGUMENTS = (
    "data_dir",
    "condition",
    "scene_id",
    "im_ids",
    "rate",
    "estimator",
    "objects",
    "target_object",
)


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("posegrasp_ros")
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "perception.launch.py")),
        launch_arguments={name: LaunchConfiguration(name) for name in PERCEPTION_ARGUMENTS}.items(),
    )
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "panda_moveit.launch.py"))
    )
    pick = Node(
        package="posegrasp_ros",
        executable="pick",
        parameters=[
            {"plan_only": ParameterValue(LaunchConfiguration("plan_only"), value_type=bool)}
        ],
        output="screen",
    )
    defaults = {
        "data_dir": os.environ.get("POSEGRASP_DATA_DIR", "data"),
        "condition": "gt",
        "scene_id": "2",
        "im_ids": "all",
        "rate": "0.5",
        "estimator": "ppf",
        "objects": "",
        "target_object": "0",
        "plan_only": "false",
    }
    declared = [DeclareLaunchArgument(n, default_value=v) for n, v in defaults.items()]
    return LaunchDescription([*declared, moveit, perception, pick])
