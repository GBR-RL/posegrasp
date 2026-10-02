"""Scene player -> C++ segmentation component -> pose estimation -> grasp planning."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = {
    "data_dir": (os.environ.get("POSEGRASP_DATA_DIR", "data"), "LM-O and detections folder"),
    "condition": ("gt", "Detections: gt (true masks) or cnos (zero-shot masks)"),
    "scene_id": ("2", "BOP scene"),
    "im_ids": ("all", "Image ids, comma-separated, or all"),
    "rate": ("0.5", "Frames per second"),
    "estimator": ("ppf", "Pose estimator: ppf or fpfh"),
    "objects": ("", "Object ids to estimate, comma-separated; empty: all"),
    "target_object": ("0", "Object to grasp; 0: the most confident estimate"),
}


def generate_launch_description() -> LaunchDescription:
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=text)
        for name, (default, text) in ARGUMENTS.items()
    ]
    arg = {name: LaunchConfiguration(name) for name in ARGUMENTS}

    def as_str(name: str) -> ParameterValue:
        return ParameterValue(arg[name], value_type=str)

    player = Node(
        package="posegrasp_ros",
        executable="bop_player",
        parameters=[
            {
                "data_dir": as_str("data_dir"),
                "condition": as_str("condition"),
                "scene_id": ParameterValue(arg["scene_id"], value_type=int),
                "im_ids": as_str("im_ids"),
                "rate": ParameterValue(arg["rate"], value_type=float),
            }
        ],
        output="screen",
    )
    segmentation = ComposableNodeContainer(
        name="perception_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="posegrasp_cpp",
                plugin="posegrasp_cpp::SegmentCloud",
                name="segment_cloud",
                extra_arguments=[{"use_intra_process_comms": True}],
            )
        ],
        output="screen",
    )
    pose = Node(
        package="posegrasp_ros",
        executable="pose_estimator",
        parameters=[
            {
                "data_dir": as_str("data_dir"),
                "estimator": as_str("estimator"),
                "objects": as_str("objects"),
            }
        ],
        output="screen",
    )
    grasps = Node(
        package="posegrasp_ros",
        executable="grasp_planner",
        parameters=[
            {
                "data_dir": as_str("data_dir"),
                "target_object": ParameterValue(arg["target_object"], value_type=int),
            }
        ],
        output="screen",
    )
    return LaunchDescription([*declared, player, segmentation, pose, grasps])
