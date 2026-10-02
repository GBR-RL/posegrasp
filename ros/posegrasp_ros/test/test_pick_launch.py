"""Launch test: the whole stack, from an LM-O frame to a MoveIt 2 pick with the Panda.

Perception as in test_perception_launch, plus move_group and the Panda on ros2_control mock
hardware; the pick node must plan and execute open, pre-grasp, grasp, close and lift for one of
the planned grasps.
"""

import json
import os
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from std_msgs.msg import String

OBJECT = 8


@pytest.mark.launch_test
def generate_test_description() -> launch.LaunchDescription:
    share = get_package_share_directory("posegrasp_ros")
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "pick.launch.py")),
        launch_arguments={
            "im_ids": "3",
            "rate": "0.2",
            "objects": str(OBJECT),
            "target_object": str(OBJECT),
        }.items(),
    )
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()])


class TestPick(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        rclpy.shutdown()

    def test_planned_and_executed_pick(self) -> None:
        node = rclpy.create_node("test_pick")
        results: list[dict] = []
        node.create_subscription(
            String, "pick/result", lambda m: results.append(json.loads(m.data)), 10
        )
        end = time.time() + 1200.0
        while time.time() < end and not results:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()
        self.assertTrue(results, "no pick result published")
        result = results[0]
        self.assertTrue(result["success"], json.dumps(result, indent=2))
        stages = result["attempts"][-1]["stages"]
        self.assertEqual(list(stages), ["open", "pre_grasp", "grasp", "close", "lift"])
        self.assertTrue(all(s["ok"] for s in stages.values()))
