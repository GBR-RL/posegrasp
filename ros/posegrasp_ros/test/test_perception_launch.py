"""Launch test: an LM-O frame in, a pose within tolerance and grasps out.

Plays one frame (scene 2, image 3) with the true masks, estimates the pose of the driller
(object 8, 97 % visible) with PPF, and plans grasps on it.
"""

import os
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import numpy as np
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from vision_msgs.msg import Detection3DArray

from posegrasp_ros.conversions import pose_to_matrix

OBJECT = 8
DIAMETER_MM = 261.5


@pytest.mark.launch_test
def generate_test_description() -> launch.LaunchDescription:
    share = get_package_share_directory("posegrasp_ros")
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "perception.launch.py")),
        launch_arguments={
            "im_ids": "3",
            "rate": "0.2",
            "objects": str(OBJECT),
            "target_object": str(OBJECT),
        }.items(),
    )
    return launch.LaunchDescription([perception, launch_testing.actions.ReadyToTest()])


class TestPerception(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        rclpy.shutdown()

    def setUp(self) -> None:
        self.node = rclpy.create_node("test_perception")

    def tearDown(self) -> None:
        self.node.destroy_node()

    def wait_for(self, messages: dict, timeout: float) -> None:
        end = time.time() + timeout
        while time.time() < end and not all(messages.values()):
            rclpy.spin_once(self.node, timeout_sec=0.5)

    def test_pose_within_tolerance_and_grasps(self) -> None:
        received: dict[str, list] = {"poses": [], "truth": [], "grasps": []}
        self.node.create_subscription(
            Detection3DArray, "poses", lambda m: received["poses"].append(m), 10
        )
        self.node.create_subscription(
            Detection3DArray, "ground_truth", lambda m: received["truth"].append(m), 10
        )
        self.node.create_subscription(
            PoseArray, "grasps", lambda m: received["grasps"].append(m), 10
        )
        self.wait_for(received, timeout=600.0)

        poses = [m for m in received["poses"] if m.detections]
        self.assertTrue(poses, "no pose estimate published")
        self.assertTrue(received["truth"], "no ground truth published")
        estimate = next(
            d for d in poses[0].detections if d.results[0].hypothesis.class_id == str(OBJECT)
        )
        truth = next(
            d
            for d in received["truth"][0].detections
            if d.results[0].hypothesis.class_id == str(OBJECT)
        )
        est = pose_to_matrix(estimate.results[0].pose.pose)
        gt = pose_to_matrix(truth.results[0].pose.pose)
        translation_error = np.linalg.norm(est[:3, 3] - gt[:3, 3])
        cos_angle = (np.trace(est[:3, :3].T @ gt[:3, :3]) - 1) / 2
        angle_error = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        self.assertLess(translation_error, 0.05 * DIAMETER_MM)
        self.assertLess(angle_error, 5.0)
        self.assertTrue(received["grasps"], "no grasps published")
        self.assertGreater(len(received["grasps"][0].poses), 0)
        self.assertEqual(received["grasps"][0].header.frame_id, "camera")

