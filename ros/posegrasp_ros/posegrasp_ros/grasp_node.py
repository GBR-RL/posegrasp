"""Grasp planning as a ROS 2 node.

Subscribes to `poses` (vision_msgs/Detection3DArray) and the scene cloud `scene/points`;
publishes the feasible grasps of the target object, best first, as `grasps`
(geometry_msgs/PoseArray, grasp frames in the camera frame: y closing, z approach), and the
gripper at the best grasp as `grasps/markers` for RViz.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import message_filters
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from posegrasp.data.bop import Dataset
from posegrasp.geometry import load_model
from posegrasp.grasp import GraspSet, Planner
from posegrasp.picking import grasp_set
from posegrasp_ros.conversions import MM_PER_M, cloud_points_mm, matrix_to_pose, pose_to_matrix


class GraspPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("grasp_planner")
        data_dir = os.environ.get("POSEGRASP_DATA_DIR", "data")
        self.declare_parameter("data_dir", data_dir)
        self.declare_parameter("grasp_dir", "")  # default: <data_dir>/grasps
        self.declare_parameter("target_object", 0)  # 0: the most confident estimate
        self.declare_parameter("keep", 10)
        data = Path(self.get_parameter("data_dir").value)
        self.grasp_dir = Path(str(self.get_parameter("grasp_dir").value) or data / "grasps")
        self.target = int(self.get_parameter("target_object").value)
        self.planner = Planner(keep=int(self.get_parameter("keep").value))
        self.models = Dataset(data / "lmo").models()
        self.grasps: dict[int, GraspSet] = {}
        if self.target:
            self.grasp_set(self.target)
        self.pub = self.create_publisher(PoseArray, "grasps", 5)
        self.pub_markers = self.create_publisher(MarkerArray, "grasps/markers", 5)
        poses = message_filters.Subscriber(self, Detection3DArray, "poses")
        scene = message_filters.Subscriber(self, PointCloud2, "scene/points")
        self.sync = message_filters.TimeSynchronizer([poses, scene], 10)
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info("grasp planner ready")

    def grasp_set(self, obj_id: int) -> GraspSet:
        if obj_id not in self.grasps:
            start = time.perf_counter()
            self.grasps[obj_id] = grasp_set(self.models[obj_id], self.grasp_dir)
            seconds = time.perf_counter() - start
            self.get_logger().info(
                f"{len(self.grasps[obj_id])} grasps for object {obj_id} ({seconds:.1f} s)"
            )
        return self.grasps[obj_id]

    def on_frame(self, poses: Detection3DArray, scene: PointCloud2) -> None:
        candidates = [
            d
            for d in poses.detections
            if not self.target or int(d.results[0].hypothesis.class_id) == self.target
        ]
        if not candidates:
            return
        chosen = max(candidates, key=lambda d: d.results[0].hypothesis.score)
        obj_id = int(chosen.results[0].hypothesis.class_id)
        model = self.models[obj_id]
        start = time.perf_counter()
        plan = self.planner.plan(
            self.grasp_set(obj_id),
            pose_to_matrix(chosen.results[0].pose.pose),
            surface=load_model(model.mesh_path, model.diameter).points,
            scene=cloud_points_mm(scene),
            diameter=model.diameter,
        )
        seconds = time.perf_counter() - start
        out = PoseArray(header=poses.header)
        out.poses = [matrix_to_pose(G) for G in plan.ranked]
        self.pub.publish(out)
        self.get_logger().info(
            f"object {obj_id}: {plan.candidates} grasps face the camera, {plan.feasible} "
            f"collision-free, {len(out.poses)} sent ({seconds:.2f} s)"
        )
        if plan.grasp is not None:
            self.pub_markers.publish(self.markers(plan.grasp, poses.header))

    def markers(self, grasp: np.ndarray, header: object) -> MarkerArray:
        centres, halves = self.planner.gripper.boxes()
        out = MarkerArray()
        for i, (centre, half) in enumerate(zip(centres, halves, strict=True)):
            box = np.eye(4)
            box[:3, 3] = centre
            m = Marker(header=header, ns="gripper", id=i, type=Marker.CUBE, action=Marker.ADD)
            m.pose = matrix_to_pose(grasp @ box)
            m.scale.x, m.scale.y, m.scale.z = (float(v) * 2 / MM_PER_M for v in half)
            m.color.r, m.color.g, m.color.b, m.color.a = 0.16, 0.47, 0.84, 0.6
            out.markers.append(m)
        return out


def main() -> None:
    rclpy.init()
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
