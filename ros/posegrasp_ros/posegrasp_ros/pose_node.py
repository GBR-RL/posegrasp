"""6-DoF pose estimation as a ROS 2 node.

Subscribes to the segmented point clouds of the C++ `segment_cloud` component (field `label` =
detection id) and to the 2D detections; publishes `poses` (vision_msgs/Detection3DArray, camera
frame, metres) and a TF frame `object_<obj_id>` per estimate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import message_filters
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros.transform_broadcaster import TransformBroadcaster
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from posegrasp.data.bop import Dataset
from posegrasp.estimators import create_estimator
from posegrasp.estimators.base import best
from posegrasp.geometry import load_model, remove_outliers
from posegrasp_ros.conversions import MM_PER_M, cloud_fields, matrix_to_pose, matrix_to_transform


def parse_ids(text: str) -> set[int]:
    return {int(v) for v in text.split(",") if v.strip()}


class PoseEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_estimator")
        self.declare_parameter("data_dir", os.environ.get("POSEGRASP_DATA_DIR", "data"))
        self.declare_parameter("estimator", "ppf")  # ppf | fpfh (frozen benchmark settings)
        self.declare_parameter("objects", "")  # comma-separated object ids; empty: all
        data_dir = Path(self.get_parameter("data_dir").value)
        name = str(self.get_parameter("estimator").value)
        self.objects = parse_ids(str(self.get_parameter("objects").value))
        self.models = Dataset(data_dir / "lmo").models()
        self.estimator = create_estimator(name, **({"restarts": 5} if name == "fpfh" else {}))
        for obj_id in sorted(self.objects):  # offline model preparation before frames arrive
            start = time.perf_counter()
            model = self.models[obj_id]
            self.estimator.prepare(obj_id, load_model(model.mesh_path, model.diameter))
            seconds = time.perf_counter() - start
            self.get_logger().info(f"{name} model of object {obj_id} ready in {seconds:.1f} s")
        self.pub = self.create_publisher(Detection3DArray, "poses", 5)
        self.tf = TransformBroadcaster(self)
        segments = message_filters.Subscriber(self, PointCloud2, "segments")
        detections = message_filters.Subscriber(self, Detection2DArray, "detections")
        self.sync = message_filters.TimeSynchronizer([segments, detections], 10)
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info(f"pose estimator ({name}) ready")

    def on_frame(self, segments: PointCloud2, detections: Detection2DArray) -> None:
        fields = cloud_fields(segments)
        points = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(float) * MM_PER_M
        labels = fields["label"].astype(int)
        out = Detection3DArray(header=segments.header)
        for d in detections.detections:
            obj_id = int(d.results[0].hypothesis.class_id)
            if (self.objects and obj_id not in self.objects) or obj_id not in self.models:
                continue
            model = self.models[obj_id]
            segment = remove_outliers(points[labels == int(d.id)])
            if len(segment) < 10:
                continue
            cloud = load_model(model.mesh_path, model.diameter)
            self.estimator.prepare(obj_id, cloud)
            start = time.perf_counter()
            estimate = best(self.estimator.hypotheses(obj_id, cloud, segment))
            seconds = time.perf_counter() - start
            if estimate.pose is None:
                continue
            T = estimate.pose.matrix()
            d3 = Detection3D(header=segments.header, id=d.id)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(obj_id)
            hypothesis.hypothesis.score = float(estimate.score)
            hypothesis.pose.pose = matrix_to_pose(T)
            d3.results.append(hypothesis)
            d3.bbox.center = hypothesis.pose.pose
            d3.bbox.size.x = d3.bbox.size.y = d3.bbox.size.z = model.diameter / MM_PER_M
            out.detections.append(d3)
            tf = TransformStamped()
            tf.header = segments.header
            tf.child_frame_id = f"object_{obj_id}"
            tf.transform = matrix_to_transform(T)
            self.tf.sendTransform(tf)
            self.get_logger().info(
                f"object {obj_id}: {len(segment)} points, score {estimate.score:.2f}, "
                f"{seconds:.2f} s"
            )
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
