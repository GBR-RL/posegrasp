"""Plays LM-O (BOP format) frames as an RGB-D camera.

Publishes per frame, all with one timestamp:
- camera/color/image_raw (rgb8), camera/depth/image_raw (16UC1, mm), camera/camera_info
- detections (vision_msgs/Detection2DArray): one per target object, `id` = its label value
- detections/labels (mono16): the detection masks, pixel value = detection `id`, 0 = none
- ground_truth (vision_msgs/Detection3DArray): the true poses, for evaluation
and once, a static transform placing the camera relative to a robot base on the table
(posegrasp.scene). Detections come from the ground-truth masks or the CNOS default detections.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from posegrasp.data import detections as det
from posegrasp.data.bop import Dataset, Frame, Target
from posegrasp.pipeline import CONDITIONS
from posegrasp.scene import base_in_camera
from posegrasp_ros.conversions import (
    array_to_image,
    camera_info,
    matrix_to_pose,
    matrix_to_transform,
)

MASK_CONDITIONS = ("gt", "cnos")  # the ROS pose node works on masks; GDRNPP gives boxes only


class BopPlayer(Node):
    def __init__(self) -> None:
        super().__init__("bop_player")
        self.declare_parameter("data_dir", os.environ.get("POSEGRASP_DATA_DIR", "data"))
        self.declare_parameter("condition", "gt")
        self.declare_parameter("scene_id", 2)
        self.declare_parameter("im_ids", "all")  # "all" or comma-separated image ids
        self.declare_parameter("rate", 0.5)  # frames per second
        self.declare_parameter("loop", True)
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("robot_reach", 0.5)  # m from the robot base to the objects' centre

        data_dir = Path(self.get_parameter("data_dir").value)
        self.condition = str(self.get_parameter("condition").value)
        if self.condition not in MASK_CONDITIONS:
            raise ValueError(f"condition must be one of {MASK_CONDITIONS}")
        scene_id = int(self.get_parameter("scene_id").value)
        self.dataset = Dataset(data_dir / "lmo")
        targets = [t for t in self.dataset.targets() if t.scene_id == scene_id]
        images = sorted({t.im_id for t in targets})
        wanted = str(self.get_parameter("im_ids").value)
        if wanted != "all":
            keep = {int(v) for v in wanted.split(",")}
            images = [i for i in images if i in keep]
        if not images:
            raise ValueError(f"no targeted images {wanted} in scene {scene_id}")
        self.frames = [(scene_id, i) for i in images]
        self.targets: dict[int, list[Target]] = {}
        for t in targets:
            self.targets.setdefault(t.im_id, []).append(t)
        self.chosen: dict[tuple[int, int, int], list[det.Detection]] = {}
        if self.condition != "gt":
            path = data_dir / "detections" / str(CONDITIONS[self.condition])
            self.chosen = det.for_targets(det.load(path), targets)

        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.reach_mm = 1000.0 * float(self.get_parameter("robot_reach").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.pub_rgb = self.create_publisher(Image, "camera/color/image_raw", 5)
        self.pub_depth = self.create_publisher(Image, "camera/depth/image_raw", 5)
        self.pub_info = self.create_publisher(CameraInfo, "camera/camera_info", 5)
        self.pub_dets = self.create_publisher(Detection2DArray, "detections", 5)
        self.pub_labels = self.create_publisher(Image, "detections/labels", 5)
        self.pub_truth = self.create_publisher(Detection3DArray, "ground_truth", 5)
        self.static_tf = StaticTransformBroadcaster(self)
        self.base_sent = False
        self.index = 0
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate").value), self.tick)
        self.get_logger().info(
            f"playing {len(self.frames)} frame(s) of scene {scene_id} with {self.condition} "
            "detections"
        )

    def tick(self) -> None:
        if self.index >= len(self.frames):
            if not self.loop:
                self.timer.cancel()
                return
            self.index = 0
        scene_id, im_id = self.frames[self.index]
        self.index += 1
        self.publish(self.dataset.frame(scene_id, im_id))

    def publish(self, frame: Frame) -> None:
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.camera_frame)
        depth = frame.depth_mm()
        K, h, w = frame.camera.K, frame.camera.height, frame.camera.width
        if not self.base_sent:
            self.send_base(header, depth, frame)
        labels = np.zeros((h, w), dtype=np.uint16)
        detections = Detection2DArray(header=header)
        for target in self.targets.get(frame.im_id, []):
            if self.condition == "gt":
                instance = next(g for g in frame.gt if g.obj_id == target.obj_id)
                mask, score = frame.gt_mask(instance), 1.0
            else:
                found = self.chosen.get((frame.scene_id, frame.im_id, target.obj_id), [])
                if not found:
                    continue
                mask, score = found[0].mask(h, w), found[0].score
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            label = len(detections.detections) + 1
            labels[mask & (labels == 0)] = label
            d = Detection2D(header=header, id=str(label))
            d.bbox.center.position.x = float(xs.min() + xs.max()) / 2
            d.bbox.center.position.y = float(ys.min() + ys.max()) / 2
            d.bbox.size_x = float(xs.max() - xs.min() + 1)
            d.bbox.size_y = float(ys.max() - ys.min() + 1)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(target.obj_id)
            hypothesis.hypothesis.score = float(score)
            d.results.append(hypothesis)
            detections.detections.append(d)
        truth = Detection3DArray(header=header)
        for g in frame.gt:
            d3 = Detection3D(header=header, id=str(g.obj_id))
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(g.obj_id)
            hypothesis.hypothesis.score = float(g.visib_fract)
            hypothesis.pose.pose = matrix_to_pose(g.pose.matrix())
            d3.results.append(hypothesis)
            d3.bbox.center = hypothesis.pose.pose
            truth.detections.append(d3)
        self.pub_rgb.publish(array_to_image(frame.rgb(), "rgb8", header))
        depth_mm = np.clip(np.round(depth), 0, 65535).astype(np.uint16)
        self.pub_depth.publish(array_to_image(depth_mm, "16UC1", header))
        self.pub_info.publish(camera_info(K, w, h, header))
        self.pub_labels.publish(array_to_image(labels, "mono16", header))
        self.pub_dets.publish(detections)
        self.pub_truth.publish(truth)

    def send_base(self, header: Header, depth: np.ndarray, frame: Frame) -> None:
        centre = np.mean([g.pose.t for g in frame.gt], axis=0)
        base = base_in_camera(depth, frame.camera.K, centre=centre, reach=self.reach_mm)
        tf = TransformStamped()
        tf.header.stamp = header.stamp
        tf.header.frame_id = self.base_frame
        tf.child_frame_id = self.camera_frame
        tf.transform = matrix_to_transform(np.linalg.inv(base))  # the camera in the base frame
        self.static_tf.sendTransform(tf)
        self.base_sent = True


def main() -> None:
    rclpy.init()
    node = BopPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
