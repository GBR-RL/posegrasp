"""MoveIt 2 pick of the planned grasps with a Franka Panda.

On the first `grasps` message (geometry_msgs/PoseArray, best first) the node adds the table to
the planning scene and tries the grasps in order: open the hand, move to a pre-grasp pose
`approach` metres back along the approach axis, move to the grasp, close the hand, lift by `lift`
metres. Motions go through move_group's MoveGroup action (OMPL), executed on whatever controllers
run (mock hardware in the demo). The outcome is published once on `pick/result` as JSON.

Grasp frames are posegrasp's (y closing, z approach, origin between the finger pads), which is
the Panda hand frame moved `tcp_offset` metres along z.
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from posegrasp_ros.conversions import MM_PER_M, matrix_to_pose, pose_to_matrix, transform_to_matrix


def shift(T: np.ndarray, local_z_mm: float = 0.0, world_z_mm: float = 0.0) -> np.ndarray:
    """T moved along its own z axis, then along the parent frame's z axis."""
    out = T.copy()
    out[:3, 3] += T[:3, 2] * local_z_mm
    out[2, 3] += world_z_mm
    return out


class PickNode(Node):
    def __init__(self) -> None:
        super().__init__("pick")
        defaults = {
            "base_frame": "panda_link0",
            "arm_group": "panda_arm",
            "hand_group": "hand",
            "ee_link": "panda_hand",
            "finger_joint": "panda_finger_joint1",
            "tcp_offset": 0.1034,  # panda_hand -> between the finger pads
            "approach": 0.10,
            "lift": 0.10,
            "attempts": 5,
            "plan_only": False,
        }
        p = {name: self.declare_parameter(name, value).value for name, value in defaults.items()}
        self.base_frame = str(p["base_frame"])
        self.arm_group, self.hand_group = str(p["arm_group"]), str(p["hand_group"])
        self.ee_link, self.finger_joint = str(p["ee_link"]), str(p["finger_joint"])
        self.tcp_mm = float(p["tcp_offset"]) * MM_PER_M
        self.approach_mm = float(p["approach"]) * MM_PER_M
        self.lift_mm = float(p["lift"]) * MM_PER_M
        self.attempts = int(p["attempts"])
        self.plan_only = bool(p["plan_only"])

        group = ReentrantCallbackGroup()
        self.move = ActionClient(self, MoveGroup, "move_action", callback_group=group)
        self.apply_scene = self.create_client(
            ApplyPlanningScene, "apply_planning_scene", callback_group=group
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(String, "pick/result", 5)
        self.create_subscription(PoseArray, "grasps", self.on_grasps, 5, callback_group=group)
        self.busy = threading.Lock()
        self.finished = False
        self.get_logger().info("pick node waiting for grasps")

    def on_grasps(self, msg: PoseArray) -> None:
        if self.finished or not msg.poses or not self.busy.acquire(blocking=False):
            return
        threading.Thread(target=self.run, args=(msg,), daemon=True).start()

    def run(self, msg: PoseArray) -> None:
        try:
            result = self.pick(msg)
            self.finished = True
            self.pub.publish(String(data=json.dumps(result)))
            self.get_logger().info(f"pick result: {json.dumps(result)}")
        except Exception as error:  # noqa: BLE001 (report any failure on the result topic)
            self.get_logger().error(f"pick failed: {error!r}")
        finally:
            self.busy.release()

    def pick(self, msg: PoseArray) -> dict[str, Any]:
        if not self.move.wait_for_server(timeout_sec=120.0):
            return {"success": False, "error": "move_group action server not available"}
        self.add_table()
        tf = self.tf_buffer.lookup_transform(
            self.base_frame, msg.header.frame_id, Time(), timeout=Duration(seconds=10.0)
        )
        camera_in_base = transform_to_matrix(tf.transform)
        tried = []
        for k, grasp_pose in enumerate(msg.poses[: self.attempts]):
            grasp = camera_in_base @ pose_to_matrix(grasp_pose)
            hand = shift(grasp, local_z_mm=-self.tcp_mm)
            stages: list[tuple[str, MoveGroup.Goal]] = [
                ("open", self.hand_goal(0.04)),
                ("pre_grasp", self.pose_goal(shift(hand, local_z_mm=-self.approach_mm))),
                ("grasp", self.pose_goal(hand)),
                ("close", self.hand_goal(0.0)),
                ("lift", self.pose_goal(shift(hand, world_z_mm=self.lift_mm))),
            ]
            record: dict[str, Any] = {"grasp": k, "stages": {}}
            ok = True
            for name, goal in stages:
                ok, planning_time = self.execute(goal)
                record["stages"][name] = {"ok": ok, "planning_time": planning_time}
                if not ok:
                    record["failed_at"] = name
                    break
            tried.append(record)
            if ok:
                return {"success": True, "grasp": k, "attempts": tried, "plan_only": self.plan_only}
        return {"success": False, "attempts": tried, "plan_only": self.plan_only}

    # ---- MoveIt requests -----------------------------------------------------------------------

    def _request(self, goal: MoveGroup.Goal, group: str) -> None:
        r = goal.request
        r.group_name = group
        r.num_planning_attempts = 5
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = 0.5
        r.max_acceleration_scaling_factor = 0.5
        r.start_state.is_diff = True
        goal.planning_options.plan_only = self.plan_only
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

    def pose_goal(self, T_mm: np.ndarray) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        self._request(goal, self.arm_group)
        pose = matrix_to_pose(T_mm)
        position = PositionConstraint()
        position.header.frame_id = self.base_frame
        position.link_name = self.ee_link
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005])
        position.constraint_region = BoundingVolume(
            primitives=[sphere], primitive_poses=[Pose(position=pose.position)]
        )
        position.weight = 1.0
        orientation = OrientationConstraint()
        orientation.header.frame_id = self.base_frame
        orientation.link_name = self.ee_link
        orientation.orientation = pose.orientation
        orientation.absolute_x_axis_tolerance = 0.05
        orientation.absolute_y_axis_tolerance = 0.05
        orientation.absolute_z_axis_tolerance = 0.05
        orientation.weight = 1.0
        goal.request.goal_constraints = [
            Constraints(position_constraints=[position], orientation_constraints=[orientation])
        ]
        return goal

    def hand_goal(self, finger_position: float) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        self._request(goal, self.hand_group)
        joint = JointConstraint(
            joint_name=self.finger_joint,
            position=finger_position,
            tolerance_above=0.002,
            tolerance_below=0.002,
            weight=1.0,
        )
        goal.request.goal_constraints = [Constraints(joint_constraints=[joint])]
        return goal

    def execute(self, goal: MoveGroup.Goal) -> tuple[bool, float]:
        handle = self._wait(self.move.send_goal_async(goal), 30.0)
        if handle is None or not handle.accepted:
            return False, math.nan
        response = self._wait(handle.get_result_async(), 180.0)
        if response is None:
            return False, math.nan
        result = response.result
        return result.error_code.val == MoveItErrorCodes.SUCCESS, float(result.planning_time)

    def add_table(self) -> None:
        """The table under the objects (the base frame's z = 0 plane), in front of the robot."""
        if not self.apply_scene.wait_for_service(timeout_sec=30.0):
            self.get_logger().warning("apply_planning_scene not available: no table added")
            return
        table = CollisionObject()
        table.header.frame_id = self.base_frame
        table.id = "table"
        table.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.6, 0.02])]
        table.primitive_poses = [Pose(position=Point(x=0.65, y=0.0, z=-0.012))]
        table.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [table]
        self._wait(self.apply_scene.call_async(ApplyPlanningScene.Request(scene=scene)), 30.0)

    @staticmethod
    def _wait(future: Any, timeout: float) -> Any:
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            return None
        return future.result()


def main() -> None:
    rclpy.init()
    node = PickNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
