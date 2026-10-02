"""Conversions between ROS messages and the numpy arrays posegrasp works with.

ROS uses metres, posegrasp millimetres: poses and clouds are scaled at this boundary only.
"""

from __future__ import annotations

import array

import numpy as np
from geometry_msgs.msg import Pose, Transform
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

MM_PER_M = 1000.0

_DTYPES = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}
_FIELD_TYPES = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def image_to_array(msg: Image) -> np.ndarray:
    dtype, channels = _DTYPES[msg.encoding]
    itemsize = np.dtype(dtype).itemsize
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = rows[:, : msg.width * channels * itemsize].copy()
    order = ">" if msg.is_bigendian else "<"
    values = pixels.view(np.dtype(dtype).newbyteorder(order)).astype(dtype)
    shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
    return values.reshape(shape)


def array_to_image(pixels: np.ndarray, encoding: str, header: Header) -> Image:
    dtype, channels = _DTYPES[encoding]
    data = np.ascontiguousarray(pixels, dtype=np.dtype(dtype).newbyteorder("<"))
    msg = Image(header=header, encoding=encoding, is_bigendian=0)
    msg.height, msg.width = int(data.shape[0]), int(data.shape[1])
    msg.step = msg.width * channels * data.itemsize
    msg.data = array.array("B", data.tobytes())  # skips per-byte checks
    return msg


def camera_info(K: np.ndarray, width: int, height: int, header: Header) -> CameraInfo:
    msg = CameraInfo(header=header, width=width, height=height, distortion_model="plumb_bob")
    msg.k = [float(v) for v in K.reshape(-1)]
    msg.d = [0.0] * 5
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0, 0.0, float(K[1, 1]), float(K[1, 2]), 0.0]
    msg.p += [0.0, 0.0, 1.0, 0.0]
    return msg


def matrix_to_pose(T: np.ndarray, *, scale: float = 1 / MM_PER_M) -> Pose:
    """4x4 transform (translation in mm by default) to a Pose in metres."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) * scale for v in T[:3, 3])
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    pose.orientation.x, pose.orientation.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = float(z), float(w)
    return pose


def pose_to_matrix(pose: Pose, *, scale: float = MM_PER_M) -> np.ndarray:
    """Pose in metres to a 4x4 transform (translation in mm by default)."""
    q = pose.orientation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z]) * scale
    return T


def matrix_to_transform(T: np.ndarray, *, scale: float = 1 / MM_PER_M) -> Transform:
    pose = matrix_to_pose(T, scale=scale)
    msg = Transform()
    msg.translation.x, msg.translation.y = pose.position.x, pose.position.y
    msg.translation.z = pose.position.z
    msg.rotation = pose.orientation
    return msg


def transform_to_matrix(msg: Transform, *, scale: float = MM_PER_M) -> np.ndarray:
    pose = Pose()
    pose.position.x, pose.position.y = msg.translation.x, msg.translation.y
    pose.position.z = msg.translation.z
    pose.orientation = msg.rotation
    return pose_to_matrix(pose, scale=scale)


def cloud_fields(msg: PointCloud2) -> dict[str, np.ndarray]:
    """Each field of a point cloud as a numpy array."""
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [np.dtype(_FIELD_TYPES[f.datatype]).newbyteorder(order) for f in msg.fields],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )
    points = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    return {name: np.asarray(points[name]) for name in dtype.names}


def cloud_points_mm(msg: PointCloud2) -> np.ndarray:
    """(N, 3) points in millimetres."""
    fields = cloud_fields(msg)
    xyz = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(np.float64)
    return xyz * MM_PER_M


def xyz_cloud(points_mm: np.ndarray, header: Header) -> PointCloud2:
    """A dense float32 x, y, z cloud in metres."""
    data = np.ascontiguousarray(points_mm / MM_PER_M, dtype="<f4")
    msg = PointCloud2(header=header, height=1, width=len(data), is_bigendian=False)
    msg.fields = [
        PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate("xyz")
    ]
    msg.point_step, msg.row_step = 12, 12 * len(data)
    msg.is_dense = True
    msg.data = array.array("B", data.tobytes())  # skips per-byte checks
    return msg
