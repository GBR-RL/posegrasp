"""Message conversions: round trips and units."""

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from posegrasp_ros.conversions import (
    array_to_image,
    camera_info,
    cloud_fields,
    cloud_points_mm,
    image_to_array,
    matrix_to_pose,
    matrix_to_transform,
    pose_to_matrix,
    transform_to_matrix,
    xyz_cloud,
)


def test_image_round_trips() -> None:
    header = Header(frame_id="camera")
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (4, 5, 3), dtype=np.uint8)
    depth = rng.integers(0, 4000, (4, 5), dtype=np.uint16)
    assert np.array_equal(image_to_array(array_to_image(rgb, "rgb8", header)), rgb)
    msg = array_to_image(depth, "16UC1", header)
    assert msg.step == 10
    assert np.array_equal(image_to_array(msg), depth)


def test_image_with_padded_rows() -> None:
    msg = array_to_image(np.arange(6, dtype=np.uint8).reshape(2, 3), "mono8", Header())
    msg.step = 4  # one padding byte per row
    msg.data = [0, 1, 2, 99, 3, 4, 5, 99]
    assert image_to_array(msg).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_pose_and_transform_scale_millimetres_to_metres() -> None:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.1, -0.4, 0.7]).as_matrix()
    T[:3, 3] = [100.0, -250.0, 1200.0]
    pose = matrix_to_pose(T)
    assert pose.position.z == 1.2
    assert np.allclose(pose_to_matrix(pose), T)
    assert np.allclose(transform_to_matrix(matrix_to_transform(T)), T)


def test_camera_info_carries_the_intrinsics() -> None:
    K = np.array([[572.4, 0.0, 325.3], [0.0, 573.6, 242.0], [0.0, 0.0, 1.0]])
    info = camera_info(K, 640, 480, Header())
    assert info.k[0] == 572.4
    assert info.p[2] == 325.3
    assert (info.width, info.height) == (640, 480)


def test_clouds_in_metres_and_labelled_fields() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 600.0]])
    cloud = xyz_cloud(points, Header(frame_id="camera"))
    assert np.allclose(cloud_points_mm(cloud), points, atol=1e-3)
    # a cloud laid out like the C++ segment_cloud output: x, y, z float32 + label uint16
    labelled = PointCloud2(height=1, width=2, point_step=14, row_step=28, is_bigendian=False)
    labelled.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="label", offset=12, datatype=PointField.UINT16, count=1),
    ]
    data = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<u2")])
    data["z"] = [0.5, 0.7]
    data["label"] = [3, 9]
    labelled.data = data.tobytes()
    fields = cloud_fields(labelled)
    assert fields["label"].tolist() == [3, 9]
    assert np.allclose(fields["z"], [0.5, 0.7])
