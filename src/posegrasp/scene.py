"""Where the robot stands: a base frame on the table plane, seen from the camera.

LM-O gives the camera's view of a table but no robot. For the ROS 2 pick, the robot base is placed
on the table plane, `reach` millimetres from the centre of the objects, on the camera's side,
facing them: x points from the base towards the objects, z up (the table normal).
"""

from __future__ import annotations

import numpy as np

from posegrasp.geometry import FloatArray, backproject


def fit_table(points: FloatArray, *, threshold: float = 8.0, seed: int = 0) -> FloatArray:
    """Plane (a, b, c, d) with unit normal towards the camera (the origin): n·p + d = 0, d > 0."""
    import open3d as o3d

    from posegrasp.geometry import to_open3d

    o3d.utility.random.seed(seed)
    plane, _ = to_open3d(points).segment_plane(threshold, 3, 1000)  # type: ignore[attr-defined]
    plane = np.asarray(plane, dtype=float)
    plane /= np.linalg.norm(plane[:3])
    return -plane if plane[3] < 0 else plane


def base_in_camera(
    depth_mm: FloatArray,
    K: FloatArray,
    *,
    centre: FloatArray | None = None,
    reach: float = 500.0,
) -> FloatArray:
    """Pose (4x4, base -> camera, mm) of a robot base frame on the table.

    centre: camera-frame point the robot should face (the scene player passes the middle of the
    objects); by default the median of the depth points standing above the table.
    """
    mask = np.zeros(depth_mm.shape, dtype=bool)
    mask[::4, ::4] = True
    points = backproject(depth_mm, mask, K)
    plane = fit_table(points)
    n, d = plane[:3], plane[3]
    height = points @ n + d
    if centre is None:
        objects = points[height > 15.0]
        centre = np.median(objects if len(objects) > 50 else points, axis=0)
    centre_on_table = centre - (centre @ n + d) * n
    camera_on_table = -d * n  # the origin projected onto the plane
    x = centre_on_table - camera_on_table
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, n
    T[:3, 3] = centre_on_table - reach * x
    return T
