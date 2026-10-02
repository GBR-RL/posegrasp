"""Robot base placement on the table plane."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from posegrasp.scene import base_in_camera, fit_table
from synthetic import HEIGHT, K_LMO, WIDTH

DATA = Path(__file__).resolve().parents[1] / "data"


def _tilted_table_view() -> np.ndarray:
    """Depth of a table plane seen at 45 degrees, 1 m away, with a 60 mm box standing on it."""
    tilt = np.deg2rad(45)
    normal = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])  # towards the camera, "up" in the image
    point = np.array([0.0, 0.0, 1000.0])
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    rays = np.stack(
        [(u - K_LMO[0, 2]) / K_LMO[0, 0], (v - K_LMO[1, 2]) / K_LMO[1, 1], np.ones(u.shape)], -1
    )
    depth = (normal @ point) / (rays @ normal)
    box = (abs(u - 320) < 25) & (abs(v - 240) < 25)
    depth[box] -= 60 / np.sin(tilt) * 0.5  # in front of the table
    return np.where(depth > 0, depth, 0.0)


def test_fit_table_normal_faces_the_camera() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-300, 300, (2000, 2))
    points = np.column_stack([xy, 900 + 0.3 * xy[:, 1]])  # a slanted plane in front
    a, b, c, d = fit_table(points)
    assert d > 0
    assert np.allclose([a, b, c], -np.array([0, -0.3, 1]) / np.linalg.norm([0, -0.3, 1]), atol=1e-3)


def test_base_frame_stands_on_the_table_facing_the_objects() -> None:
    depth = _tilted_table_view()
    T = base_in_camera(depth, K_LMO, reach=500.0)
    R, origin = T[:3, :3], T[:3, 3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)
    up = R[:, 2]
    assert up @ -origin > 0  # the camera is above the table
    # the base is 500 mm from the scene centre, which lies straight ahead along x
    centre = np.array([0.0, 0.0, 1000.0])
    ahead = R.T @ (centre - origin)
    assert ahead[0] == pytest.approx(500.0, abs=25.0)
    assert abs(ahead[1]) < 25.0


@pytest.mark.dataset
@pytest.mark.skipif(not (DATA / "lmo" / "test").exists(), reason="LM-O not downloaded")
def test_lmo_objects_sit_on_the_fitted_table() -> None:
    from posegrasp.data.bop import Dataset

    ds = Dataset(DATA / "lmo")
    frame = ds.frame(2, 3)
    centre = np.mean([g.pose.t for g in frame.gt], axis=0)
    T = base_in_camera(frame.depth_mm(), frame.camera.K, centre=centre)
    in_base = np.linalg.inv(T)
    heights = [(in_base @ np.append(g.pose.t, 1.0))[2] for g in frame.gt]
    # object centres are above the table and within the arm's reach
    assert all(0 < h < 200 for h in heights)
    distances = [np.linalg.norm((in_base @ np.append(g.pose.t, 1.0))[:2]) for g in frame.gt]
    assert all(200 < r < 800 for r in distances)
