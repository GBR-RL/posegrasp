"""Physics check of the grasps (M5): MuJoCo, CPU, headless.

The Franka Hand of MuJoCo Menagerie floats in the robot base frame of posegrasp.scene (table at
z = 0, z up), driven by a mocap body through a weld, with gravity compensated. The objects of the
image stand at their true poses: the target as a free body, the others fixed, all with collision
geometry from a CoACD convex decomposition of the CAD model. A trial executes one planned grasp:
the scene settles, the open hand approaches from `approach` metres back along the approach axis,
closes, and lifts by `lift` metres. It succeeds when the object rose by at least half the lift
and still touches both fingers.
"""

from __future__ import annotations

import math
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Dataset, Frame
from posegrasp.geometry import FloatArray, load_vertices
from posegrasp.scene import base_in_camera

# The benchmark workflow caches the hand and the decompositions under a key naming this commit and
# the CoACD settings of convex_parts: change the key with them.
MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
MENAGERIE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
HAND_FILES = (
    "hand.xml",
    "LICENSE",
    "assets/hand.stl",
    *(f"assets/hand_{i}.obj" for i in range(5)),
    "assets/finger_0.obj",
    "assets/finger_1.obj",
)
TCP_OFFSET = 0.1034  # hand frame -> between the finger pads, along z (m)
GRASP_FORCE = 70.0  # N on the finger tendon (35 N per finger), as a Franka Hand grasp command
FRICTION = 0.5  # as in the geometric grasp test
DENSITY = 400.0  # kg/m^3: the LM-O objects are light plastic and ceramic figures


def fetch_hand(cache: Path) -> Path:
    """The Franka Hand of MuJoCo Menagerie (Apache-2.0) at a pinned commit; returns hand.xml."""
    folder = cache / "franka_hand"
    for name in HAND_FILES:
        path = folder / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{MENAGERIE}/{MENAGERIE_COMMIT}/franka_emika_panda/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
    return folder / "hand.xml"


def convex_parts(
    mesh_path: Path, cache: Path, *, threshold: float = 0.08, max_parts: int = 12
) -> list[Path]:
    """CoACD convex decomposition of a CAD model (in metres), cached as OBJ files.

    A coarse setting (concavity 0.08, at most 12 hulls) keeps handles and hollows that a single
    convex hull would fill, at a cost the simulation can afford for thousands of trials.
    """
    import coacd
    import trimesh

    folder = cache / "convex" / mesh_path.stem
    done = folder / "done"
    if not done.exists():
        folder.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.apply_scale(0.001)
        coacd.set_log_level("error")
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
            threshold=threshold,
            max_convex_hull=max_parts,
            resolution=1000,
            mcts_iterations=60,
            seed=0,
        )
        for i, (vertices, faces) in enumerate(parts):
            trimesh.Trimesh(vertices, faces).export(folder / f"part_{i:03d}.obj")
        done.write_text(str(len(parts)))
    return sorted(folder.glob("part_*.obj"))


def _quat(R: FloatArray) -> str:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return f"{w:.9f} {x:.9f} {y:.9f} {z:.9f}"


def _pos(t: FloatArray) -> str:
    return " ".join(f"{v:.6f}" for v in t)


@dataclass(frozen=True, slots=True)
class SceneObject:
    obj_id: int
    pose: FloatArray  # 4x4 model -> world, metres
    parts: tuple[Path, ...]
    free: bool = False  # the target


COLOURS = ("0.84 0.47 0.16 1", "0.62 0.62 0.6 1")  # target, other objects


def _required(parent: ET.Element, tag: str) -> ET.Element:
    element = parent.find(tag)
    if element is None:
        raise ValueError(f"hand model without <{tag}>")
    return element


def _add_objects(world: ET.Element, objects: Sequence[SceneObject]) -> None:
    for obj in objects:
        body = ET.SubElement(
            world,
            "body",
            name=f"obj_{obj.obj_id}",
            pos=_pos(obj.pose[:3, 3]),
            quat=_quat(obj.pose[:3, :3]),
        )
        if obj.free:
            ET.SubElement(body, "freejoint", name="target_free")
        rgba = COLOURS[0] if obj.free else COLOURS[1]
        for i in range(len(obj.parts)):
            ET.SubElement(
                body, "geom", attrib={"class": "object"}, mesh=f"o{obj.obj_id}_{i}", rgba=rgba
            )


def _floating_hand(hand: ET.Element, pose: FloatArray) -> ET.Element:
    """The hand body with a free joint, gravity compensated, at `pose`."""
    body = _required(_required(hand, "worldbody"), "body")
    body.set("pos", _pos(pose[:3, 3]))
    body.set("quat", _quat(pose[:3, :3]))
    for part in body.iter("body"):
        part.set("gravcomp", "1")
    body.insert(0, ET.Element("freejoint", name="hand_free"))
    for geom in body.iter("geom"):
        geom.set("friction", f"{FRICTION} 0.005 0.0001")
    return body


def _hand_mechanics(root: ET.Element, hand: ET.Element) -> None:
    """Finger coupling, the grasp actuator, contact exclusions, and the mocap weld."""
    for tag in ("contact", "tendon", "equality", "actuator"):
        element = hand.find(tag)
        if element is not None:
            root.append(element)
    # Menagerie's finger servo is soft (100 N/m: a few newtons on a held object). A stiff servo
    # saturating at GRASP_FORCE squeezes with a constant force, like the hand's grasp action.
    stiffness = 5000.0
    for actuator in root.iter("general"):
        if actuator.get("tendon") == "split":
            actuator.set("gainprm", f"{stiffness * 0.04 / 255:.9f} 0 0")
            actuator.set("biasprm", f"0 {-stiffness} -100")
            actuator.set("forcerange", f"{-GRASP_FORCE} {GRASP_FORCE}")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality, "weld", body1="mocap", body2="hand", solref="0.01 1", solimp="0.95 0.99 0.001"
    )


def build_model_xml(
    hand_xml: Path,
    objects: Sequence[SceneObject],
    *,
    hand_pose: FloatArray,
    camera: tuple[FloatArray, float] | None = None,
) -> str:
    """MJCF of the table, the objects and the free-floating hand at `hand_pose` (4x4, metres).

    camera: optional (pose of an OpenCV camera in the world, vertical field of view in degrees),
    added as the camera "view" for rendering.
    """
    hand = ET.parse(hand_xml).getroot()
    root = ET.Element("mujoco", model="posegrasp pick")
    compiler = _required(hand, "compiler")
    compiler.set("meshdir", str((hand_xml.parent / "assets").resolve()))
    root.append(compiler)
    ET.SubElement(
        root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", impratio="10"
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.5 0.5 0.5")
    defaults = _required(hand, "default")
    object_class = ET.SubElement(defaults, "default", attrib={"class": "object"})
    friction = f"{FRICTION} 0.005 0.0001"
    ET.SubElement(object_class, "geom", type="mesh", friction=friction, density=str(DENSITY))
    root.append(defaults)
    asset = _required(hand, "asset")
    for obj in objects:
        for i, part in enumerate(obj.parts):
            ET.SubElement(asset, "mesh", name=f"o{obj.obj_id}_{i}", file=str(part.resolve()))
    root.append(asset)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 2", dir="0 0 -1", diffuse="0.6 0.6 0.6")
    ET.SubElement(
        world, "geom", name="table", type="plane", size="1.5 1.5 0.01",
        rgba="0.86 0.86 0.84 1", friction=friction,
    )  # fmt: skip
    if camera is not None:
        pose, fovy = camera
        R = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # OpenCV (z forward) -> MuJoCo (-z)
        ET.SubElement(
            world, "camera", name="view", pos=_pos(pose[:3, 3]), quat=_quat(R), fovy=f"{fovy:.3f}"
        )
    _add_objects(world, objects)
    ET.SubElement(
        world, "body", name="mocap", mocap="true", pos=_pos(hand_pose[:3, 3]),
        quat=_quat(hand_pose[:3, :3]),
    )  # fmt: skip
    world.append(_floating_hand(hand, hand_pose))
    _hand_mechanics(root, hand)
    return ET.tostring(root, encoding="unicode")


@dataclass
class TrialResult:
    success: bool
    reason: str  # "ok", "unstable_scene", "dropped", "not_lifted"
    rise: float  # metres the object rose
    frames: list[NDArray[np.uint8]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {"physics_success": self.success, "physics_reason": self.reason, "rise": self.rise}


class PickTrial:
    """One planned grasp executed in MuJoCo."""

    def __init__(
        self,
        xml: str,
        grasp: FloatArray,
        *,
        approach: float = 0.10,
        lift: float = 0.10,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.grasp = grasp
        self.approach = approach
        self.lift = lift
        self.target = int(self.model.joint("target_free").bodyid[0])
        self.fingers = {self.model.body(n).id for n in ("left_finger", "right_finger")}
        self.mocap = self.model.body("mocap").mocapid[0]
        self.actuator = self.model.actuator("actuator8").id

    def hand_pose(self, offset_along_approach: float = 0.0, rise: float = 0.0) -> FloatArray:
        T = self.grasp.copy()
        T[:3, 3] += T[:3, 2] * (offset_along_approach - TCP_OFFSET)
        T[2, 3] += rise
        return T

    def _set_mocap(self, T: FloatArray) -> None:
        from scipy.spatial.transform import Rotation

        self.data.mocap_pos[self.mocap] = T[:3, 3]
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.data.mocap_quat[self.mocap] = [w, x, y, z]

    def _run(
        self, seconds: float, *, start: FloatArray | None = None, end: FloatArray | None = None,
        record: Callable[[], None] | None = None,
    ) -> None:  # fmt: skip
        steps = max(1, round(seconds / self.model.opt.timestep))
        for k in range(steps):
            if start is not None and end is not None:
                s = (k + 1) / steps
                T = start.copy()
                T[:3, 3] = (1 - s) * start[:3, 3] + s * end[:3, 3]
                self._set_mocap(T)
            self.mujoco.mj_step(self.model, self.data)
            if record is not None and k % 20 == 0:
                record()

    def _touches_both_fingers(self) -> bool:
        touched = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = {self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]}
            if self.target in bodies:
                touched |= bodies & self.fingers
        return touched == self.fingers

    def tracking_camera(self, distance: float = 0.7) -> Any:
        """A free camera on the target, looking from the scene camera's direction."""
        self.mujoco.mj_forward(self.model, self.data)
        target = self.data.xpos[self.target]
        forward = target - self.data.cam_xpos[self.model.camera("view").id]
        forward /= np.linalg.norm(forward)
        camera = self.mujoco.MjvCamera()
        camera.lookat[:] = target
        camera.distance = distance
        camera.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
        camera.elevation = math.degrees(math.asin(forward[2]))
        return camera

    def run(self, *, render: Any = None, camera: Any = "view") -> TrialResult:
        """render: an optional mujoco.Renderer recording frames from `camera` (a camera name or
        a mujoco.MjvCamera)."""
        frames: list[NDArray[np.uint8]] = []

        def record() -> None:
            if render is not None:
                render.update_scene(self.data, camera=camera)
                frames.append(render.render().copy())

        self.data.ctrl[self.actuator] = 255  # open
        pre = self.hand_pose(-self.approach)
        self._set_mocap(pre)
        self.mujoco.mj_forward(self.model, self.data)
        start = self.data.xpos[self.target].copy()
        self._run(0.4, record=record)
        settled = self.data.xpos[self.target].copy()
        if np.linalg.norm(settled - start) > 0.01:
            return TrialResult(False, "unstable_scene", 0.0, frames)
        at_grasp = self.hand_pose()
        self._run(1.0, start=pre, end=at_grasp, record=record)
        self.data.ctrl[self.actuator] = 0  # close
        self._run(0.8, record=record)
        self._run(1.0, start=at_grasp, end=self.hand_pose(rise=self.lift), record=record)
        self._run(0.4, record=record)
        rise = float(self.data.xpos[self.target][2] - settled[2])
        if rise < self.lift / 2:
            return TrialResult(False, "not_lifted", rise, frames)
        if not self._touches_both_fingers():
            return TrialResult(False, "dropped", rise, frames)
        return TrialResult(True, "ok", rise, frames)


@dataclass
class SceneSetup:
    """The objects of one image in the robot base frame, ready for trials."""

    frame: Frame
    world_from_camera: FloatArray  # 4x4, mm
    objects: dict[int, tuple[FloatArray, tuple[Path, ...]]]  # obj_id -> (pose in world m, parts)

    @staticmethod
    def of(dataset: Dataset, frame: Frame, cache: Path) -> SceneSetup:
        models = dataset.models()
        centre = np.mean([g.pose.t for g in frame.gt], axis=0)
        vertices = [
            load_vertices(models[g.obj_id].eval_mesh) @ g.pose.R.T + g.pose.t for g in frame.gt
        ]
        base = base_in_camera(frame.depth_mm(), frame.camera.K, centre=centre, objects=vertices)
        world_from_camera: FloatArray = np.asarray(np.linalg.inv(base), dtype=np.float64)
        objects = {}
        for g in frame.gt:
            T = world_from_camera @ g.pose.matrix()
            T[:3, 3] /= 1000.0
            parts = tuple(convex_parts(models[g.obj_id].mesh_path, cache))
            objects[g.obj_id] = (T, parts)
        return SceneSetup(frame, world_from_camera, objects)

    def camera(self) -> tuple[FloatArray, float]:
        pose = self.world_from_camera.copy()
        pose[:3, 3] /= 1000.0
        K, h = self.frame.camera.K, self.frame.camera.height
        return pose, math.degrees(2 * math.atan(h / 2 / K[1, 1]))

    def trial(self, target: int, grasp_cam: FloatArray, hand_xml: Path) -> PickTrial:
        """grasp_cam: grasp -> camera, millimetres."""
        grasp = self.world_from_camera @ grasp_cam
        grasp[:3, 3] /= 1000.0
        objects = [
            SceneObject(obj_id, pose, parts, free=obj_id == target)
            for obj_id, (pose, parts) in self.objects.items()
        ]
        hand = grasp.copy()
        hand[:3, 3] += hand[:3, 2] * (-0.10 - TCP_OFFSET)
        xml = build_model_xml(hand_xml, objects, hand_pose=hand, camera=self.camera())
        return PickTrial(xml, grasp)


def simulate_grasps(
    dataset: Dataset, rows: Sequence[dict[str, Any]], *, cache: Path
) -> Iterator[dict[str, Any]]:
    """Physics outcome of every planned grasp in `rows` (grasp rows of `posegrasp pick`)."""
    hand_xml = fetch_hand(cache)
    by_image: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in rows:
        by_image.setdefault((r["scene_id"], r["im_id"]), []).append(r)
    for (scene_id, im_id), image_rows in by_image.items():
        setup = SceneSetup.of(dataset, dataset.frame(scene_id, im_id), cache)
        for row in image_rows:
            out = {k: row[k] for k in ("scene_id", "im_id", "obj_id", "success", "reason")}
            if row.get("grasp") is None:
                out.update(physics_success=False, physics_reason="no_grasp", rise=0.0)
            else:
                trial = setup.trial(row["obj_id"], np.asarray(row["grasp"]), hand_xml)
                out.update(trial.run().row())
            yield out
