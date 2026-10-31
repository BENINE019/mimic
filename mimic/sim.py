#!/usr/bin/env python3
"""Stage 7: replay a plan with a simulated arm.

Builds a MuJoCo scene from a clip's table-plane data (one simple block per
object, sized from its mask, a cylinder for containers), places a Franka Panda
at the edge of the table, and executes a list of pick-and-place steps with
inverse kinematics. Grasping is scripted: when the gripper reaches an object
the object follows the gripper until release. This is kinematic replay, not a
physics-based grasp, and the README says so.

Outputs in out/<clip>/sim/<name>/:
  episode.mp4        rendered video (640x480, 30 fps)
  result.json        final object positions (cm, table frame) and per-step status

Usage (from the repo root):
  python -m mimic.sim clips/c01.mp4                 # copy baseline: replay the demonstrated moves
  python -m mimic.sim clips/c01.mp4 --no-render     # kinematics only, for quick checks

Requires: pip install mujoco, and MuJoCo Menagerie cloned at ~/mujoco_menagerie
(or set MENAGERIE=/path/to/mujoco_menagerie).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import mujoco

MENAGERIE = Path(os.environ.get("MENAGERIE", Path.home() / "mujoco_menagerie"))
PANDA_XML = MENAGERIE / "franka_emika_panda" / "panda.xml"

TCP_OFFSET = 0.1034           # hand frame to fingertip centre, metres
HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FPS = 30
APPROACH = 0.12               # hover height above pick and place, metres
BASE_GAP = 0.35               # arm base this far beyond the nearest object (+y side)
REACH = 0.72                  # farthest object the arm can comfortably pick, metres
CAMERA_HEIGHT = 0.95
CAMERA_BACK = 1.05           # camera this far from the objects, on the side opposite the arm

HEIGHTS_CM = [  # (keyword, height cm, is container)
    ("cup", 8.0, True), ("mug", 8.0, True), ("container", 8.0, True), ("bowl", 6.0, True),
    ("basket", 10.0, True), ("box", 6.0, True), ("jar", 9.0, True), ("bin", 10.0, True), ("tray", 2.0, True),
    ("pencil", 1.0, False), ("pen", 1.2, False), ("marker", 1.5, False), ("eraser", 2.0, False),
    ("case", 2.5, False), ("wallet", 2.0, False), ("scissors", 1.0, False), ("glasses", 3.0, False),
    ("key", 0.6, False), ("remote", 2.0, False), ("phone", 0.8, False), ("block", 4.0, False),
]
FOOTPRINTS_CM = [  # (keyword, long side, short side)
    ("pencil", 15.0, 1.2), ("marker", 13.0, 1.8), ("pen", 14.0, 1.2), ("eraser", 5.0, 2.5), ("case", 10.0, 6.0),
    ("wallet", 11.0, 9.0), ("scissors", 16.0, 6.0), ("glasses", 14.0, 4.5), ("key", 6.0, 2.5),
    ("remote", 16.0, 4.0), ("phone", 15.0, 7.0), ("block", 4.0, 4.0),
]
RADII_CM = [("cup", 4.0), ("mug", 4.0), ("container", 4.5), ("bowl", 7.0), ("basket", 12.0), ("box", 6.0),
            ("jar", 4.0), ("bin", 10.0), ("tray", 10.0)]
COLORS = {
    "red": (0.85, 0.15, 0.15), "pink": (0.95, 0.55, 0.7), "orange": (0.95, 0.5, 0.1), "brown": (0.5, 0.3, 0.15),
    "yellow": (0.95, 0.85, 0.2), "green": (0.2, 0.6, 0.25), "blue": (0.2, 0.4, 0.85), "purple": (0.5, 0.25, 0.7),
    "black": (0.12, 0.12, 0.12), "white": (0.92, 0.92, 0.9), "gray": (0.55, 0.55, 0.55), "grey": (0.55, 0.55, 0.55),
    "silver": (0.75, 0.75, 0.78), "cream": (0.95, 0.9, 0.75),
}


def UPRIGHT_Q(key: str, scene) -> list[float]:
    """Quaternion standing the object's long axis vertical."""
    o = scene.objects[key]
    if o["w_cm"] >= o["d_cm"]:      # long axis along x: rotate 90 deg about y
        return [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
    return [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]   # long axis along y: rotate about x


def spec_of(concept: str) -> tuple[float, bool]:
    for word, h, cont in HEIGHTS_CM:
        if word in concept:
            return h, cont
    return 2.0, False


def color_of(concept: str) -> tuple[float, float, float]:
    for word, rgb in COLORS.items():
        if word in concept:
            return rgb
    return (0.6, 0.6, 0.6)


def footprint_cm(bbox_px, H) -> tuple[float, float]:
    """Width and depth in cm of an image box projected onto the table."""
    import cv2
    x0, y0, x1, y1 = bbox_px
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(pts, np.array(H, np.float64)).reshape(-1, 2)
    ext = w.max(axis=0) - w.min(axis=0)
    return float(np.clip(ext[0], 2.0, 25.0)), float(np.clip(ext[1], 2.0, 25.0))


def scene_objects(table_tracks: dict, events: dict) -> list[dict]:
    """Objects to simulate: everything stage 5 kept, with size, height and colour."""
    roles = {o["key"]: o for o in events["objects"]}
    H = table_tracks["table"]["H_px_to_cm"]
    out = []
    for o in table_tracks["objects"]:
        if o["key"] not in roles:
            continue
        r = roles[o["key"]]
        first = o["frames"][0]
        bw, bd = footprint_cm(first["bbox"], H)      # projected box, used only for orientation and as fallback
        h, cont = spec_of(o["concept"])
        w, d = bw, bd
        for word, long_cm, short_cm in FOOTPRINTS_CM:
            if word in o["concept"]:
                w, d = (long_cm, short_cm) if bw >= bd else (short_cm, long_cm)
                break
        for word, radius in RADII_CM:
            if cont and word in o["concept"]:
                w = d = 2 * radius
                break
        out.append({
            "key": o["key"], "concept": o["concept"], "role": r["role"], "attached_to": r["attached_to"],
            "xy": list(r["start_xy"]), "w_cm": w, "d_cm": d, "h_cm": h, "container": cont,
            "rgb": color_of(o["concept"]),
        })
    return out


def fit_to_reach(objects: list[dict], steps: list[list[dict]] | None = None) -> float:
    """Shrink the layout about its centroid until every object is within the arm's reach.

    Edits object xy in place (and the pick/place positions in any step lists given),
    so plans and success checks stay consistent. Returns the scale factor used.
    """
    xy = np.array([o["xy"] for o in objects], float)
    c = xy.mean(axis=0)
    base = np.array([c[0], xy[:, 1].max() + BASE_GAP * 100])
    far = np.linalg.norm(xy - base, axis=1).max() / 100.0
    if far <= REACH:
        return 1.0
    k = 1.0
    for _ in range(20):
        k *= 0.95
        sxy = c + (xy - c) * k
        base = np.array([c[0], sxy[:, 1].max() + BASE_GAP * 100])
        if np.linalg.norm(sxy - base, axis=1).max() / 100.0 <= REACH:
            break
    for o, p in zip(objects, c + (xy - c) * k):
        o["xy"] = [round(float(p[0]), 2), round(float(p[1]), 2)]
    for lst in steps or []:
        for st in lst:
            for f in ("pick_xy", "place_xy"):
                if st.get(f) is not None:
                    p = c + (np.array(st[f], float) - c) * k
                    st[f] = [round(float(p[0]), 2), round(float(p[1]), 2)]
    return k


def plan_from_events(events: dict, objects: list[dict]) -> list[dict]:
    """The copy baseline: reproduce each demonstrated move at its recorded place position."""
    by_key = {o["key"]: o for o in objects}
    steps = []
    for e in events["events"]:
        if e["object"] not in by_key:
            continue
        steps.append({"object": e["object"], "pick_xy": list(e["pick_xy"]), "place_xy": list(e["place_xy"]),
                      "into": e.get("in"), "onto": e.get("on"), "carried": e.get("carried", [])})
    return steps


class Scene:
    def __init__(self, objects: list[dict], render: bool = True, width: int = 640, height: int = 480):
        if not PANDA_XML.exists():
            raise FileNotFoundError(f"{PANDA_XML} not found; clone mujoco_menagerie or set MENAGERIE")
        self.objects = {o["key"]: dict(o) for o in objects}
        for o in self.objects.values():
            o["long_m"] = max(o["w_cm"], o["d_cm"]) / 100.0
            o["short_m"] = min(o["w_cm"], o["d_cm"]) / 100.0
            o["yaw"] = 0.0 if o["w_cm"] >= o["d_cm"] else np.pi / 2
        xy = np.array([o["xy"] for o in objects]) / 100.0
        self.centroid = xy.mean(axis=0)
        self.base_xy = np.array([self.centroid[0], xy[:, 1].max() + BASE_GAP])
        self.spec = self._build()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.arm_qadr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.arm_dofadr = [self.model.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.finger_qadr = [self.model.joint("finger_joint1").qposadr[0], self.model.joint("finger_joint2").qposadr[0]]
        self.tcp = self.model.site("tcp").id
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.cam = self.model.camera("phone").id
        self.reset()

    def _build(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(PANDA_XML))
        spec.modelname = "mimic"
        # Robot base: at the table edge, facing the objects (its +x toward -y).
        base = spec.body("link0")
        base.pos = [self.base_xy[0], self.base_xy[1], 0.0]
        base.quat = [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]
        hand = spec.body("hand")
        hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0.005, 0.005], rgba=[0, 1, 0, 0.0])

        wb = spec.worldbody
        wb.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
        wb.add_light(pos=[self.centroid[0] - 0.5, self.centroid[1] - 0.5, 1.0], dir=[0.4, 0.4, -1], diffuse=[0.4, 0.4, 0.4])
        # Table top at z = 0, generous size, centred between arm and objects.
        tcx, tcy = self.centroid[0], (self.centroid[1] + self.base_xy[1]) / 2
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[tcx, tcy, -0.02],
                    size=[0.8, 0.8, 0.02], rgba=[0.82, 0.76, 0.65, 1])
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, -0.75],
                    size=[3, 3, 0.1], rgba=[0.35, 0.37, 0.4, 1])
        # Marker outline for orientation (origin of the table frame).
        wb.add_geom(name="marker", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.0005],
                    size=[0.0475, 0.0475, 0.0005], rgba=[0.05, 0.05, 0.05, 1])

        for key, o in self.objects.items():
            h = o["h_cm"] / 100.0
            b = wb.add_body(name=key, pos=[o["xy"][0] / 100.0, o["xy"][1] / 100.0, h / 2])
            b.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name=f"free:{key}")
            rgba = [*o["rgb"], 1.0]
            if o["container"]:
                r = min(o["w_cm"], o["d_cm"]) / 200.0
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0], rgba=rgba)
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r * 0.8, h / 2 + 0.002, 0],
                           rgba=[rgba[0] * 0.6, rgba[1] * 0.6, rgba[2] * 0.6, 1], pos=[0, 0, 0.004])
            else:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[o["w_cm"] / 200.0, o["d_cm"] / 200.0, h / 2], rgba=rgba)
        # A phone-like camera on the far side of the table, looking at the objects.
        focus = wb.add_body(name="focus", pos=[self.centroid[0], self.centroid[1], 0.05])
        focus.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], rgba=[0, 0, 0, 0])
        wb.add_camera(name="phone", pos=[self.centroid[0], self.centroid[1] - CAMERA_BACK, CAMERA_HEIGHT],
                      mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="focus", fovy=42)
        return spec

    # ---- state helpers -------------------------------------------------------
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_Q
        self.set_fingers(0.04)
        for key, o in self.objects.items():
            self.set_object(key, o["xy"][0] / 100.0, o["xy"][1] / 100.0, o["h_cm"] / 200.0)
        mujoco.mj_forward(self.model, self.data)

    def set_fingers(self, width: float) -> None:
        self.data.qpos[self.finger_qadr] = [width, width]

    def set_object(self, key: str, x: float, y: float, z: float) -> None:
        adr = self.model.joint(f"free:{key}").qposadr[0]
        self.data.qpos[adr:adr + 3] = [x, y, z]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

    def object_pos(self, key: str) -> np.ndarray:
        adr = self.model.joint(f"free:{key}").qposadr[0]
        return self.data.qpos[adr:adr + 3].copy()

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp].copy()

    # ---- inverse kinematics --------------------------------------------------
    def ik(self, target: np.ndarray, iters: int = 80, tol: float = 1e-3) -> float:
        """Damped least squares: TCP at target, gripper pointing straight down, yaw free."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        ids = [self.model.joint(j).id for j in ARM_JOINTS]
        lo, hi = self.model.jnt_range[ids, 0], self.model.jnt_range[ids, 1]
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            ep = target - self.data.site_xpos[self.tcp]
            R = self.data.site_xmat[self.tcp].reshape(3, 3)
            er = np.cross(R[:, 2], down)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 0.02:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]])
            e = np.concatenate([ep, er * 0.5])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.005 * np.eye(6), e)
            dq = np.clip(dq, -0.15, 0.15)
            self.data.qpos[self.arm_qadr] = np.clip(self.data.qpos[self.arm_qadr] + dq, lo, hi)
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(target - self.data.site_xpos[self.tcp]))

    def align_yaw(self, yaw: float) -> None:
        """Turn the wrist (joint7) so the fingers close across the given direction (radians, world)."""
        j7 = self.model.joint("joint7")
        qadr = j7.qposadr[0]
        mujoco.mj_forward(self.model, self.data)
        R = self.data.site_xmat[self.tcp].reshape(3, 3)
        # finger closing direction in the hand frame is +y; measure its world yaw
        cur = np.arctan2(R[1, 1], R[0, 1])
        delta = (yaw - cur + np.pi / 2) % np.pi - np.pi / 2      # fingers are symmetric: mod 180 degrees
        lo, hi = self.model.jnt_range[j7.id]
        self.data.qpos[qadr] = np.clip(self.data.qpos[qadr] - delta, lo, hi)
        mujoco.mj_forward(self.model, self.data)

    def set_object_pose(self, key: str, x: float, y: float, z: float, quat) -> None:
        adr = self.model.joint(f"free:{key}").qposadr[0]
        self.data.qpos[adr:adr + 3] = [x, y, z]
        self.data.qpos[adr + 3:adr + 7] = quat

    # ---- execution -----------------------------------------------------------
    def execute(self, steps: list[dict], out_dir: Path | None = None) -> dict:
        frames = []
        traj = []
        log = []
        holding: list[str] = []
        hold_offsets: dict[str, np.ndarray] = {}

        def record():
            tp = self.tcp_pos()
            traj.append({"q": [round(float(v), 4) for v in self.data.qpos[self.arm_qadr]],
                         "gripper": round(float(self.data.qpos[self.finger_qadr[0]]), 4),
                         "tcp": [round(float(v), 4) for v in tp],
                         "holding": list(holding)})
            if self.renderer is not None:
                self.renderer.update_scene(self.data, camera=self.cam)
                frames.append(self.renderer.render().copy())

        state = {"yaw": None, "upright": None, "tilt": 1.0}

        def slerp(q0, q1, t):
            q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
            d = np.dot(q0, q1)
            if d < 0:
                q1, d = -q1, -d
            if d > 0.9995:
                q = q0 + t * (q1 - q0)
                return (q / np.linalg.norm(q)).tolist()
            th = np.arccos(d)
            return ((np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)).tolist()

        def goto(target: np.ndarray, seconds: float, tilt_to: float | None = None) -> float:
            start = self.tcp_pos()
            n = max(2, int(seconds * FPS))
            worst = 0.0
            tilt_from = state["tilt"]
            for i in range(1, n + 1):
                p = start + (target - start) * (i / n)
                worst = max(worst, self.ik(p))
                if state["yaw"] is not None:
                    self.align_yaw(state["yaw"])
                if tilt_to is not None:
                    state["tilt"] = tilt_from + (tilt_to - tilt_from) * (i / n)
                for k in holding:
                    tp = self.tcp_pos()
                    if k == state["upright"]:
                        # tilt from flat (as grasped) to hanging upright below the fingertips
                        t = state["tilt"]
                        q = slerp([1, 0, 0, 0], UPRIGHT_Q(k, self), t)
                        drop = (self.objects[k]["long_m"] / 2 - 0.01) * t
                        self.set_object_pose(k, tp[0], tp[1], tp[2] - drop, q)
                    else:
                        self.set_object(k, *(tp + hold_offsets[k]))
                mujoco.mj_forward(self.model, self.data)
                record()
            return worst

        for _ in range(FPS // 2):
            record()
        for i, st in enumerate(steps):
            key = st["object"]
            if key not in self.objects:
                log.append({"step": i, "object": key, "status": "unknown object"})
                continue
            o = self.objects[key]
            pick = self.object_pos(key)
            grasp_z = max(o["h_cm"] / 100.0, 0.012)
            if st.get("pick_xy") is not None:  # copy baseline: go where the demonstration grasped
                pick = np.array([st["pick_xy"][0] / 100.0, st["pick_xy"][1] / 100.0, pick[2]])
            pick_t = np.array([pick[0], pick[1], grasp_z])
            px, py = np.array(st["place_xy"]) / 100.0
            if st.get("into") and st["into"] in self.objects:
                c = self.objects[st["into"]]
                place_z_obj = 0.004 + o["h_cm"] / 200.0
                release_z = c["h_cm"] / 100.0 + 0.03
            elif st.get("onto") and st["onto"] in self.objects:
                b = self.objects[st["onto"]]
                place_z_obj = b["h_cm"] / 100.0 + o["h_cm"] / 200.0
                release_z = b["h_cm"] / 100.0 + grasp_z
            else:
                place_z_obj = o["h_cm"] / 200.0
                release_z = grasp_z
            place_t = np.array([px, py, release_z])

            state["yaw"] = o["yaw"]          # close the fingers across the object's short side
            state["upright"] = None
            e1 = goto(pick_t + [0, 0, APPROACH], 1.0)
            e2 = goto(pick_t, 0.5)
            if e2 > 0.02:
                log.append({"step": i, "object": key, "status": f"unreachable pick ({e2 * 100:.1f} cm off)"})
                goto(pick_t + [0, 0, APPROACH], 0.5)
                continue
            gap = float(np.linalg.norm(self.object_pos(key)[:2] - self.tcp_pos()[:2]))
            if gap > 0.03:
                self.set_fingers(0.004)
                for _ in range(FPS // 3):
                    record()
                self.set_fingers(0.04)
                log.append({"step": i, "object": key, "status": f"missed: nothing at the pick position ({gap * 100:.0f} cm away)"})
                goto(pick_t + [0, 0, APPROACH], 0.5)
                continue
            self.set_fingers(max(0.004, o["short_m"] / 2))
            holding = [key] + [k for k in st.get("carried", []) if k in self.objects]
            for k in holding:
                hold_offsets[k] = self.object_pos(k) - self.tcp_pos()
            for _ in range(FPS // 6):
                record()
            into = st.get("into") if st.get("into") in self.objects else None
            if into is not None and o["long_m"] > 2.5 * o["short_m"]:
                state["upright"] = key        # long thin thing into a cup: carry it standing up
                state["tilt"] = 0.0
                goto(pick_t + [0, 0, APPROACH + o["long_m"] / 2], 0.8, tilt_to=1.0)
                release_z = max(release_z, o["long_m"] + 0.02)
                place_t = np.array([px, py, release_z])
            else:
                goto(pick_t + [0, 0, APPROACH], 0.6)
            goto(place_t + [0, 0, APPROACH], 1.0)
            e3 = goto(place_t, 0.5)
            # release: objects settle onto the table / into the container / onto the base
            for k in holding:
                p = self.object_pos(k)
                if k == state["upright"]:
                    self.set_object_pose(k, px, py, 0.004 + self.objects[k]["long_m"] / 2, UPRIGHT_Q(k, self))
                else:
                    z = place_z_obj if k == key else self.objects[k]["h_cm"] / 200.0
                    self.set_object(k, p[0], p[1], z)
            holding = []
            state["upright"] = None
            self.set_fingers(0.04)
            for _ in range(FPS // 6):
                record()
            goto(place_t + [0, 0, APPROACH], 0.6)
            status = "ok" if e3 <= 0.02 else f"place {e3 * 100:.1f} cm off"
            log.append({"step": i, "object": key, "status": status, "place_error_cm": round(e3 * 100, 2)})
        for _ in range(FPS // 2):
            record()

        final = {k: [round(v, 2) for v in (self.object_pos(k)[:2] * 100)] for k in self.objects}
        result = {"steps": log, "final_xy_cm": final, "trajectory": traj,
                  "ok": all(s["status"] == "ok" for s in log) and len(log) == len(steps)}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            if frames:
                import cv2
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(str(out_dir / "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                for f in frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
                result["video"] = str(out_dir / "episode.mp4")
                result["frames"] = len(frames)
            (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def load_clip(out_dir: Path) -> tuple[dict, dict]:
    tt = json.loads((out_dir / "table_tracks.json").read_text())
    ev = json.loads((out_dir / "events.json").read_text())
    return tt, ev


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 7: replay a plan with a simulated Panda")
    p.add_argument("video", type=Path)
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    p.add_argument("--name", default="replay", help="sub-folder name under out/<clip>/sim/")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    out_dir = args.output or Path("out") / args.video.stem
    tt, ev = load_clip(out_dir)
    objects = scene_objects(tt, ev)
    steps = plan_from_events(ev, objects)
    print(f"{len(objects)} objects, {len(steps)} steps (copy baseline)")
    scene = Scene(objects, render=not args.no_render)
    res = scene.execute(steps, out_dir / "sim" / args.name)
    for s in res["steps"]:
        print(f"  step {s['step']}: {s['object']:<24} {s['status']}")
    print(f"ok={res['ok']}" + (f", video {res['video']} ({res['frames']} frames)" if "video" in res else ""))


if __name__ == "__main__":
    main()
