"""FreeCAD geometry and reverse-generated, noise-free assembly plans (mm)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from numbers import Real
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .model import POINT_NAMES, apply_transforms


def _positive_fields(config, allow_zero=False):
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError(f"{field.name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class CADGeometry:
    radius: float = 300.0
    nose_length: float = 600.0
    body_length: float = 1200.0
    tail_length: float = 600.0
    exhaust_radius: float = 90.0
    exhaust_length: float = 190.0
    exhaust_offset: float = 150.0
    fin_span: float = 350.0
    fin_root_chord: float = 350.0
    fin_tip_chord: float = 170.0
    fin_sweep: float = 150.0
    fin_thickness: float = 18.0

    def __post_init__(self):
        _positive_fields(self)
        if self.nose_length < self.radius:
            raise ValueError("机头长度必须不小于三段共用半径")
        if not self.exhaust_radius < self.exhaust_offset < self.radius-self.exhaust_radius:
            raise ValueError("出气筒中心偏移必须大于出气筒半径，且两者之和必须小于三段共用半径")
        if self.fin_root_chord > self.tail_length or self.fin_sweep+self.fin_tip_chord > self.fin_root_chord:
            raise ValueError("尾翼根部长度不能超过机尾长度；后掠量与端部长度之和不能超过根部长度")
        if self.fin_thickness >= self.radius:
            raise ValueError("尾翼厚度必须小于三段共用半径")


def world_frame_spec(geometry: CADGeometry) -> dict:
    """A fixed right-handed world frame; these aids are not aircraft geometry."""
    g = geometry
    return dict(name="W", units="mm", origin=[0.0,0.0,0.0], handedness="right",
                origin_description="Fixed external world reference; coincides with B only for the default final pose",
                fixed=True, axes=[
                    dict(name="X_W", direction=[1,0,0], length=g.body_length/2+g.tail_length+g.exhaust_length+150, color="#d74743"),
                    dict(name="Y_W", direction=[0,1,0], length=g.radius+g.fin_span+150, color="#279659"),
                    dict(name="Z_W", direction=[0,0,1], length=g.radius+450, color="#3976d2")])


@dataclass(frozen=True)
class CADMotion:
    """Fixed ideal target distances, not random disturbance magnitudes (mm)."""
    front_gap_mm: float = 250.0
    rear_gap_mm: float = 250.0
    initial_offset_mm: float = 350.0
    final_x_mm: float = 0.0
    final_y_mm: float = 0.0
    final_z_mm: float = 0.0
    final_roll_deg: float = 0.0
    final_pitch_deg: float = 0.0
    final_yaw_deg: float = 0.0

    def __post_init__(self):
        for f in fields(self):
            value=getattr(self,f.name)
            if isinstance(value,bool) or not isinstance(value,Real) or not np.isfinite(value):
                raise ValueError(f"{f.name} 必须是有限数值")
        if self.initial_offset_mm < 0: raise ValueError("initial_offset_mm 必须非负")
        if self.front_gap_mm <= 0 or self.rear_gap_mm <= 0:
            raise ValueError("前、后接口目标间隙必须大于 0")


@dataclass
class CADPlan:
    checkpoints: np.ndarray  # [5,9,3], per-stage perfect positions
    poses: np.ndarray  # [5,3,4,4], assembled CAD frame -> stage world frame
    forward: np.ndarray  # [4,3,4,4], k -> k+1
    reverse: np.ndarray  # [4,3,4,4], k+1 -> k (indexed in forward stage order)


def checkpoint_coordinates(geometry: CADGeometry) -> np.ndarray:
    """Select labelled analytic surface points once in the complete CAD frame."""
    g = geometry
    points = []
    for part in range(3):
        for i, angle in enumerate(np.deg2rad([20, 140, 260])):
            fraction = [0.25, 0.5, 0.75][i]
            if part == 0:
                x = -g.body_length/2-g.nose_length*fraction
                radius = g.radius*np.sqrt(1-fraction**2)
            else:
                length = g.body_length if part == 1 else g.tail_length
                start = -g.body_length/2 if part == 1 else g.body_length/2
                x, radius = start+length*fraction, g.radius
            points.append([x, radius*np.cos(angle), radius*np.sin(angle)])
    return np.asarray(points)


def make_plan(geometry: CADGeometry, motion: CADMotion, seed: int = 0,
              checkpoints: np.ndarray | None = None) -> CADPlan:
    """Construct deterministic ideal targets backwards from the complete CAD.

    Adjustment already achieves the final orientation/centreline, with only
    interface gaps remaining. Perfect rear mating and final readjustment have
    identical targets; their ideal incremental motion is exactly identity.
    Stage 0 is a fixed offset reference placement, not a random disturbance.
    ``seed`` is retained for call compatibility and never affects targets.
    """
    final = checkpoint_coordinates(geometry) if checkpoints is None else np.asarray(checkpoints, float)
    if final.shape != (9, 3) or not np.isfinite(final).all():
        raise ValueError("CAD checkpoints must have shape (9,3) and be finite")
    poses = np.tile(np.eye(4), (5, 3, 1, 1))
    poses[2, 2, 0, 3] = motion.rear_gap_mm
    poses[1] = poses[2]
    poses[1, 0, 0, 3] = -motion.front_gap_mm
    poses[0] = poses[1]
    # A documented reference arrangement before initial positioning; no RNG.
    poses[0, :, :3, 3] += motion.initial_offset_mm*np.array(
        [[-1, 1, 0.25], [0, -1, -0.25], [1, 1, 0.25]])
    points = np.stack([apply_transforms(final, h[:, :3, :3], h[:, :3, 3]) for h in poses])
    forward = poses[1:]@np.linalg.inv(poses[:-1])
    reverse = np.linalg.inv(forward)
    return CADPlan(points, poses, forward, reverse)


def freecad_command() -> list[str]:
    """Use a configured FreeCAD Python or FreeCADCmd; no user GUI session needed."""
    python = os.environ.get("FREECAD_PYTHON")
    if python:
        return [python]
    mac = Path("/Applications/FreeCAD.app/Contents/Resources/bin/python")
    if mac.exists():
        return [str(mac)]
    cmd = os.environ.get("FREECAD_CMD") or shutil.which("FreeCADCmd") or shutil.which("freecadcmd")
    if cmd:
        return [cmd]
    raise RuntimeError("FreeCAD not found. Set FREECAD_PYTHON or FREECAD_CMD to the installed executable.")


def export_cad(directory: Path, geometry: CADGeometry, checkpoints=None, motion=None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "geometry_request.json"
    from .forward import body_frames, final_body_pose
    transform = final_body_pose(motion or CADMotion())
    frames = transform @ body_frames(geometry)
    from .checkpoints import checkpoint_definition
    names, local, assembled = checkpoint_definition(geometry, checkpoints)
    assembled = assembled @ transform[:3,:3].T + transform[:3,3]
    request.write_text(json.dumps(dict(geometry=asdict(geometry), point_names=names, cad_transform=transform.tolist(), local_checkpoints=local.tolist(),
                                      body_frames={name: dict(name=name, final_pose=frames[i].tolist(), origin_description="Axial midpoint of main section; final pose is relative to fixed W") for i,name in enumerate("ABC")},
                                      world_frame=world_frame_spec(geometry),
                                      checkpoints=assembled.tolist())), encoding="utf-8")
    env = os.environ.copy()
    env["AIRCRAFT_CAD_REQUEST"] = str(request.resolve())
    script = Path(__file__).with_name("freecad_build.py")
    result = subprocess.run([*freecad_command(), str(script)], env=env, capture_output=True, text=True, timeout=180)
    (directory / "build.log").write_text(result.stdout+result.stderr, encoding="utf-8")
    manifest = directory / "geometry.json"
    if result.returncode or not manifest.is_file():
        detail = next((line for line in reversed((result.stderr or result.stdout).splitlines()) if line.strip()), "未生成几何文件")
        raise RuntimeError(f"FreeCAD 建模/测点校验失败：{detail}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def cad_model_spec(geometry: CADGeometry, motion: CADMotion, manifest: dict) -> dict:
    return dict(version="3.0", generation_mode="cad_reverse", planning_mode="fixed_targets", units="mm",
                point_names=list(POINT_NAMES), stage_names=["初始/首次调姿前", "调姿后", "前对合后", "后对合后/再调姿前", "再调姿后/最终完整"],
                geometry=asdict(geometry), motion=asdict(motion), cad_engine=manifest["engine"],
                world_frame=manifest.get("world_frame", world_frame_spec(geometry)),
                reference_points=dict(zip(POINT_NAMES, manifest["checkpoints"])),
                final_target_coordinates=dict(zip(POINT_NAMES, manifest["checkpoints"])),
                transform_convention="Column vectors: x_world=R@x_CAD+t; H=[[R,t],[0,0,0,1]]. All parts share the assembled CAD reference frame.",
                target_file="planned_checkpoints.npz (split: [sample,stage,point,xyz]); fixed before noise or faults",
                pose_file="planned_transforms.npz (split_poses: [sample,stage,part,4,4]; split_forward/reverse: [sample,operation,part,4,4])",
                reverse_indexing="reverse[k-1] maps stage k to k-1; generation order is 4 -> 3 -> 2 -> 1 -> 0",
                reverse_rules=["4->3: identity R and zero t; rear mating target equals final target", "3->2: translate C by fixed +X gap; no rotation", "2->1: translate A by fixed -X gap; no rotation", "1->0: fixed reference placement offsets; no random perturbation"],
                forward_rules=["Fit each part to its fixed stage-1 target aligned to the final frame", "Fit A and B CAD poses; move A by G_B inverse(G_A)", "Fit B and C CAD poses; move C by G_B inverse(G_C)", "Fit all checkpoints to final complete CAD geometry, shared action"],
                sampling="No random sampling in the ideal plan. All pose rotations are identity. Random process and measurement errors occur only in forward simulation. No collision constraints.",
                target_policy="All ideal stage targets are fixed across samples. Stage 1 already has final orientation and centreline with axial gaps; stage 3 equals stage 4. Actual final feedback action corrects forward accumulated errors and need not be identity. Executed actions and truth stay in evaluation/.")
