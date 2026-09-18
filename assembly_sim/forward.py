"""CAD-local rigid bodies, fixed targets, and observation-driven forward execution.

Column-vector convention, mm and degrees. Errors act on the left in W.
No contact/deformation model is implied by the stage labels.
"""
from dataclasses import dataclass, fields
import numpy as np
from .cad import CADGeometry, CADMotion, checkpoint_coordinates
from .model import rigid_fit, POINT_NAMES
from .checkpoints import point_groups, validate_local


@dataclass(frozen=True)
class ForwardSettings:
    initial_rotation_deg: float = 25.0  # bounded rotation-vector norm
    initial_translation_mm: float = 350.0  # per-axis uniform offset around stage 1
    process_rotation_deg: float = 0.002  # per-axis rotation-vector Gaussian std

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
        if self.initial_rotation_deg > 180:
            raise ValueError("initial_rotation_deg must be <= 180")


def rigid(rotation_vector, translation):
    w = np.asarray(rotation_vector, float)
    theta = np.linalg.norm(w)
    h = np.eye(4)
    if theta > 1e-15:
        x, y, z = w/theta
        k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        h[:3,:3] = np.eye(3)+np.sin(theta)*k+(1-np.cos(theta))*(k@k)
    h[:3,3] = translation
    return h


def fit_pose(source, target):
    r, t = rigid_fit(source, target)
    h = np.eye(4)
    h[:3,:3], h[:3,3] = r, t
    return h


def world_points(local, poses, point_names=POINT_NAMES):
    local = np.asarray(local, float)
    if local.shape != (len(point_names), 3):
        raise ValueError("Point coordinates do not match point_names")
    result = np.empty_like(local)
    for b, indices in enumerate(point_groups(point_names)):
        result[indices] = local[indices] @ poses[b,:3,:3].T + poses[b,:3,3]
    return result


def body_frames(geometry):
    g = geometry
    final = np.tile(np.eye(4), (3,1,1))
    final[:,0,3] = [-g.body_length/2-g.nose_length/2, 0, g.body_length/2+g.tail_length/2]
    return final


def final_body_pose(motion):
    """Editable final B pose: Rz(yaw) Ry(pitch) Rx(roll), column vectors."""
    rx=rigid([np.deg2rad(motion.final_roll_deg),0,0],[0,0,0])
    ry=rigid([0,np.deg2rad(motion.final_pitch_deg),0],[0,0,0])
    rz=rigid([0,0,np.deg2rad(motion.final_yaw_deg)],[0,0,0])
    result=rz @ ry @ rx
    result[:3,3]=[motion.final_x_mm,motion.final_y_mm,motion.final_z_mm]
    return result


def target_definition(geometry=CADGeometry(), motion=CADMotion(), checkpoints=None, point_names=POINT_NAMES):
    """Four targets, stages 1..4 only. Stage 0 is a sampled condition, not a target."""
    transform = final_body_pose(motion)
    final = transform @ body_frames(geometry)
    assembled = checkpoint_coordinates(geometry) @ transform[:3,:3].T + transform[:3,3] if checkpoints is None else np.asarray(checkpoints, float)
    if assembled.shape != (len(point_names),3) or not np.isfinite(assembled).all():
        raise ValueError("CAD checkpoints must match point_names and be finite")
    local = assembled.copy()
    for b, indices in enumerate(point_groups(point_names)):
        local[indices] = (local[indices]-final[b,:3,3]) @ final[b,:3,:3]
    validate_local(point_names, local)
    targets = np.tile(final, (4,1,1,1))
    targets[0,0,:3,3] -= transform[:3,0]*motion.front_gap_mm
    targets[:2,2,:3,3] += transform[:3,0]*motion.rear_gap_mm
    points = np.stack([world_points(local, h, point_names) for h in targets])
    return local, targets, points


def commands(observed, stage, targets, controller_policy="body_relative", point_names=POINT_NAMES):
    """Mating follows observed B; fixed design targets remain quality references."""
    if controller_policy not in ("body_relative", "fixed_world"):
        raise ValueError("Unknown controller_policy")
    groups = point_groups(point_names)
    result = np.tile(np.eye(4), (3,1,1))
    if stage == 4:
        result[:] = fit_pose(observed, targets[3])
    elif stage in (2,3) and controller_policy == "body_relative":
        # G_B maps ideal assembled-world geometry to the observed body pose.
        # This is equivalent to T_B_est D_Bm inverse(T_m_est) in local frames.
        part = 0 if stage == 2 else 2
        s = groups[part]
        g_b = fit_pose(targets[3,groups[1]],observed[groups[1]])
        moving_target = targets[3,s] @ g_b[:3,:3].T + g_b[:3,3]
        result[part] = fit_pose(observed[s],moving_target)
    else:
        for part in active_parts(stage):
            s = groups[part]
            result[part] = fit_pose(observed[s],targets[stage-1,s])
    return result


def active_parts(stage):
    return {1:(0,1,2), 2:(0,), 3:(2,), 4:(0,1,2)}[stage]


def fault_choices(stage, part):
    """Respect explicit locations; sample only physically active combinations."""
    from numbers import Integral
    if stage is not None and (isinstance(stage,bool) or not isinstance(stage,Integral) or stage not in (1,2,3,4)):
        raise ValueError("fault_stage must be 1..4 or null")
    if part not in (None,"A","B","C","ABC"):
        raise ValueError("fault_part must be A/B/C/ABC or null")
    options={1:("A","B","C"),2:("A",),3:("C",),4:("ABC",)}
    valid={k:tuple(p for p in parts if part is None or p==part) for k,parts in options.items() if stage is None or k==stage}
    valid={k:p for k,p in valid.items() if p}
    if not valid:
        raise ValueError("所选阶段与故障部件不相容：1=A/B/C，2=A，3=C，4=ABC；不会自动改写你的选择。")
    return valid


def action_id(stage, part):
    return f"h1_{part}" if stage == 1 else {2:"h2_front",3:"h3_rear",4:"h4_final"}[stage]


@dataclass
class ForwardTrajectory:
    observed: np.ndarray
    true: np.ndarray
    poses: np.ndarray
    estimated_poses: np.ndarray
    commanded: np.ndarray
    executed: np.ndarray
    process_errors: np.ndarray
    measurement_errors: np.ndarray


def simulate_forward(local, target_poses, target_points, model, settings, seed, fault=None, point_names=POINT_NAMES):
    # Separate streams keep initial conditions and sensor noise paired when
    # process parameters/fault injection change.
    initial_rng, process_rng, sensor_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    groups = point_groups(point_names)
    validate_local(point_names, local)
    poses = np.tile(np.eye(4), (5,3,1,1))
    for b in range(3):
        axis = initial_rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = initial_rng.uniform(-settings.initial_rotation_deg,settings.initial_rotation_deg)
        offset = initial_rng.uniform(-settings.initial_translation_mm,settings.initial_translation_mm,3)
        poses[0,b] = rigid(axis*np.deg2rad(angle),target_poses[0,b,:3,3]+offset)
        poses[0,b,:3,:3] = poses[0,b,:3,:3] @ target_poses[0,b,:3,:3]
    measurement = sensor_rng.normal(0, model.measurement_sigma, (5,len(point_names),3))
    rotation_noise = process_rng.normal(0, np.deg2rad(settings.process_rotation_deg), (4,3,3))
    translation_noise = process_rng.normal(0, model.process_sigma, (4,3,3))
    truth = np.empty((5,len(point_names),3)); observed = np.empty_like(truth)
    estimated = np.empty_like(poses)
    cmd = np.tile(np.eye(4), (4,3,1,1)); executed = cmd.copy(); errors = cmd.copy()
    truth[0] = world_points(local, poses[0], point_names); observed[0] = truth[0]+measurement[0]
    for stage in range(1,5):
        for b in range(3):
            s = groups[b]
            estimated[stage-1,b] = fit_pose(local[s],observed[stage-1,s])
        cmd[stage-1] = commands(observed[stage-1], stage, target_points, point_names=point_names)
        for b in active_parts(stage):
            e = rigid(rotation_noise[stage-1,b],translation_noise[stage-1,b])
            if fault and fault['stage'] == stage and (stage == 4 or fault['part'] == 'ABC'[b]):
                e = rigid([0,0,0],fault['offset']) @ e
            errors[stage-1,b] = e
        if stage == 4:
            errors[3] = errors[3,0].copy()  # entire assembly has ONE execution
        executed[stage-1] = errors[stage-1] @ cmd[stage-1]
        poses[stage] = executed[stage-1] @ poses[stage-1]
        truth[stage] = world_points(local, poses[stage], point_names)
        observed[stage] = truth[stage]+measurement[stage]
    for b in range(3):
        s = groups[b]
        estimated[4,b] = fit_pose(local[s],observed[4,s])
    return ForwardTrajectory(observed,truth,poses,estimated,cmd,executed,errors,measurement)
