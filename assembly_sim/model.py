"""Coordinate-driven structural model of four aircraft assembly operations.

All coordinates are millimetres and use column-vector rigid transforms,
``x_next = R @ x_previous + t``. Arrays store one point per row.

The feedback transforms are computed from the previous *observations* and
applied to the previous *true* coordinates. Measurement noise therefore can
affect later physical coordinates through a positioning action, but is never
copied directly into the physical state. The reference geometry, mating
targets and feedback rules are specified assumptions, not discovered causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


POINT_NAMES = tuple(f"{part}{index}" for part in "ABC" for index in range(1, 4))
STAGE_NAMES = ("初始", "首次调姿", "前对合", "后对合", "再调姿")

# Three non-collinear labelled points are sufficient to determine a rigid pose.
# They are a teaching geometry, not an aircraft surface/contact-mechanics mesh.
_LOCAL_TEMPLATE = np.array(
    [[-100.0, -40.0, 0.0], [100.0, -40.0, 0.0], [100.0, 40.0, 0.0]]
)
_REFERENCE = np.tile(_LOCAL_TEMPLATE, (3, 1))
_INITIAL_TRANSLATIONS = np.array(
    [[-750.0, 50.0, 10.0], [0.0, 0.0, 0.0], [750.0, -50.0, -10.0]]
)
_ADJUSTED_TRANSLATIONS = np.array(
    [[-650.0, 20.0, 0.0], [0.0, 0.0, 0.0], [650.0, -20.0, 0.0]]
)
_ASSEMBLED_TRANSLATIONS = np.array(
    [[-500.0, 0.0, 0.0], [0.0, 0.0, 0.0], [500.0, 0.0, 0.0]]
)
_FINAL_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_FINAL_TRANSLATION = np.array([10.0, 20.0, 30.0])


@dataclass(frozen=True)
class ModelConfig:
    """Per-axis noise standard deviations and the final radial limit, in mm."""

    process_sigma: float = 0.02
    measurement_sigma: float = 0.03
    quality_limit: float = 1.0

    def __post_init__(self) -> None:
        for name in ("process_sigma", "measurement_sigma", "quality_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or (name == "quality_limit" and value == 0):
                relation = "positive" if name == "quality_limit" else "nonnegative"
                raise ValueError(f"{name} must be {relation}")


@dataclass(frozen=True)
class Fault:
    """One local offset after a stage action and before its observation.

    ``offset`` is expressed in that stage's global coordinate frame. The
    resulting coordinate deviation persists via the structural recurrence;
    the offset is not injected a second time at subsequent stages.
    """

    stage: int
    point: str
    offset: tuple[float, float, float]

    def __post_init__(self) -> None:
        _check_stage(self.stage)
        if self.point not in POINT_NAMES:
            raise ValueError(f"point must be one of {POINT_NAMES}")
        _finite_array(self.offset, (3,), "fault offset")


@dataclass
class Trajectory:
    """One complete run; measured coordinates are the algorithm's input data.

    The additional true/nominal arrays and action matrices are simulator audit
    information. ``rotations[k-1,b]`` and ``translations[k-1,b]`` describe the
    action actually executed on part ``b`` at operation ``k``.
    """

    observed: np.ndarray  # [5 stages, 9 points, 3 coordinates]
    true: np.ndarray
    nominal: np.ndarray
    rotations: np.ndarray  # [4 operations, 3 parts, 3, 3]
    translations: np.ndarray  # [4 operations, 3 parts, 3]


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array of shape {shape}") from exc
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _check_stage(stage: int) -> None:
    if isinstance(stage, bool) or not isinstance(stage, Integral) or not 1 <= stage <= 4:
        raise ValueError("stage must be an integer from 1 to 4")


def _positioned(translations: np.ndarray) -> np.ndarray:
    return _REFERENCE + np.repeat(translations, 3, axis=0)


def nominal_trajectory(config: ModelConfig | None = None) -> np.ndarray:
    """Analytical ideal coordinates, with all disturbances and faults absent.

    The optional configuration is accepted for a uniform public interface;
    changing noise scales or the quality limit cannot change nominal geometry.
    A fresh array is returned on every call.
    """

    if config is not None and not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    initial = _positioned(_INITIAL_TRANSLATIONS)
    adjusted = _positioned(_ADJUSTED_TRANSLATIONS)
    front_translations = _ADJUSTED_TRANSLATIONS.copy()
    front_translations[0] = _ASSEMBLED_TRANSLATIONS[0]
    front = _positioned(front_translations)
    rear = _positioned(_ASSEMBLED_TRANSLATIONS)
    final = rear @ _FINAL_ROTATION.T + _FINAL_TRANSLATION
    return np.stack([initial, adjusted, front, rear, final])


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares proper rigid transform mapping labelled source to target.

    Uses the SVD/Kabsch solution. At least three non-collinear points are
    required in each array; coplanar points are valid. Reflections are excluded.
    """

    try:
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("source and target must be numeric coordinate arrays") from exc
    if source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 3:
        raise ValueError("source must have shape (n, 3), with n >= 3")
    if target.shape != source.shape:
        raise ValueError("target must have the same shape as source")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must contain only finite values")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(centered_source) < 2 or np.linalg.matrix_rank(centered_target) < 2:
        raise ValueError("rigid fitting requires non-collinear points")
    left, _, right_t = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(right_t.T @ left.T) >= 0 else -1.0
    rotation = right_t.T @ correction @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def stage_transforms(
    previous_observed: np.ndarray, stage: int, planned: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the next positioning action from the preceding nine observations.

    1: fit each part to its prescribed adjustment target independently.
    2: fit current A/B poses; move only A to the prescribed pose relative to B.
    3: fit current B/C poses; move only C to the prescribed pose relative to B.
    4: fit all nine points to the fixed final target; move the locked assembly
       using one shared transform.

    For stage 2, G_A and G_B map their local reference coordinates to current
    global coordinates. The executed action is H_A = G_B D_BA inv(G_A).
    Stage 3 uses H_C = G_B D_BC inv(G_C), with the analogous convention.
    """

    _check_stage(stage)
    previous = _finite_array(previous_observed, (9, 3), "previous_observed")
    if planned is not None:
        planned = _finite_array(planned, (5, 9, 3), "planned")
    rotations = np.repeat(np.eye(3)[None], 3, axis=0)
    translations = np.zeros((3, 3))
    if stage == 1:
        targets = _positioned(_ADJUSTED_TRANSLATIONS) if planned is None else planned[1]
        for part in range(3):
            group = slice(part * 3, part * 3 + 3)
            rotations[part], translations[part] = rigid_fit(previous[group], targets[group])
    elif stage in (2, 3):
        moving_part = 0 if stage == 2 else 2
        moving = slice(moving_part * 3, moving_part * 3 + 3)
        template_b = _LOCAL_TEMPLATE if planned is None else planned[4, 3:6]
        template_m = _LOCAL_TEMPLATE if planned is None else planned[4, moving]
        rotation_b, translation_b = rigid_fit(template_b, previous[3:6])
        rotation_m, translation_m = rigid_fit(template_m, previous[moving])
        # Both relative target rotations are I. The target translation is in B's frame.
        relative_translation = _ASSEMBLED_TRANSLATIONS[moving_part] if planned is None else np.zeros(3)
        rotation = rotation_b @ rotation_m.T
        translation = translation_b + rotation_b @ relative_translation - rotation @ translation_m
        rotations[moving_part] = rotation
        translations[moving_part] = translation
    else:
        target = nominal_trajectory()[4] if planned is None else planned[4]
        rotation, translation = rigid_fit(previous, target)
        rotations[:] = rotation
        translations[:] = translation
    return rotations, translations


def apply_transforms(points: np.ndarray, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Apply one proper rigid action per part, without adding any disturbances."""

    points = _finite_array(points, (9, 3), "points")
    rotations = _finite_array(rotations, (3, 3, 3), "rotations")
    translations = _finite_array(translations, (3, 3), "translations")
    gram = rotations @ rotations.transpose(0, 2, 1)
    if not np.allclose(gram, np.eye(3), atol=1e-7, rtol=0) or not np.allclose(
        np.linalg.det(rotations), 1.0, atol=1e-7, rtol=0
    ):
        raise ValueError("rotations must be proper orthogonal matrices (SO(3))")
    grouped = points.reshape(3, 3, 3)
    moved = np.einsum("bij,bpj->bpi", rotations, grouped) + translations[:, None, :]
    return moved.reshape(9, 3)


def predict_next(previous_observed: np.ndarray, stage: int, planned: np.ndarray | None = None) -> np.ndarray:
    """Coordinate-only normal prediction for local innovation scoring.

    This uses measured coordinates in place of the unobserved true state.
    Consequently its residual includes current/previous measurement error and
    should be calibrated on healthy trajectories before declaring an anomaly.
    """

    rotations, translations = stage_transforms(previous_observed, stage, planned)
    return apply_transforms(previous_observed, rotations, translations)


def simulate(config: ModelConfig, seed: int = 42, fault: Fault | None = None,
             planned: np.ndarray | None = None) -> Trajectory:
    """Generate a complete trajectory using feedback actions and fresh errors.

    X_k = H_k(Y_(k-1)) X_(k-1) + U_k + F_k
    Y_k = X_k + epsilon_k

    Noise is sampled before the recurrence. The same configuration and seed
    therefore give exactly matched exogenous noises for healthy/fault runs.
    ``quality_limit`` is a later decision threshold, not a generation parameter.
    """

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if fault is not None and not isinstance(fault, Fault):
        raise TypeError("fault must be a Fault or None")
    rng = np.random.default_rng(seed)
    process_noise = rng.normal(0.0, config.process_sigma, size=(4, 9, 3))
    measurement_noise = rng.normal(0.0, config.measurement_sigma, size=(5, 9, 3))
    nominal = nominal_trajectory(config) if planned is None else _finite_array(planned, (5, 9, 3), "planned").copy()
    truth, observed = np.empty_like(nominal), np.empty_like(nominal)
    rotations = np.empty((4, 3, 3, 3))
    translations = np.empty((4, 3, 3))
    truth[0] = nominal[0]
    observed[0] = truth[0] + measurement_noise[0]
    for stage in range(1, 5):
        rotation, translation = stage_transforms(observed[stage - 1], stage, planned)
        rotations[stage - 1], translations[stage - 1] = rotation, translation
        truth[stage] = apply_transforms(truth[stage - 1], rotation, translation) + process_noise[stage - 1]
        if fault is not None and stage == fault.stage:
            truth[stage, POINT_NAMES.index(fault.point)] += np.asarray(fault.offset, dtype=float)
        observed[stage] = truth[stage] + measurement_noise[stage]
    return Trajectory(observed, truth, nominal, rotations, translations)


def model_spec() -> dict[str, Any]:
    """Return the fixed geometry, rules and coordinate conventions as JSON data."""

    nominal = nominal_trajectory()
    return {
        "version": "1.0",
        "units": "mm",
        "point_names": list(POINT_NAMES),
        "stage_names": list(STAGE_NAMES),
        "reference_points": {name: _REFERENCE[index].tolist() for index, name in enumerate(POINT_NAMES)},
        "initial_positions": {name: nominal[0, index].tolist() for index, name in enumerate(POINT_NAMES)},
        "target_coordinates": {
            str(stage): {name: nominal[stage, index].tolist() for index, name in enumerate(POINT_NAMES)}
            for stage in range(1, 5)
        },
        "target_relative_poses": {
            "D_BA": {"R": np.eye(3).tolist(), "t": [-500.0, 0.0, 0.0]},
            "D_BC": {"R": np.eye(3).tolist(), "t": [500.0, 0.0, 0.0]},
        },
        "stage_rules": {
            "1": "Each part: rigid_fit(previous observed part, fixed adjustment target).",
            "2": "Move only A: H_A = G_B D_BA inverse(G_A). B and C remain fixed.",
            "3": "Move only C: H_C = G_B D_BC inverse(G_C). Connected A and B remain fixed.",
            "4": "All nine points determine one rigid_fit to the fixed final target; move the entire assembly.",
        },
        "parameter_meaning": {
            "P_or_X": "True 3D point coordinate in the global frame.",
            "Y": "Observed coordinate X + fresh measurement noise.",
            "G_b": "Pose fitted from a part's local reference template to its current observations.",
            "D_BA_or_D_BC": "Fixed intended A/C pose expressed in B's local coordinate frame.",
            "H_or_R_t": "Executed incremental rigid motion, computed from preceding observations.",
            "U": "Fresh independent per-axis Gaussian local point disturbance in the current global frame.",
            "F": "One injected local offset after the selected stage action and before its measurement.",
            "epsilon": "Fresh measurement error; influences later motion only through feedback.",
            "quality_limit": "Fixed radial final-coordinate tolerance; does not modify trajectory generation.",
        },
        "assumptions": [
            "The named checkpoints actually participate in the specified feedback positioning rules.",
            "Rigid motion is shared within each moved group; added local deviations may change distances.",
            "Geometry, feedback rules and causal edges are supplied assumptions, not causal discovery output.",
        ],
    }
