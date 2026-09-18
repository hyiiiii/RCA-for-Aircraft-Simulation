"""Observation-only stage/point diagnosis on the specified process graph.

The normal structural prediction is known; only the residual distribution is
calibrated from healthy observations. This module never reads simulator truth,
fault labels, executed transforms, or an oracle-selected downstream point.
"""
from __future__ import annotations

import numpy as np

from .graph import ancestors, build_graph, directed_path
from .model import POINT_NAMES, STAGE_NAMES, nominal_trajectory, predict_next


def _observations(value, batch=False, point_names=POINT_NAMES):
    array = np.asarray(value, dtype=float)
    expected = (5, len(point_names), 3)
    if (batch and (array.ndim != 4 or array.shape[1:] != expected or len(array) < 2)) or (
        not batch and array.shape != expected
    ):
        raise ValueError("observations must be (N>=2,5,number_of_points,3)" if batch else "observations must be (5,number_of_points,3)")
    if not np.isfinite(array).all():
        raise ValueError("observations must be finite")
    return array


def innovations(observed: np.ndarray, planned: np.ndarray | None = None, generation_mode=None, controller_policy="body_relative", point_names=POINT_NAMES) -> np.ndarray:
    measured = _observations(observed, point_names=point_names)
    if generation_mode == "cad_forward":
        from .forward import commands, world_points
        return np.stack([measured[k] - world_points(measured[k-1], commands(measured[k-1], k, planned[1:], controller_policy, point_names), point_names) for k in range(1,5)])
    return np.stack([measured[k] - predict_next(measured[k-1], k, planned) for k in range(1, 5)])


def _plans(value, observations, point_names=POINT_NAMES):
    if value is None:
        return [None]*len(observations)
    planned = _observations(value, True, point_names)
    if planned.shape != observations.shape:
        raise ValueError("planned and observed batch shapes must match")
    return planned


def fit_reference(train: np.ndarray, calibration: np.ndarray, alpha: float = 0.01,
                  train_planned: np.ndarray | None = None,
                  calibration_planned: np.ndarray | None = None, generation_mode=None, controller_policy="body_relative", point_names=POINT_NAMES) -> dict:
    train, calibration = _observations(train, True, point_names), _observations(calibration, True, point_names)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if (train_planned is None) != (calibration_planned is None):
        raise ValueError("Both training and calibration plans are required for CAD data")
    train_plans, calibration_plans = _plans(train_planned, train, point_names), _plans(calibration_planned, calibration, point_names)
    residuals = np.stack([innovations(row, plan, generation_mode, controller_policy, point_names) for row, plan in zip(train, train_plans)])
    center = residuals.mean(axis=0)
    # A small numerical floor keeps zero-noise experiments well defined.
    scale = np.maximum(residuals.std(axis=0, ddof=1), 1e-6)
    normal_scores = np.stack([np.linalg.norm((innovations(row, plan, generation_mode, controller_policy, point_names)-center)/scale, axis=-1)
                              for row, plan in zip(calibration, calibration_plans)])
    threshold = max(float(np.quantile(normal_scores.max(axis=(1, 2)), 1-alpha, method="higher")), 1e-6)
    return dict(point_names=list(point_names), controller_policy=controller_policy, version=1, method="structural_innovation", center=center.tolist(), scale=scale.tolist(),
                score_threshold=threshold, alpha=alpha, n_train=len(train), n_calibration=len(calibration),
                calibration_statistic="maximum standardized innovation over 4 stages x all configured points",
                generation_mode=generation_mode or ("feedback" if train_planned is None else "cad_reverse"),
                nominal_coordinates=(nominal_trajectory() if train_planned is None else train_plans[0]).tolist(),
                input_policy="healthy measured coordinates and public CAD process plans; no truth, executed actions or fault labels")


def trace_trajectory(observed: np.ndarray, reference: dict, quality_limit: float = 1.0,
                     graph: dict | None = None, observed_point: str | None = None,
                     planned: np.ndarray | None = None) -> dict:
    point_names = tuple(reference.get("point_names", POINT_NAMES))
    measured = _observations(observed, point_names=point_names)
    if not np.isfinite(quality_limit) or quality_limit <= 0:
        raise ValueError("quality_limit must be finite and positive")
    center, scale = np.asarray(reference["center"], float), np.asarray(reference["scale"], float)
    if center.shape != (4, len(point_names), 3) or scale.shape != center.shape or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid reference residual center/scale")
    threshold = float(reference["score_threshold"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid reference threshold")
    forward = reference.get("generation_mode") == "cad_forward"
    graph = graph or build_graph(forward=forward,controller_policy=reference.get("controller_policy","fixed_world"),point_names=point_names)
    if (reference.get("generation_mode", "feedback") in ("cad_reverse", "cad_forward")) != (planned is not None):
        raise ValueError("CAD diagnosis requires the matching sample's public plan and a CAD reference")
    nominal = nominal_trajectory() if planned is None else _observations(planned, point_names=point_names)
    final_errors = np.linalg.norm(measured[4]-nominal[4], axis=-1)
    defects = [point_names[i] for i in np.flatnonzero(final_errors > quality_limit)]
    if observed_point is None:
        observed_point = point_names[int(np.argmax(final_errors))]
    if observed_point.startswith("s4_"):
        observed_point = observed_point[3:]
    if observed_point not in point_names:
        raise ValueError(f"observed_point must be one of {point_names}")
    observed_node = f"s4_{observed_point}"
    candidate_ids = ancestors(graph, observed_node, include_self=True)
    residuals = innovations(measured, planned, reference.get("generation_mode"),reference.get("controller_policy","fixed_world"),point_names)
    normalized = np.linalg.norm((residuals-center)/scale, axis=-1)
    ranking = []
    for k in range(1, 5):
        for i, point in enumerate(point_names):
            node = f"s{k}_{point}"
            if node in candidate_ids:
                ranking.append(dict(node=node, stage=k, stage_name=STAGE_NAMES[k], point=point,
                                    score=float(normalized[k-1, i]),
                                    residual_mm=float(np.linalg.norm(residuals[k-1, i])),
                                    residual_xyz=residuals[k-1, i].tolist()))
    if forward:
        # Fault candidates are executing bodies/actions, never individual points.
        ranking = []
        for node in graph["nodes"]:
            if node["kind"] != "action" or node["id"] not in candidate_ids:
                continue
            k = node["stage"]
            indices = [i for i,p in enumerate(point_names) if p[0] in node["moved_parts"]]
            ranking.append(dict(node=node["id"],stage=k,stage_name=STAGE_NAMES[k],
                point=node["moved_parts"],part=node["moved_parts"],
                score=float(normalized[k-1,indices].max()),
                residual_mm=float(np.sqrt(np.mean(residuals[k-1,indices]**2))),
                residual_xyz=residuals[k-1,indices].mean(axis=0).tolist()))
    ranking.sort(key=lambda row: (-row["score"], row["stage"], row["point"]))
    entry_error = float(final_errors[point_names.index(observed_point)])
    entry_abnormal = entry_error > quality_limit
    significant = bool(ranking and ranking[0]["score"] > threshold)
    predicted = ranking[0]["node"] if entry_abnormal and significant else None
    status = "localized_candidate" if predicted else "inconclusive" if entry_abnormal else "no_quality_defect_at_entry"
    return dict(trace_method="structural_innovation", observed_node=observed_node,
                predicted_root=predicted, ranked_roots=[row["node"] for row in ranking],
                scores={row["node"]: row["score"] for row in ranking}, ranking=ranking,
                quality_defect=bool(defects), entry_quality_defect=entry_abnormal,
                defect_points=defects, final_errors_mm=dict(zip(point_names, final_errors.tolist())),
                maximum_final_error_mm=float(final_errors.max()), entry_error_mm=entry_error,
                quality_limit_mm=quality_limit, score_threshold=threshold, trace_status=status,
                candidate_policy="executed rigid-body actions among terminal ancestors" if forward else "ancestors of terminal entry, including the entry; stages 1..4",
                path=directed_path(graph, predicted, observed_node),
                notes="Candidate ranking under the specified feedback model; not proof of a unique physical cause.")
