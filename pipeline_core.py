"""Simulation -> fixed graph -> quality inspection -> stage/point tracing.

Entry names and data/<name>, graph/<name> organization follow the user's local
Causal Discovery project. Discovery, graph fusion and graph-learning scores
are intentionally absent. Core calculations are specific to the new model.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Callable

import numpy as np

from assembly_sim.graph import build_graph, save_graph
from assembly_sim.model import ModelConfig, POINT_NAMES
from assembly_sim.tracing import fit_reference, trace_trajectory
from assembly_sim.cad import CADGeometry, CADMotion


@dataclass(frozen=True)
class TrialConfig:
    dataset_name: str = "assembly_dataset"
    n_total: int = 600
    test_fraction: float = 0.2
    seed: int = 42
    process_sigma: float = 0.02
    measurement_sigma: float = 0.03
    fault_fraction: float = 0.5
    fault_magnitude: float = 2.0
    fault_stage: int | None = None
    fault_part: str | None = None
    data_root: str = "data"
    graph_root: str = "graph"
    forward_settings: dict = field(default_factory=dict)
    cad_geometry: dict = field(default_factory=dict)
    cad_motion: dict = field(default_factory=dict)
    cad_checkpoints: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.dataset_name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",self.dataset_name):
            raise ValueError("dataset_name must contain 1-80 letters/digits/underscores/hyphens")
        for name,minimum in (("n_total",2),("seed",0)):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,Integral) or v<minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("test_fraction","fault_fraction","fault_magnitude"):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if not 0<self.test_fraction<1 or not 1<=self.n_test<self.n_total:
            raise ValueError("test_fraction must be between 0 and 1 and produce nonempty train/test sets")
        if not 0<=self.fault_fraction<=1 or self.fault_magnitude<=0:
            raise ValueError("fault_fraction must be in [0,1]; fault_magnitude must be positive")
        ModelConfig(self.process_sigma,self.measurement_sigma)
        from assembly_sim.forward import ForwardSettings, fault_choices
        fault_choices(self.fault_stage,self.fault_part)
        for name,cls in (("forward_settings",ForwardSettings),("cad_geometry",CADGeometry),("cad_motion",CADMotion)):
            if not isinstance(getattr(self,name),dict):
                raise ValueError(f"{name} must be an object")
            try: cls(**getattr(self,name))
            except TypeError as exc: raise ValueError(f"Invalid {name}: {exc}") from exc

        from assembly_sim.checkpoints import checkpoint_definition
        checkpoint_definition(CADGeometry(**self.cad_geometry), self.cad_checkpoints)

    @property
    def n_test(self):
        return math.floor(self.n_total*self.test_fraction+0.5)

    @property
    def n_train(self):
        return self.n_total-self.n_test


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _progress(callback, message):
    if callback:
        callback(message)


def _save_observations(directory: Path, observed: dict[str, np.ndarray], point_names=POINT_NAMES) -> None:
    np.savez_compressed(directory / "observations.npz", **observed)
    with (directory / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "stage", "point", "x", "y", "z"])
        for split, values in observed.items():
            for run, trajectory in enumerate(values):
                for k in range(5):
                    for i, point in enumerate(point_names):
                        writer.writerow([f"{split}_{run:05d}", split, k, point, *trajectory[k, i]])
    columns = [f"s{k}_{point}_{axis}" for k in range(5) for point in point_names for axis in "xyz"]
    for split, values in observed.items():
        name = {"train": "train_observed.csv", "calibration": "calibration_normal.csv", "test": "test_observed.csv"}[split]
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", *columns])
            writer.writerows([[f"{split}_{i:05d}", *row.reshape(-1)] for i, row in enumerate(values)])


def create_dataset(config: TrialConfig, progress=None) -> Path:
    if not isinstance(config,TrialConfig):
        raise TypeError("config must be TrialConfig")
    from assembly_sim.forward_dataset import create_forward_dataset
    return create_forward_dataset(config,progress)


def load_plans(directory: Path | str) -> dict[str, np.ndarray]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("generation_mode", "feedback") == "feedback":
        return {}
    if metadata["generation_mode"] == "cad_forward":
        with np.load(directory / "target_states.npz", allow_pickle=False) as archive:
            points = archive["target_checkpoints"]
        if points.shape != (4,len(metadata.get("point_names",POINT_NAMES)),3) or not np.isfinite(points).all():
            raise ValueError("Invalid forward targets")
        # Compatibility adapter: stage 0 is unused by prediction/diagnosis.
        plan = np.concatenate([points[:1], points])
        return {key: np.tile(plan, (count,1,1,1)) for key,count in metadata["splits"].items()}
    if metadata["generation_mode"] != "cad_reverse":
        raise ValueError("Unsupported generation_mode")
    with np.load(directory / "planned_checkpoints.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"train", "calibration", "test"}:
            raise ValueError("planned_checkpoints.npz requires train/calibration/test")
        return {key: archive[key].copy() for key in archive.files}


def load_dataset(directory: Path | str) -> dict[str, np.ndarray]:
    directory = Path(directory)
    metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    names=metadata.get("point_names",POINT_NAMES)
    from assembly_sim.checkpoints import point_groups
    point_groups(names)
    with np.load(directory / "observations.npz", allow_pickle=False) as archive:
        if set(archive.files) not in ({"train", "test"},{"train", "calibration", "test"}):
            raise ValueError("observations.npz must contain train/test")
        values={key:archive[key].copy() for key in archive.files}
    for split,rows in values.items():
        if rows.ndim!=4 or rows.shape[1:]!=(5,len(names),3) or not len(rows) or not np.isfinite(rows).all():
            raise ValueError(f"{split} 观测坐标必须是有限的 [流程数,5,测点数,3] 数组")
        if split in metadata.get("splits",{}) and metadata["splits"][split]!=len(rows):
            raise ValueError(f"{split} 样本数与 metadata 不一致")
    return values


def evaluate_results(results: list[dict], labels: list[dict]) -> dict:
    if len(results) != len(labels):
        raise ValueError("Evaluation result/label length mismatch")
    injected = [i for i, row in enumerate(labels) if row["injected"]]
    healthy = [i for i, row in enumerate(labels) if not row["injected"]]
    detected_injected = [i for i in injected if results[i]["quality_defect"]]
    localized = [i for i in detected_injected if results[i]["predicted_root"]]

    def ratio(n, d):
        return n/d if d else None

    metrics = dict(n_test=len(results), n_injected=len(injected), n_healthy=len(healthy),
                   n_quality_defects=sum(r["quality_defect"] for r in results),
                   n_detected_injected=len(detected_injected), n_localized_candidates=len(localized),
                   quality_detection_rate_on_injected=ratio(len(detected_injected), len(injected)),
                   healthy_quality_false_positive_rate=ratio(sum(results[i]["quality_defect"] for i in healthy), len(healthy)),
                   diagnosis_coverage_on_detected=ratio(len(localized), len(detected_injected)),
                   root_top1_on_detected=ratio(sum(results[i]["predicted_root"] == labels[i]["root"] for i in detected_injected), len(detected_injected)))
    for k in [3, 5]:
        metrics[f"root_top{k}_on_detected"] = ratio(sum(labels[i]["root"] in results[i]["ranked_roots"][:k] for i in detected_injected), len(detected_injected))
    metrics["interpretation"] = "Quality exceeds a preset final tolerance; injected faults may be compensated. Root metrics condition on injected + final quality defect."
    return metrics


def analyze_dataset(directory: Path | str, graph_directory: Path | str,
                    quality_limit: float | None = None,
                    progress: Callable[[str], None] | None = None,
                    reference_file: Path | str | None = None) -> dict:
    directory, graph_directory = Path(directory).resolve(), Path(graph_directory).resolve()
    values = load_dataset(directory)
    plans = load_plans(directory)
    if plans and any(plans[key].shape != values[key].shape for key in values):
        raise ValueError("CAD plans must match dataset sample counts and shapes")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    quality_limit = metadata.get("quality_limit_mm",1.0) if quality_limit is None else quality_limit
    if not np.isfinite(quality_limit) or quality_limit<=0:
        raise ValueError("终检限度必须为正数")
    _progress(progress, "建立独立溯源参考：使用正常训练样本或已保存的参考文件；不使用测试标签")
    from assembly_sim.diagnosis import prepare_reference, evaluation_labels, truth_annotations
    reference=prepare_reference(directory,values,plans,metadata,reference_file)
    graph = build_graph(forward=metadata.get("generation_mode") == "cad_forward",controller_policy=metadata.get("controller_policy","fixed_world"), point_names=metadata.get("point_names",POINT_NAMES))
    if plans:
        graph["conditioning"] = "Public CAD geometry and fixed stage targets; action inputs match the selected controller."
    results = []
    for i, observed in enumerate(values["test"]):
        result = trace_trajectory(observed, reference, quality_limit, graph,
                                  planned=plans["test"][i] if plans else None)
        result["sample_id"] = f"test_{i:05d}"
        results.append(result)
    graph_directory.mkdir(parents=True, exist_ok=True)
    write_json(graph_directory / "normal_reference.json", reference)
    write_json(graph_directory / "trace_results.json", results)
    with (graph_directory / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        columns = ["sample_id", "quality_defect", "observed_node", "maximum_final_error_mm", "predicted_root", "trace_status"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    # Test labels are read only after every prediction has been computed.
    labels=evaluation_labels(directory,len(results))
    truth=truth_annotations(directory)
    write_json(graph_directory/"ground_truth.json",truth)
    metrics=evaluate_results(results,labels) if labels is not None else dict(n_test=len(results),
        n_quality_defects=sum(r["quality_defect"] for r in results),evaluation="test labels unavailable; diagnosis still completed")
    example = next((row for row in results if row["quality_defect"]), results[0])
    annotation=truth.get(example["sample_id"],{})
    save_graph(graph_directory, graph, example["predicted_root"], example["observed_node"],
               true_root=annotation.get("root") if annotation.get("injected") is True else None, inferred_path=example["path"])
    summary = dict(task="diagnosis",dataset_dir=str(directory), graph_dir=str(graph_directory), metrics=metrics, example=example,
                   reference_source=reference.get("reference_source",str(reference_file or directory)),
                   quality_limit_mm=quality_limit,
                   files=dict(ground_truth=str(graph_directory/"ground_truth.json"), observations=str(directory/"observations.csv"), graph=str(graph_directory/"causal_graph.svg"),
                              traces=str(graph_directory/"trace_results.json"), results=str(graph_directory/"results.csv")))
    if plans:
        summary["files"].update(cad_preview=str(directory / "cad_preview.html"),
                                cad_model=str(directory / "cad" / "aircraft.FCStd"),
                                checkpoints=str(directory / "checkpoints.csv"),
                                planned_transforms=str(directory / ("target_states.npz" if metadata.get("generation_mode") == "cad_forward" else "planned_transforms.npz")))
    write_json(graph_directory / "summary.json", summary)
    _progress(progress, f"完成：{metrics['n_quality_defects']} / {len(results)} 条轨迹终检超限；结果 {graph_directory}")
    return summary


def run_pipeline(config: TrialConfig, progress=None) -> dict:
    """Generate general-purpose train/test data and its specified process graph."""
    directory=Path(config.data_root).expanduser().resolve()/config.dataset_name
    graph_directory=Path(config.graph_root).expanduser().resolve()/config.dataset_name
    if directory.exists() or graph_directory.exists():
        raise FileExistsError(f"Dataset or result already exists: {config.dataset_name}; choose a new dataset name")
    create_dataset(config,progress)
    metadata = json.loads((directory/"metadata.json").read_text())
    save_graph(graph_directory,build_graph(forward=True, point_names=metadata["point_names"]))
    summary=dict(task="generation_only",dataset_dir=str(directory),graph_dir=str(graph_directory),
        splits=dict(train=config.n_train,test=config.n_test),n_total=config.n_total,
        example=dict(sample_id="test_00000"),
        files=dict(observations=str(directory/"observations.csv"),labels=str(directory/"labels.json"),
                   graph=str(graph_directory/"causal_graph.svg")))
    write_json(directory/"generation_summary.json",summary)
    write_json(graph_directory/"summary.json",summary)
    _progress(progress,f"生成完成：{config.n_total} 条流程，训练 {config.n_train} / 测试 {config.n_test}；未运行溯源或校准。")
    return summary
