#!/usr/bin/env python3
"""飞机四阶段装配仿真桌面入口；运行固定结构方程，不进行因果发现。

后台线程只通过 queue 发送消息，所有 Tk 操作均在主线程完成。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk, filedialog
except ImportError as exc:
    tk = ttk = messagebox = simpledialog = None  # type: ignore[assignment]
    TK_IMPORT_ERROR = str(exc)
else:
    TK_IMPORT_ERROR = ""

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_ROOT = PROJECT_ROOT / "app_state"
POINTS = tuple(f"{part}{i}" for part in "ABC" for i in range(1, 4))
STAGES = {0: "初始参考", 1: "首次调姿", 2: "前对合", 3: "后对合", 4: "再调姿 / 终检"}
DEFAULTS = {
    "dataset_name":"assembly_dataset", "n_total":"600", "test_fraction":"0.2", "seed":"42",
    "process_sigma":"0.02", "measurement_sigma":"0.03", "fault_fraction":"0.5",
    "fault_magnitude":"2.0", "fault_stage":"随机", "fault_part":"随机",
    "cad_geometry":"{}", "cad_motion":"{}", "forward_settings":"{}", "cad_checkpoints":"{}",
}
FIELDS = [
    ("dataset_name","数据集名称"),("n_total","总流程数"),("test_fraction","测试集占比 0–1"),
    ("seed","随机种子"),("process_sigma","工艺平移 σ / mm"),("measurement_sigma","测量噪声 σ / mm"),
    ("fault_fraction","全体故障比例 0–1"),("fault_magnitude","故障幅值 / mm"),
    ("fault_stage","故障注入阶段"),("fault_part","故障部件"),
]


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace a complete JSON file atomically; never leave a partial ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def validate_parameters(values):
    from dataclasses import asdict
    from pipeline_core import TrialConfig
    result={}
    for key,default in DEFAULTS.items():
        value=values.get(key,default)
        try:
            if key in ("n_total","seed"): value=int(value)
            elif key in ("test_fraction","process_sigma","measurement_sigma","fault_fraction","fault_magnitude"): value=float(value)
            elif key=="fault_stage": value=None if value=="随机" else int(value)
            elif key=="fault_part": value=None if value=="随机" else value
            elif key in ("cad_geometry","cad_motion","forward_settings","cad_checkpoints"): value=json.loads(value)
        except (ValueError,TypeError):
            raise ValueError(f"{dict(FIELDS).get(key,key)}格式无效") from None
        result[key]=value
    result.update(data_root=str(PROJECT_ROOT/"data"),graph_root=str(PROJECT_ROOT/"graph"))
    return asdict(TrialConfig(**result))


def ancestor_path_edges(graph: dict, source: str, target: str) -> set[tuple[str, str]]:
    """All directed edges on a source-to-target path; never invent cross edges."""
    edges = [(str(e["source"]), str(e["target"])) for e in graph.get("edges", [])]
    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for a, b in edges:
        outgoing.setdefault(a, set()).add(b)
        incoming.setdefault(b, set()).add(a)

    def closure(start: str, adjacency: dict[str, set[str]]) -> set[str]:
        seen, todo = {start}, [start]
        while todo:
            for node in adjacency.get(todo.pop(), set()):
                if node not in seen:
                    seen.add(node)
                    todo.append(node)
        return seen

    descendants = closure(source, outgoing)
    ancestors = closure(target, incoming)
    return {(a, b) for a, b in edges if a in descendants and b in ancestors}


def preview_graph() -> dict:
    """Same parent rules as the pipeline; do not keep a second graph definition."""
    from assembly_sim.graph import build_graph
    return build_graph(forward=True)


class SimulationApp:
    def __init__(self, root: Any, *, persist: bool = True):
        self.root = root
        self.persist = persist
        self.running = False
        self.messages: queue.Queue = queue.Queue()
        self.result: dict[str, Any] | None = None
        self.graph = preview_graph()
        self.history = read_json(STATE_ROOT / "history.json", []) if persist else []
        self.presets = read_json(STATE_ROOT / "presets.json", {}) if persist else {}
        if not isinstance(self.history, list):
            self.history = []
        if not isinstance(self.presets, dict):
            self.presets = {}
        settings = read_json(STATE_ROOT / "settings.json", {}) if persist else {}
        settings = settings if isinstance(settings, dict) else {}
        if "n_total" not in settings and "n_train" in settings:
            try:
                total=sum(int(settings.get(k,0)) for k in ("n_train","n_calibration","n_test"))
                settings=dict(settings,n_total=str(total),test_fraction=str(int(settings.get("n_test",0))/total))
            except (ValueError,ZeroDivisionError):
                settings={} 
        self.vars = {key: tk.StringVar(root, str(settings.get(key, value))) for key, value in DEFAULTS.items()}
        saved_name = self.vars["dataset_name"].get()
        if (PROJECT_ROOT / "data" / saved_name).exists() or (PROJECT_ROOT / "graph" / saved_name).exists():
            self.vars["dataset_name"].set(f"assembly_{datetime.now():%Y%m%d_%H%M%S}")
        self.status = tk.StringVar(root, "就绪 · CAD 正向数据生成")
        self.error = tk.StringVar(root, "")
        self.diagnosis_dataset=tk.StringVar(root,"")
        self.diagnosis_limit=tk.StringVar(root,"1.0")
        self.diagnosis_sample=tk.StringVar(root,"")
        self.diagnosis_results={}
        self.current_task="generation"
        self.show_initial = tk.BooleanVar(root, True)
        self.preset_name = tk.StringVar(root, "")
        self.start_time = 0.0
        self.root.title("飞机装配仿真 · CAD 建模与根因溯源")
        self.root.geometry("1360x900")
        self.root.minsize(1060, 740)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self.callback_error
        self.build_ui()
        self.refresh_history()
        self.draw_graph()
        self.root.after(80, self.poll)

    def build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        from assembly_sim.ui_controls import install_checkmarks
        install_checkmarks(self.root)
        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", font=("Helvetica", 12))
        style.configure("Title.TLabel", font=("Helvetica", 21, "bold"), foreground="#183549")
        style.configure("Treeview", rowheight=25, font=("Helvetica", 11))
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="飞机四阶段装配仿真", style="Title.TLabel").pack(anchor="w")
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, width=292, padding=(0, 0, 16, 0))
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="仿真参数", font=("Helvetica", 15, "bold")).pack(anchor="w", pady=(0, 7))
        form = ttk.Frame(sidebar)
        form.pack(fill="x")
        self.input_widgets = []
        for row, (key, label) in enumerate(FIELDS):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            if key == "fault_stage":
                entry = ttk.Combobox(form, textvariable=self.vars[key], values=("随机", "1", "2", "3", "4"), width=11, state="readonly")
            elif key == "fault_part":
                entry = ttk.Combobox(form, textvariable=self.vars[key], values=("随机", "A", "B", "C", "ABC"), width=11, state="readonly")
                self.fault_selector = entry
            else:
                entry = ttk.Entry(form, textvariable=self.vars[key], width=13)
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=5)
            self.input_widgets.append(entry)
        form.columnconfigure(1, weight=1)
        self.cad_settings_button = ttk.Button(sidebar, text="CAD 尺寸、测点与目标…", command=self.configure_cad)
        self.cad_settings_button.pack(fill="x", pady=(4, 0))
        ttk.Button(sidebar,text="使用说明…",command=self.show_help).pack(fill="x",pady=(4,0))
        self.run_button = ttk.Button(sidebar, text="生成数据", command=self.start)
        self.run_button.pack(fill="x", pady=(5, 5))
        self.progress = ttk.Progressbar(sidebar, mode="indeterminate")
        self.progress.pack(fill="x")
        ttk.Label(sidebar, textvariable=self.error, foreground="#b43a35", wraplength=266).pack(anchor="w", pady=7)
        ttk.Separator(sidebar).pack(fill="x", pady=8)
        ttk.Label(sidebar, text="参数预设").pack(anchor="w")
        self.preset_combo = ttk.Combobox(sidebar, textvariable=self.preset_name, values=sorted(self.presets), state="readonly")
        self.preset_combo.pack(fill="x", pady=6)
        preset_actions = ttk.Frame(sidebar)
        preset_actions.pack(fill="x")
        ttk.Button(preset_actions, text="载入", command=self.load_preset).pack(side="left", expand=True, fill="x")
        ttk.Button(preset_actions, text="另存为", command=self.save_preset).pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(side="left", fill="both", expand=True)
        self.overview_tab = ttk.Frame(self.tabs, padding=14)
        from assembly_sim.cad_view import CADAssemblyView
        self.cad_tab = CADAssemblyView(self.tabs, on_load=self.load_cad_directory)
        self.diagnosis_tab = ttk.Frame(self.tabs,padding=16)
        self.graph_tab = self.diagnosis_tab
        self.log_tab = ttk.Frame(self.tabs, padding=8)
        self.history_tab = ttk.Frame(self.tabs, padding=8)
        for tab, title in ((self.overview_tab, "总览"), (self.cad_tab, "三维装配"), (self.diagnosis_tab,"根因溯源"), (self.log_tab, "日志"), (self.history_tab, "历史")):
            self.tabs.add(tab, text=title)
        self.tabs.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        load_bar=ttk.Frame(self.diagnosis_tab);load_bar.pack(fill="x",pady=6)
        ttk.Label(load_bar,text="数据集目录").pack(side="left")
        ttk.Entry(load_bar,textvariable=self.diagnosis_dataset).pack(side="left",fill="x",expand=True,padx=8)
        ttk.Button(load_bar,text="载入数据集…",command=self.choose_diagnosis_dataset).pack(side="right")
        options=ttk.Frame(self.diagnosis_tab);options.pack(fill="x",pady=6)
        ttk.Label(options,text="终检限度 / mm").pack(side="left")
        ttk.Entry(options,textvariable=self.diagnosis_limit,width=10).pack(side="left",padx=8)
        self.trace_button=ttk.Button(options,text="开始溯源",command=self.start_diagnosis)
        self.trace_button.pack(side="left",padx=12)
        result_bar=ttk.Frame(self.diagnosis_tab);result_bar.pack(fill="x",pady=6)
        ttk.Label(result_bar,text="查看测试轨迹结果").pack(side="left")
        self.diagnosis_sample_box=ttk.Combobox(result_bar,textvariable=self.diagnosis_sample,values=(),state="readonly",width=20)
        self.diagnosis_sample_box.pack(side="left",padx=8)
        self.diagnosis_sample_box.bind("<<ComboboxSelected>>",self.select_diagnosis_sample)
        ttk.Button(result_bar,text="查看总览与排名",command=lambda:self.tabs.select(self.overview_tab)).pack(side="left")
        self.diagnosis_caption=tk.StringVar(self.root,"")
        ttk.Label(self.diagnosis_tab,textvariable=self.diagnosis_caption,wraplength=850).pack(anchor="w",pady=6)
        self.summary = tk.Text(self.overview_tab, wrap="word", borderwidth=0, background="white", padx=14, pady=12, font=("Helvetica", 13), height=15)
        self.summary.pack(fill="both", expand=True)
        self.set_text(self.summary, "尚无数据")
        cad_toolbar = ttk.Frame(self.overview_tab)
        cad_toolbar.pack(fill="x", pady=8)
        ttk.Button(cad_toolbar, text="查看内嵌三维装配", command=self.open_cad_preview).pack(side="left")
        ttk.Button(cad_toolbar, text="打开 FreeCAD 模型", command=self.open_cad_model).pack(side="left", padx=8)
        ttk.Button(cad_toolbar, text="浏览器预览", command=lambda:self.open_cad_file("cad_preview.html")).pack(side="left")
        ttk.Label(self.overview_tab, text="根因排名").pack(anchor="w", pady=(12, 6))
        self.ranking = self.make_table(self.overview_tab, ("node", "stage", "point", "score", "residual_mm"), ("节点", "阶段", "部件 / 测点", "分数", "新增残差 / mm"), height=7)
        toolbar = ttk.Frame(self.graph_tab)
        toolbar.pack(fill="x", pady=(0, 5))
        self.initial_toggle=ttk.Checkbutton(toolbar, text="初始状态与首次调姿", variable=self.show_initial, command=self.draw_graph)
        self.initial_toggle.pack(side="left")
        ttk.Button(toolbar, text="打开完整 SVG", command=self.open_svg).pack(side="right")
        canvas_wrap = ttk.Frame(self.graph_tab)
        canvas_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_wrap, background="white", highlightthickness=0)
        yscroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        xscroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas_wrap.columnconfigure(0, weight=1)
        canvas_wrap.rowconfigure(0, weight=1)
        self.log = tk.Text(self.log_tab, wrap="word", background="#f9fafc", borderwidth=0, font=("Menlo", 11), state="disabled")
        self.log.pack(fill="both", expand=True)
        self.history_table = self.make_table(self.history_tab, ("time", "dataset", "status", "elapsed"), ("时间", "数据集", "结果", "秒"))
        self.history_table.bind("<Double-1>", self.open_history)
        ttk.Label(outer, textvariable=self.status, foreground="#315268").pack(anchor="w", pady=(10, 0))

    @staticmethod
    def make_table(parent: Any, columns: tuple, labels: tuple, height: int = 18) -> Any:
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill="both", expand=True)
        tree = ttk.Treeview(wrapper, columns=columns, show="headings", height=height)
        for col, label in zip(columns, labels):
            tree.heading(col, text=label)
            tree.column(col, width=120, minwidth=70, anchor="center")
        scroll = ttk.Scrollbar(wrapper, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return tree

    @staticmethod
    def set_text(widget: Any, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content)
        widget.configure(state="disabled")

    def log_message(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def save_settings(self) -> None:
        if self.persist:
            atomic_write_json(STATE_ROOT / "settings.json", {key: var.get() for key, var in self.vars.items()})

    def start(self) -> None:
        if self.running:
            return
        self.error.set("")
        try:
            params = validate_parameters({key: var.get() for key, var in self.vars.items()})
            from pipeline_core import TrialConfig
            config = TrialConfig(**params)
            self.save_settings()
        except Exception as exc:
            self.error.set(str(exc))
            self.status.set("参数或初始化错误，未开始运行")
            self.log_message(f"未开始：{exc}")
            return
        self.current_task="generation"
        self.trace_button.configure(state="disabled")
        self.running = True
        self.start_time = time.monotonic()
        self.run_button.configure(state="disabled")
        for widget in self.input_widgets:
            widget.configure(state="disabled")
        self.progress.start(12)
        self.status.set("运行中 · 生成通用仿真数据")
        self.log_message(f"开始：{params['dataset_name']}，种子 {params['seed']}")

        def worker() -> None:
            try:
                from pipeline_core import run_pipeline
                result = run_pipeline(config, progress=lambda text: self.messages.put(("progress", str(text))))
                self.messages.put(("complete", (params, result)))
            except Exception:
                self.messages.put(("failed", (params, traceback.format_exc())))

        threading.Thread(target=worker, name="assembly-simulation", daemon=True).start()

    def choose_diagnosis_dataset(self):
        if self.running: return
        path=filedialog.askdirectory(parent=self.root,title="选择已有数据集目录",initialdir=PROJECT_ROOT/"data")
        if not path: return
        try:
            self.load_cad_directory(Path(path))
            self.tabs.select(self.diagnosis_tab)
        except (ValueError,OSError) as exc:
            self.error.set(str(exc))

    def start_diagnosis(self):
        if self.running:return
        self.error.set("")
        try:
            directory=Path(self.diagnosis_dataset.get()).expanduser().resolve()
            if not self.diagnosis_dataset.get() or not (directory/"observations.npz").is_file():
                raise ValueError("请先载入包含 observations.npz 的数据集")
            limit=float(self.diagnosis_limit.get())
            if not math.isfinite(limit) or limit<=0: raise ValueError("终检限度必须是正数")
        except (ValueError,OSError) as exc:
            self.error.set(str(exc));return
        output=PROJECT_ROOT/"graph"/directory.name/("diagnosis_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.current_task="diagnosis";self.running=True;self.start_time=time.monotonic()
        self.run_button.configure(state="disabled");self.trace_button.configure(state="disabled")
        for widget in self.input_widgets:widget.configure(state="disabled")
        self.progress.start(12)
        self.status.set("运行中 · 独立因果溯源")
        self.diagnosis_caption.set("正在溯源…")
        params=dict(dataset_name=directory.name,operation="diagnosis",dataset_dir=str(directory),quality_limit=limit)
        def worker():
            try:
                from pipeline_core import analyze_dataset
                result=analyze_dataset(directory,output,limit,progress=lambda text:self.messages.put(("progress",str(text))))
                self.messages.put(("complete",(params,result)))
            except Exception:
                self.messages.put(("failed",(params,traceback.format_exc())))
        threading.Thread(target=worker,name="assembly-diagnosis",daemon=True).start()

    def select_diagnosis_sample(self,_event=None):
        if self.running or not self.result:return
        selected=self.diagnosis_results.get(self.diagnosis_sample.get())
        if selected is None:return
        self.show_result(dict(self.result,example=selected))
        self.tabs.select(self.graph_tab)

    def poll(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "progress":
                    self.log_message(payload)
                    self.status.set(f"运行中 · {payload}")
                else:
                    self.finish(kind, payload)
        except queue.Empty:
            pass
        except Exception as exc:
            self.error.set(f"结果显示失败：{exc}；生成文件仍可在项目目录查看。")
            self.log_message(traceback.format_exc())
        finally:
            self.root.after(80, self.poll)

    def finish(self, kind: str, payload: tuple) -> None:
        params, detail = payload
        elapsed = time.monotonic() - self.start_time
        self.running = False
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.trace_button.configure(state="normal")
        for widget in self.input_widgets:
            widget.configure(state="readonly" if isinstance(widget, ttk.Combobox) else "normal")
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "dataset": params["dataset_name"], "status": "完成" if kind == "complete" else "失败", "elapsed": round(elapsed, 2), "parameters": params}
        if kind == "complete":
            entry["result"] = detail
            try:
                self.show_result(detail)
                if self.current_task=="diagnosis":self.tabs.select(self.diagnosis_tab)
                self.status.set(f"已完成 · {params['dataset_name']} · {elapsed:.1f} 秒")
            except Exception as exc:
                self.error.set(f"生成已完成，但结果显示失败：{exc}")
                self.status.set("数据已生成 · 请查看日志")
                self.log_message(traceback.format_exc())
            self.log_message(f"已完成，数据：{detail.get('dataset_dir', '')}")
        else:
            entry["error"] = detail
            self.error.set("运行失败，请查看日志中的具体错误。")
            if self.current_task=="diagnosis":self.diagnosis_caption.set("溯源失败，请查看日志")
            self.status.set("运行失败 · 参数与界面已恢复，可修改后重试")
            self.log_message(detail)
        self.history.insert(0, entry)
        self.history = self.history[:200]
        if self.persist:
            try:
                atomic_write_json(STATE_ROOT / "history.json", self.history)
            except OSError as exc:
                self.log_message(f"历史记录保存失败：{exc}")
        self.refresh_history()

    def resolve_node(self, value: Any) -> str:
        if isinstance(value, dict):
            candidate = value.get("node") or value.get("id")
            if candidate:
                return str(candidate)
            for node in self.graph.get("nodes", []):
                if node.get("stage") == value.get("stage") and node.get("point") == value.get("point"):
                    return str(node["id"])
            return ""
        return "" if value is None else str(value)

    def show_result(self, result: dict) -> None:
        self.result = result
        self.diagnosis_dataset.set(str(result.get("dataset_dir","")))
        if result.get("task")=="diagnosis":
            rows=read_json(Path(result["files"]["traces"]),[])
            self.diagnosis_results={row["sample_id"]:row for row in rows}
            self.diagnosis_sample_box.configure(values=tuple(self.diagnosis_results))
            self.diagnosis_sample.set(result.get("example",{}).get("sample_id",""))
            self.diagnosis_caption.set(f"已完成 · {len(rows)} 条测试流程")
        else:
            self.diagnosis_results={};self.diagnosis_sample_box.configure(values=())
            self.diagnosis_caption.set("数据已载入")
        graph_file = Path(result["graph_dir"]) / "fixed_graph.json" if result.get("graph_dir") else None
        graph = read_json(graph_file, None) if graph_file else None
        if isinstance(graph, dict) and "nodes" in graph and "edges" in graph:
            self.graph = graph
        else:
            if graph_file:
                self.log_message(f"未找到完整图文件，显示模型结构预览：{graph_file}")
            from assembly_sim.graph import build_graph
            meta=read_json(Path(result.get("dataset_dir",""))/"metadata.json",{})
            self.graph = build_graph(forward=meta.get("generation_mode","cad_forward")=="cad_forward",
                controller_policy=meta.get("controller_policy","body_relative"),point_names=meta.get("point_names",POINTS))
        example = result.get("example") or {}
        self.truth_annotations = read_json(Path(result.get('graph_dir',''))/'ground_truth.json',None)
        if not isinstance(self.truth_annotations,dict):
            from assembly_sim.diagnosis import truth_annotations
            self.truth_annotations=truth_annotations(Path(result.get('dataset_dir','')))
        metrics = result.get("metrics") or {}
        text = ["本次结果", "", f"数据集：{result.get('dataset_dir', '')}", f"图与分析：{result.get('graph_dir', '')}", "", f"示例流程：{example.get('sample_id', '无')}", f"终检质量缺陷：{'是' if example.get('quality_defect') else '否'}", f"终检观测点：{example.get('observed_node', '无')}", f"推断根因：{example.get('predicted_root') or '无'}", f"真实根因：{self.truth_label(example.get('sample_id'))}", f"溯源状态：{example.get('trace_status', '—')}", "", "验证指标："]
        text.extend(f"  {key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}" for key, value in metrics.items() if key != "interpretation")
        metadata=read_json(Path(result.get("dataset_dir",""))/"metadata.json",{})
        if metadata.get("schema_version",0)>=5 and result.get("task")!="diagnosis" and not result.get("metrics"):
            counts=metadata.get("splits",{})
            text=["数据生成完成", "", f"数据集：{result.get('dataset_dir','')}",
                  f"总流程数：{sum(counts.values())}",f"训练：{counts.get('train',0)}；测试：{counts.get('test',0)}",
                  ]
        self.set_text(self.summary, "\n".join(text))
        self.ranking.delete(*self.ranking.get_children())
        ranking = example.get("ranking") or []
        for item in ranking[:20]:
            if isinstance(item, dict):
                self.ranking.insert("", "end", values=tuple(self.pretty(item.get(key, "")) for key in ("node", "stage", "point", "score", "residual_mm")))
        self.draw_graph()
        directory = Path(result.get("dataset_dir", ""))
        metadata = read_json(directory / "metadata.json", {})
        if metadata.get("generation_mode") in ("cad_forward", "cad_reverse"):
            try:
                self.cad_tab.load_dataset(directory, str(example.get("sample_id", "test_00000")))
            except ValueError as exc:
                self.log_message(str(exc))
        else:
            self.cad_tab.clear()

    @staticmethod
    def pretty(value: Any) -> Any:
        return f"{value:.5g}" if isinstance(value, float) else value

    def open_cad_preview(self) -> None:
        self.tabs.select(self.cad_tab)

    def load_cad_directory(self, directory: Path) -> None:
        if self.running:
            self.error.set("数据生成中，请完成后再切换数据集。")
            return
        directory = directory.expanduser().resolve()
        summary = read_json(PROJECT_ROOT / "graph" / directory.name / "summary.json", None)
        if isinstance(summary, dict) and Path(summary.get("dataset_dir", "")).resolve() == directory:
            self.show_result(summary)
        else:
            self.show_result(dict(dataset_dir=str(directory), example=dict(sample_id="test_00000")))
            self.set_text(self.summary, f"已载入数据集：{directory}\n\n可在三维装配页的表格检查坐标；因果图展示已知装配结构。")
        self.open_cad_preview()

    def on_tab_changed(self, _event=None) -> None:
        if self.tabs.select() != str(self.cad_tab):
            self.cad_tab.stop()

    def open_cad_model(self) -> None:
        self.open_cad_file("cad/aircraft.FCStd")

    def open_cad_file(self, name: str) -> None:
        if not self.result:
            self.error.set("请先运行 CAD 数据生成。")
            return
        path = Path(self.result["dataset_dir"]) / name
        if not path.is_file():
            self.error.set("本次数据没有对应的 CAD 文件，请先生成或载入 CAD 数据。")
            return
        if name.endswith(".FCStd"):
            import subprocess
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-a", "FreeCAD", str(path)])
            elif sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", str(path)])
        else:
            webbrowser.open(path.as_uri())

    def show_help(self):
        dialog=tk.Toplevel(self.root)
        dialog.title("仿真参数、数据与红线说明")
        dialog.geometry("900x720")
        frame=ttk.Frame(dialog,padding=12); frame.pack(fill="both",expand=True)
        text=tk.Text(frame,wrap="word",font=("Helvetica",13),padx=12,pady=12)
        scroll=ttk.Scrollbar(frame,command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y");text.pack(fill="both",expand=True)
        text.insert("1.0",(PROJECT_ROOT/"docs/ui_guide.md").read_text(encoding="utf-8"))
        text.configure(state="disabled")

    def configure_cad(self) -> None:
        if self.running: return
        if getattr(self,"cad_editor",None) is not None and self.cad_editor.winfo_exists():
            self.cad_editor.lift();return
        from assembly_sim.cad_editor import CADSettingsDialog
        def save(values):
            for key, value in values.items(): self.vars[key].set(json.dumps(value))
            self.save_settings()
            counts = {p:len(rows) for p,rows in values['cad_checkpoints'].items()}
            self.status.set(f"CAD 设置已保存 · 机头 {counts['A']} / 机身 {counts['B']} / 机尾 {counts['C']} 点；用于下一次生成")
        keys=('cad_geometry','cad_motion','forward_settings','cad_checkpoints')
        self.cad_editor=CADSettingsDialog(self.root,{k:json.loads(self.vars[k].get()) for k in keys},save)

    def draw_graph(self) -> None:
        from assembly_sim.graph import graph_layout, inference_path_edges, directed_path
        c = self.canvas; c.delete("all")
        positions, bands, width, height = graph_layout(self.graph,self.show_initial.get())
        first = 0 if self.show_initial.get() else 1
        for stage in range(first,5):
            c.create_text(125+(stage-first)*290,30,text=f"{stage} · {STAGES[stage]}",font=("Helvetica",13,"bold"),fill="#294c65")
        for part,(y,h) in bands.items():
            c.create_rectangle(25,y,width-25,y+h,fill="#f4f8fb",outline="#dde8ef")
            c.create_text(35,y+15,text=part+" "+dict(A="机头",B="机身",C="机尾")[part],anchor="w",fill="#536e81")
        nodes={n['id']:n for n in self.graph['nodes']}
        example=(self.result or {}).get('example') or {}
        source=self.resolve_node(example.get('predicted_root'));target=self.resolve_node(example.get('observed_node'))
        path=example.get('path',directed_path(self.graph,source,target))
        highlight=inference_path_edges(self.graph,path)
        truth=getattr(self,'truth_annotations',{}).get(example.get('sample_id'),{})
        true_root=truth.get('root') if truth.get('injected') is True else None
        if example.get('sample_id'):
            self.diagnosis_caption.set(f"{example['sample_id']} · 推断根因：{source or '无'} · 真实根因：{self.truth_label(example['sample_id'])}")
        for edge in sorted(self.graph['edges'],key=lambda e:(e['source'],e['target']) in highlight):
            a,b=edge['source'],edge['target']
            if a not in positions or b not in positions:continue
            x,y=positions[a];xx,yy=positions[b]
            x+=54 if nodes[a]['kind']=='action' else 8
            xx-=54 if nodes[b]['kind']=='action' else 8
            marked=(a,b) in highlight
            color='#cf4238' if marked else '#b8c8d3' if edge['kind']=='state' else '#608cb1'
            c.create_line(x,y,(x+xx)/2,y,(x+xx)/2,yy,xx,yy,smooth=True,splinesteps=20,arrow='last',fill=color,width=2.8 if marked else 1,tags=('causal_edge','inferred_path' if marked else 'structural_edge',a+'->'+b))
        for name,(x,y) in positions.items():
            node=nodes[name];color='#cf4238' if name in (source,target) else '#294c65'
            if node['kind']=='action':
                label={1:'调姿 '+node['moved_parts'],2:'前对合 A',3:'后对合 C',4:'再调姿 ABC'}[node['stage']]
                c.create_rectangle(x-54,y-20,x+54,y+20,fill='#e7f2fa',outline=color,width=1.5)
                c.create_text(x,y,text=label+'\nR, t',fill=color,font=('Helvetica',11))
            else:
                c.create_oval(x-8,y-8,x+8,y+8,fill='white',outline=color,width=1.8)
                c.create_text(x+12,y,text=node['point'],anchor='w',fill=color,font=('Helvetica',11))
        if true_root in positions:
            x,y=positions[true_root];action=nodes[true_root]['kind']=='action'
            dx,dy=(59,25) if action else (13,13)
            c.create_rectangle(x-dx,y-dy,x+dx,y+dy,outline='#18865b',width=3,tags=('true_root',true_root))
            c.create_text(x,y-dy-10,text='真实根因',fill='#18865b',font=('Helvetica',11,'bold'),tags='true_root')
        if source in positions:
            x,y=positions[source]
            c.create_text(x,y+35,text='推断根因',fill='#cf4238',font=('Helvetica',11,'bold'),tags='predicted_root')
        c.configure(scrollregion=(0,0,width,height))

    def truth_label(self,sample_id):
        truth=getattr(self,'truth_annotations',{}).get(sample_id)
        if not truth:return '未知'
        if truth.get('injected') is False:return '无（未注入）'
        root=truth.get('root')
        return root if root in {n['id'] for n in self.graph['nodes']} else '未知'

    def open_svg(self) -> None:
        if not self.result:
            self.error.set("请先载入数据或运行溯源")
            return
        from assembly_sim.graph import render_svg, directed_path
        example=self.result.get('example') or {}
        source=example.get('predicted_root');target=example.get('observed_node')
        truth=getattr(self,'truth_annotations',{}).get(example.get('sample_id'),{})
        directory=Path(self.result.get('graph_dir') or PROJECT_ROOT/'graph'/'preview')
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'selected_trace.svg'
        render_svg(path,self.graph,source,target,true_root=truth.get('root') if truth.get('injected') is True else None,
                   inferred_path=example.get('path',directed_path(self.graph,source,target)))
        webbrowser.open(path.resolve().as_uri())

    def refresh_history(self) -> None:
        self.history_table.delete(*self.history_table.get_children())
        for index, entry in enumerate(self.history):
            if isinstance(entry, dict):
                self.history_table.insert("", "end", iid=str(index), values=tuple(entry.get(key, "") for key in ("time", "dataset", "status", "elapsed")))

    def open_history(self, _event: Any = None) -> None:
        if self.running:
            return
        selection = self.history_table.selection()
        if not selection:
            return
        entry = self.history[int(selection[0])]
        if isinstance(entry.get("result"), dict):
            self.show_result(entry["result"])
            self.tabs.select(self.overview_tab)
            self.status.set(f"查看历史 · {entry.get('time')} · {entry.get('dataset')}")
        elif entry.get("error"):
            self.log_message(entry["error"])
            self.tabs.select(self.log_tab)

    def load_preset(self) -> None:
        if self.running:
            return
        values = self.presets.get(self.preset_name.get())
        if isinstance(values, dict):
            for key, value in values.items():
                if key in self.vars:
                    self.vars[key].set(str(value))
            self.error.set("")

    def save_preset(self) -> None:
        name = simpledialog.askstring("保存参数预设", "预设名称：", parent=self.root)
        if not name or not name.strip():
            return
        self.presets[name.strip()] = {key: var.get() for key, var in self.vars.items()}
        try:
            if self.persist:
                atomic_write_json(STATE_ROOT / "presets.json", self.presets)
            self.preset_combo.configure(values=sorted(self.presets))
            self.preset_name.set(name.strip())
        except OSError as exc:
            self.error.set(f"预设保存失败：{exc}")

    def callback_error(self, exc_type: Any, exc: Exception, tb: Any) -> None:
        self.error.set(f"界面操作失败：{exc}")
        self.log_message("".join(traceback.format_exception(exc_type, exc, tb)))

    def close(self) -> None:
        if self.running:
            self.error.set("仿真仍在运行，请待本次完成后关闭窗口。")
            return
        try:
            self.save_settings()
        except OSError as exc:
            self.log_message(f"设置保存失败：{exc}")
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="飞机四阶段装配仿真桌面界面")
    parser.add_argument("--smoke-test", action="store_true", help="初始化 Tk、绘制结构图并自动关闭，不写状态文件")
    parser.add_argument("--dataset-dir", type=Path, help="启动时在内嵌三维页打开已有 CAD 数据集")
    args = parser.parse_args(argv)
    if tk is None:
        print(f"无法启动桌面界面：当前 Python 缺少 Tkinter（{TK_IMPORT_ERROR}）。请使用包含 Tk 的 Python 运行；命令行仿真不依赖 Tk。", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"无法初始化 Tk 图形环境：{exc}", file=sys.stderr)
        return 2
    app = SimulationApp(root, persist=not args.smoke_test)
    if args.dataset_dir:
        app.load_cad_directory(args.dataset_dir)
    if args.smoke_test:
        app.tabs.select(app.graph_tab)
        root.update_idletasks()
        app.show_initial.set(True)
        app.draw_graph()
        assert len(app.canvas.find_all()) > 100, "Graph canvas was not populated"
        if args.dataset_dir:
            app.tabs.select(app.cad_tab)
            root.update_idletasks()
            assert app.cad_tab.data is not None, "CAD dataset failed to load"
            assert len(app.cad_tab.canvas.find_withtag("cad_mesh")) > 100, "CAD mesh was not embedded"
            assert len(app.cad_tab.table.get_children()) == len(app.cad_tab.data.point_names), "Checkpoint table was not populated"
            app.cad_tab.set_stage(3)
            root.update_idletasks()
            assert "等待再调姿" in app.cad_tab.stage_caption.get()
            app.cad_tab.play(1)
            def verify_animation():
                assert app.cad_tab.stage.get() > 3, "Embedded animation did not advance"
                app.cad_tab.stop()
                print("Embedded CAD smoke test passed: mesh, configured checkpoints, stages and animation.")
                root.destroy()
            root.after(350,verify_animation)
        else:
            root.after(350, root.destroy)
    root.mainloop()
    if args.smoke_test:
        print("Tk smoke test passed: interface initialized; full fixed graph drawn; window closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
