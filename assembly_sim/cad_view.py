"""Native Tk viewer for the same FreeCAD meshes and plans used by generation.

No browser, web server, or CAD GUI process is needed to inspect a saved dataset.
Only the Tk main thread calls widget methods or advances animation callbacks.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from .checkpoints import point_groups, validate_local
from .cad import CADGeometry, world_frame_spec


STATES = ("初始 · 调姿前参考位置", "调姿后 · 三段分离", "前对合后 · AB 连接",
          "后对合后 · ABC 连接，等待再调姿", "再调姿后 · 最终完美目标")
COLORS = np.array([[69, 167, 196], [102, 138, 197], [220, 163, 100]])


def interpolate_pose(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate rotation on SO(3), translation linearly; preserve endpoints."""
    if fraction <= 0:
        return first.copy()
    if fraction >= 1:
        return second.copy()
    result = first.copy()
    for part in range(3):
        a, b = first[part, :3, :3], second[part, :3, :3]
        relative = b @ a.T
        angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
        if angle < 1e-10:
            rotation = a
        else:
            if np.pi-angle < 1e-6:
                # The rotation axis is the eigenvector with eigenvalue +1.
                _, vectors = np.linalg.eigh((relative+relative.T)/2)
                axis = vectors[:, -1]
            else:
                axis = np.array([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0],
                                 relative[1,0]-relative[0,1]])/(2*np.sin(angle))
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            theta = angle*fraction
            rotation = (np.eye(3)+np.sin(theta)*skew+(1-np.cos(theta))*(skew@skew))@a
        result[part, :3, :3] = rotation
        result[part, :3, 3] = first[part, :3, 3]*(1-fraction)+second[part, :3, 3]*fraction
    return result


class CADViewData:
    """Viewer may load evaluator truth for display only; diagnostics remain observation-only."""
    def __init__(self, directory: Path | str):
        self.directory = Path(directory).resolve()
        with (self.directory / "cad/geometry.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        self.point_names = tuple(manifest["point_names"])
        self.groups = point_groups(self.point_names)
        self.checkpoints = np.asarray(manifest["checkpoints"], float)
        self.fixed_targets = False
        self.forward = False
        self.body_frames = manifest.get("body_frames", {})
        self.world_frame = manifest.get("world_frame",world_frame_spec(CADGeometry(**manifest.get("geometry",{}))))
        if self.checkpoints.shape != (len(self.point_names),3) or not np.isfinite(self.checkpoints).all():
            raise ValueError("CAD checkpoint 数据无效")
        self.meshes = []
        if len(manifest["meshes"]) != 3:
            raise ValueError("CAD 需要三个部件")
        for part, mesh in zip("ABC", manifest["meshes"]):
            vertices, triangles = np.asarray(mesh["vertices"], float), np.asarray(mesh["triangles"])
            if (mesh["part"] != part or vertices.ndim != 2 or vertices.shape[1] != 3
                    or not len(vertices) or not np.isfinite(vertices).all() or triangles.ndim != 2
                    or triangles.shape[1] != 3 or not len(triangles) or triangles.dtype.kind not in "iu"
                    or triangles.min() < 0 or triangles.max() >= len(vertices)):
                raise ValueError("CAD 网格无效")
            self.meshes.append((vertices, triangles))
        if len(self.meshes) != 3:
            raise ValueError("CAD 需要三个部件")
        self.plans, self.poses, self.observed = {}, {}, {}
        self.actual, self.actual_poses = {}, {}
        metadata_path = self.directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("generation_mode") == "cad_forward":
            from .forward import fit_pose
            self.forward = True
            spec = json.loads((self.directory / "target_states.json").read_text())
            local = np.asarray(spec["local_checkpoints"],float)
            target_poses = np.asarray(spec["target_poses"],float)
            targets = np.asarray(spec["target_checkpoints"],float)
            if local.shape != (len(self.point_names),3) or target_poses.shape != (4,3,4,4) or targets.shape != (4,len(self.point_names),3):
                raise ValueError("Invalid local frames/target shapes")
            from .forward import world_points
            r = target_poses[...,:3,:3]
            if (not all(np.isfinite(a).all() for a in (local,target_poses,targets))
                or not np.allclose(r@r.swapaxes(-1,-2),np.eye(3),atol=1e-7)
                or not np.allclose(np.linalg.det(r),1,atol=1e-7)
                or not np.allclose(target_poses[...,3,:],[0,0,0,1])
                or not np.allclose(np.stack([world_points(local,h,self.point_names) for h in target_poses]),targets,atol=1e-6,rtol=0)
                or not np.allclose(targets[3],self.checkpoints,atol=1e-6,rtol=0)
                or not np.allclose(target_poses[2],target_poses[3],atol=1e-8,rtol=0)):
                raise ValueError("CAD local frames and fixed targets do not match")
            validate_local(self.point_names, local)
            if tuple(metadata.get("point_names", self.point_names)) != self.point_names or tuple(spec.get("point_names", self.point_names)) != self.point_names:
                raise ValueError("CAD、观测与目标的测点编号不一致")
            self.final_poses = target_poses[3]
            self.target_render_poses = target_poses @ np.linalg.inv(self.final_poses)
            with np.load(self.directory / "observations.npz",allow_pickle=False) as archive:
                for split in metadata["splits"]:
                    y = archive[split]
                    if y.ndim != 4 or not len(y) or y.shape[1:] != (5,len(self.point_names),3) or not np.isfinite(y).all():
                        raise ValueError("Invalid observed coordinates")
                    h = np.empty((len(y),5,3,4,4))
                    for n,row in enumerate(y):
                        for k in range(5):
                            for b in range(3):
                                sl = self.groups[b]
                                h[n,k,b] = fit_pose(local[sl],row[k,sl]) @ np.linalg.inv(self.final_poses[b])
                    self.poses[split],self.observed[split] = h,y.copy()
                    # The stage-0 slot is a display adapter, NOT an initial target.
                    self.plans[split] = np.tile(np.concatenate([targets[:1],targets]),(len(y),1,1,1))
            self.fixed_targets = True
            self.load_actual()
            return

        with np.load(self.directory / "planned_checkpoints.npz", allow_pickle=False) as plans, \
             np.load(self.directory / "planned_transforms.npz", allow_pickle=False) as transforms, \
             np.load(self.directory / "observations.npz", allow_pickle=False) as observed:
            for split in ("train", "calibration", "test"):
                p, h, y = plans[split], transforms[f"{split}_poses"], observed[split]
                if (p.ndim != 4 or not len(p) or p.shape[1:] != (5,len(self.point_names),3) or y.shape != p.shape
                        or h.shape != (len(p),5,3,4,4)
                        or not all(np.isfinite(array).all() for array in (p,h,y))):
                    raise ValueError(f"{split} 的坐标和位姿维度无效")
                r = h[..., :3, :3]
                if (not np.allclose(r@r.swapaxes(-1,-2), np.eye(3), atol=1e-7)
                        or not np.allclose(np.linalg.det(r),1,atol=1e-7)
                        or not np.allclose(h[...,3,:],[0,0,0,1])):
                    raise ValueError(f"{split} 位姿不是刚体变换")
                expected = (np.einsum("nsbij,bpj->nsbpi",r,self.checkpoints.reshape(3,3,3))
                            + h[..., :3,3][...,None,:]).reshape(p.shape)
                if not np.allclose(expected,p,atol=1e-6,rtol=0):
                    raise ValueError(f"{split} CAD 位姿与理想测点不匹配")
                self.plans[split], self.poses[split], self.observed[split] = p,h,y
        self.fixed_targets = all(np.allclose(p[:,3],p[:,4],atol=1e-8,rtol=0) for p in self.plans.values())
        self.load_actual()

    def load_actual(self):
        """Optional simulator truth, exclusively for the actual-state display."""
        from .forward import fit_pose
        path = self.directory / "evaluation/true_trajectories.npz"
        if not path.exists():
            return
        with np.load(path,allow_pickle=False) as archive:
            for split,y in self.observed.items():
                x = archive[split]
                if x.shape != y.shape or not np.isfinite(x).all():
                    raise ValueError("Invalid actual coordinates")
                self.actual[split] = x.copy()
                h = np.empty((len(x),5,3,4,4))
                for n,row in enumerate(x):
                    for k in range(5):
                        for b in range(3):
                            sl = self.groups[b]
                            h[n,k,b] = fit_pose(self.checkpoints[sl],row[k,sl])
                self.actual_poses[split] = h

    def frame(self, split: str, sample: int, stage: float, target=False, actual=False):
        if split not in self.poses or not 0 <= sample < len(self.poses[split]) or not 0 <= stage <= 4:
            raise ValueError("轨迹或阶段超出范围")
        k = min(int(stage),3)
        source = (self.actual_poses if actual and split in self.actual_poses else self.poses)[split][sample]
        if self.forward and target:
            source = np.concatenate([source[:1],self.target_render_poses])
        poses = interpolate_pose(source[k],source[k+1],stage-k)
        from .forward import world_points
        points = world_points(self.checkpoints, poses, self.point_names)
        return poses,points


class CADAssemblyView(ttk.Frame):
    def __init__(self, parent, on_load=None):
        super().__init__(parent, padding=8)
        from .ui_controls import install_checkmarks
        install_checkmarks(self)
        self.on_load = on_load
        self.data = None
        self.yaw, self.pitch, self.zoom = -0.42,0.4,1.0
        self._drag = None
        self._draw_job = self._animation_job = None
        self._direction = 0
        self._last_time = 0
        self._stage_value = 4.0
        self.split = tk.StringVar(self,"test")
        self.sample = tk.StringVar(self,"0")
        self.part = tk.StringVar(self,"A")
        self.point = tk.StringVar(self,"A1")
        self.stage = tk.DoubleVar(self,4)
        self.show_targets = tk.BooleanVar(self,True)
        self.show_world = tk.BooleanVar(self,True)
        self.layer = tk.StringVar(self,"阶段实际")
        self.dataset_caption = tk.StringVar(self,"未载入数据")
        self.stage_caption = tk.StringVar(self,STATES[4])
        self.detail_caption = tk.StringVar(self,"阶段实际、阶段观测、阶段目标 / mm。")
        top = ttk.Frame(self); top.pack(fill="x")
        ttk.Button(top,text="载入已有数据…",command=self.choose_dataset).pack(side="left")
        ttk.Label(top,textvariable=self.dataset_caption).pack(side="left",padx=10)
        bar = ttk.Frame(self); bar.pack(fill="x",pady=8)
        ttk.Label(bar,text="数据划分").pack(side="left")
        self.split_box = ttk.Combobox(bar,textvariable=self.split,values=("train","test"),state="disabled",width=11)
        self.split_box.pack(side="left",padx=5); self.split_box.bind("<<ComboboxSelected>>",self.change_split)
        ttk.Label(bar,text="样本序号").pack(side="left")
        self.sample_box = ttk.Spinbox(bar,textvariable=self.sample,from_=0,to=0,width=6,state="disabled",command=self.change_sample)
        self.sample_box.pack(side="left",padx=5)
        self.sample_box.bind("<Return>",self.change_sample); self.sample_box.bind("<FocusOut>",self.change_sample)
        self.forward_button = ttk.Button(bar,text="▶ 正向装配",command=lambda:self.play(1),state="disabled")
        self.forward_button.pack(side="left",padx=3)
        layer_box = ttk.Combobox(bar,textvariable=self.layer,values=("阶段实际","观测拟合","阶段目标"),state="readonly",width=8)
        layer_box.pack(side="left",padx=3); layer_box.bind("<<ComboboxSelected>>",lambda _:self.schedule_draw())
        ttk.Button(bar,text="重置视角",command=self.reset_camera).pack(side="right")
        self.target_toggle=ttk.Checkbutton(bar,text="阶段目标点",variable=self.show_targets,command=self.schedule_draw)
        self.target_toggle.pack(side="right",padx=6)
        middle = ttk.Frame(self); middle.pack(fill="both",expand=True)
        info = ttk.Frame(middle,width=215,padding=(10,0,0)); info.pack(side="right",fill="y"); info.pack_propagate(False)
        pose_bar = ttk.Frame(info); pose_bar.pack(fill="x",pady=(0,5))
        ttk.Label(pose_bar,text="绝对位姿 R、t").pack(side="left")
        part_box = ttk.Combobox(pose_bar,textvariable=self.part,values=("A","B","C"),state="readonly",width=3)
        part_box.pack(side="right"); part_box.bind("<<ComboboxSelected>>",lambda _:self.schedule_draw())
        self.matrix = tk.Text(info,height=8,width=26,font=("Menlo",9),background="#edf3f7",relief="flat",state="disabled")
        self.matrix.pack(fill="x")
        self.canvas = tk.Canvas(middle,background="#eaf1f6",highlightthickness=0,height=140)
        self.canvas.pack(side="left",fill="both",expand=True)
        self.canvas.bind("<Configure>",lambda _:self.schedule_draw())
        self.canvas.bind("<ButtonPress-1>",self.drag_start)
        self.canvas.bind("<B1-Motion>",self.drag_move)
        self.canvas.bind("<ButtonRelease-1>",lambda _:setattr(self,"_drag",None))
        self.canvas.bind("<MouseWheel>",self.wheel)
        self.canvas.bind("<Button-4>",lambda _:self.change_zoom(1.1))
        self.canvas.bind("<Button-5>",lambda _:self.change_zoom(1/1.1))
        hint_bar = ttk.Frame(self); hint_bar.pack(fill="x",pady=(4,0))
        ttk.Label(hint_bar,text="拖动旋转 · 滚轮缩放",foreground="#627a8a").pack(side="left")
        self.world_toggle=ttk.Checkbutton(hint_bar,text="坐标系",variable=self.show_world,command=self.schedule_draw)
        self.world_toggle.pack(side="right")
        ttk.Label(self,textvariable=self.stage_caption,font=("Helvetica",14,"bold"),foreground="#087e8e").pack(anchor="w",pady=(6,0))
        self.slider = ttk.Scale(self,from_=0,to=4,variable=self.stage,command=self.scrub,state="disabled")
        self.slider.pack(fill="x")
        stages = ttk.Frame(self); stages.pack(fill="x")
        self.stage_buttons = []
        for k,title in enumerate(("0 初始","1 调姿","2 前对合","3 后对合","4 再调姿")):
            button = ttk.Button(stages,text=title,command=lambda s=k:self.set_stage(s),state="disabled")
            button.pack(side="left",expand=True,fill="x",padx=2)
            self.stage_buttons.append(button)
        ttk.Label(self,textvariable=self.detail_caption).pack(anchor="w",pady=(8,4))
        table_frame = ttk.Frame(self); table_frame.pack(fill="x")
        columns = ("point","actual","observed","target")
        self.table = ttk.Treeview(table_frame,columns=columns,show="headings",height=5,selectmode="browse")
        for col,title,width in zip(columns,("测点","阶段实际 (X, Y, Z)","阶段观测 (X, Y, Z)","阶段目标 (X, Y, Z)"),(55,245,245,245)):
            self.table.heading(col,text=title); self.table.column(col,width=width,minwidth=50,anchor="center",stretch=col!="point")
        self.table.grid(row=0,column=0,sticky="nsew")
        scroll = ttk.Scrollbar(table_frame,orient="vertical",command=self.table.yview)
        scroll.grid(row=0,column=1,sticky="ns")
        horizontal = ttk.Scrollbar(table_frame,orient="horizontal",command=self.table.xview)
        horizontal.grid(row=1,column=0,sticky="ew")
        self.table.configure(yscrollcommand=scroll.set,xscrollcommand=horizontal.set)
        table_frame.columnconfigure(0,weight=1)
        self.table.bind("<<TreeviewSelect>>",self.select_point)
        self.bind("<Destroy>",self.on_destroy)

    def clear(self,message="尚未载入 CAD 实体，请生成或载入数据集。"):
        self.stop(); self.data = None
        self.dataset_caption.set(message)
        self.canvas.delete("all"); self.table.delete(*self.table.get_children())
        self.matrix.configure(state="normal"); self.matrix.delete("1.0","end"); self.matrix.configure(state="disabled")
        self.stage_caption.set("初始 → 调姿 → 前对合 → 后对合 → 再调姿")
        for widget in (self.split_box,self.sample_box,self.forward_button,self.slider,*self.stage_buttons):
            widget.configure(state="disabled")
        self.schedule_draw()

    def load_dataset(self,directory, sample_id="test_00000"):
        self.clear("正在读取 CAD 数据…")
        try:
            loaded = CADViewData(directory)
        except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
            self.clear(f"CAD 数据无法载入：{exc}")
            raise ValueError(f"CAD 数据无法载入：{exc}") from exc
        self.data = loaded
        split,_,index = sample_id.rpartition("_")
        self.split.set(split if split in loaded.poses else "test")
        selected = int(index) if index.isdecimal() else 0
        self.sample.set(str(min(selected,len(loaded.poses[self.split.get()])-1)))
        self.split_box.configure(state="readonly",values=tuple(loaded.poses))
        self.sample_box.configure(state="normal",to=len(loaded.poses[self.split.get()])-1)
        for widget in (self.forward_button,self.slider,*self.stage_buttons): widget.configure(state="normal")
        for name in loaded.point_names: self.table.insert("","end",iid=name,values=(name,"","",""))
        self.point.set(loaded.point_names[0]); self.part.set("A")
        self.dataset_caption.set(loaded.directory.name)
        self.layer.set("阶段实际" if loaded.actual else "观测拟合" if loaded.forward else "阶段目标")
        self.reset_camera(); self.set_stage(0 if loaded.forward else 4)

    def choose_dataset(self):
        path = filedialog.askdirectory(parent=self,title="选择包含 cad 和 observations.npz 的数据集目录",
                                       initialdir=Path(__file__).resolve().parents[1]/"data")
        if path:
            try:
                if self.on_load: self.on_load(Path(path))
                else: self.load_dataset(path)
            except ValueError: pass  # The viewer displays the precise error and clears old data.

    def sample_index(self):
        try: index = int(self.sample.get())
        except ValueError: index = 0
        index = max(0,min(index,len(self.data.poses[self.split.get()])-1))
        self.sample.set(str(index))
        return index

    def change_split(self,_event=None):
        if self.data is None: return
        self.sample.set("0")
        self.sample_box.configure(to=len(self.data.poses[self.split.get()])-1)
        self.change_sample()

    def change_sample(self,_event=None):
        if self.data is None: return
        self.stop(); self.sample_index(); self.schedule_draw()

    def set_stage(self,value):
        self.stop(); self._stage_value = float(value); self.stage.set(value); self.schedule_draw()

    def scrub(self,value):
        self.stop(); self._stage_value = float(value); self.schedule_draw()

    def play(self,direction):
        if direction != 1:
            raise ValueError("Only forward playback is supported")
        if self.data is None: return
        if self._direction == direction:
            self.stop(); return
        self.stop()
        value = self.stage.get()
        if direction == 1 and value >= 4: value = 0
        if direction == -1 and value <= 0: value = 4
        self._stage_value = value; self.stage.set(value)
        self._direction = direction; self._last_time = time.perf_counter()
        self.forward_button.configure(text="Ⅱ 暂停")
        self.tick()

    def tick(self):
        self._animation_job = None
        if not self._direction or self.data is None: return
        now = time.perf_counter()
        self._stage_value = np.clip(self._stage_value+self._direction*(now-self._last_time)/1.8,0,4)
        self._last_time = now; self.stage.set(self._stage_value); self.schedule_draw()
        if self._stage_value in (0,4): self.stop()
        else: self._animation_job = self.after(40,self.tick)

    def stop(self):
        self._direction = 0
        if self._animation_job is not None:
            self.after_cancel(self._animation_job); self._animation_job = None
        self.forward_button.configure(text="▶ 正向装配")

    def on_destroy(self,event):
        if event.widget is not self: return
        for job in (self._draw_job,self._animation_job):
            if job is not None: self.after_cancel(job)
        self._draw_job = self._animation_job = None

    def reset_camera(self):
        self.yaw,self.pitch,self.zoom = -0.42,0.4,1.0; self.schedule_draw()

    def drag_start(self,event): self._drag = (event.x,event.y)

    def drag_move(self,event):
        if self._drag is None: return
        self.yaw += (event.x-self._drag[0])*0.005
        self.pitch = float(np.clip(self.pitch+(event.y-self._drag[1])*0.005,-1.4,1.4))
        self._drag = (event.x,event.y); self.schedule_draw()

    def wheel(self,event):
        delta = event.delta if abs(event.delta)<120 else event.delta/120*4
        self.change_zoom(float(np.exp(np.clip(delta*0.025,-0.5,0.5))))

    def change_zoom(self,factor): self.zoom = float(np.clip(self.zoom*factor,0.25,5)); self.schedule_draw()

    def select_point(self,_event=None):
        selected = self.table.selection()
        if selected:
            self.point.set(selected[0]); self.part.set(selected[0][0]); self.schedule_draw()

    def pick_point(self,name):
        self.table.selection_set(name); self.table.see(name)

    def schedule_draw(self):
        if self._draw_job is None: self._draw_job = self.after_idle(self.draw)

    def draw(self):
        self._draw_job = None
        c = self.canvas; c.delete("all")
        width,height = max(c.winfo_width(),1),max(c.winfo_height(),1)
        if self.data is None:
            c.create_text(width/2,height/2,text="请载入 CAD 数据",justify="center",fill="#627e91",font=("Helvetica",14))
            return
        split,index,stage = self.split.get(),self.sample_index(),float(self.stage.get())
        poses,points = self.data.frame(split,index,stage,target=self.layer.get()=="阶段目标",actual=self.layer.get()=="阶段实际")
        cy,sy,cp,sp = np.cos(self.yaw),np.sin(self.yaw),np.cos(self.pitch),np.sin(self.pitch)
        camera = np.array([[cy,-sy,0],[sy*sp,cy*sp,-cp],[sy*cp,cy*cp,sp]])
        reference_span = np.ptp(self.data.checkpoints[:,0])+1800
        scale = min(width/reference_span,height/1700)*self.zoom
        def project(vertices):
            result = np.einsum("ij,kj->ik",vertices,camera)
            result[:,:2] *= scale; result[:,:2] += [width/2,height/2]
            return result
        for x in range(-3000,3001,500):
            line = project(np.array([[x,-2000,-450],[x,2000,-450]],float))
            c.create_line(*line[:,:2].ravel(),fill="#d8e3eb")
        for y in range(-2000,2001,500):
            line = project(np.array([[-3000,y,-450],[3000,y,-450]],float))
            c.create_line(*line[:,:2].ravel(),fill="#d8e3eb")
        faces,fill_colors = [],[]
        for b,(vertices,triangles) in enumerate(self.data.meshes):
            world = np.einsum("ij,kj->ik",vertices,poses[b,:3,:3])+poses[b,:3,3]
            projected = project(world)
            faces.append(projected[triangles])
            camera_vertices = np.einsum("ij,kj->ik",world,camera)
            edges = camera_vertices[triangles]
            normals = np.cross(edges[:,1]-edges[:,0],edges[:,2]-edges[:,0])
            lengths = np.maximum(np.linalg.norm(normals,axis=1),1e-12)
            shade = 0.65+0.35*np.abs(np.einsum("ij,j->i",normals,[0.3,-0.5,0.8])/lengths)
            fill_colors.append(np.clip(COLORS[b]*shade[:,None],0,255).astype(int))
        faces,fill_colors = np.concatenate(faces),np.concatenate(fill_colors)
        for face in np.argsort(faces[:,:,2].mean(axis=1)):
            color = "#%02x%02x%02x" % tuple(fill_colors[face])
            c.create_polygon(*faces[face,:,:2].ravel(),fill=color,outline=color,tags="cad_mesh")
        if self.show_world.get():
            origin = np.asarray(self.data.world_frame["origin"],float)
            for axis in self.data.world_frame["axes"]:
                endpoint = origin+np.asarray(axis["direction"])*axis["length"]
                line = project(np.stack([origin,endpoint]))
                c.create_line(*line[:,:2].ravel(),fill=axis["color"],width=2,arrow="last",
                              arrowshape=(10,12,5),dash=(6,3),tags="world_axis")
                c.create_text(line[1,0]+7,line[1,1]-9,text=axis["name"],fill=axis["color"],
                              font=("Helvetica",11,"bold"),tags="world_label")
            o = project(origin[None])[0]
            c.create_oval(o[0]-3,o[1]-3,o[0]+3,o[1]+3,fill="#20394b",outline="white",tags="world_label")
            c.create_text(o[0]+6,o[1]+12,text="O_W (0,0,0)",anchor="nw",fill="#20394b",tags="world_label")
        if self.show_world.get() and self.data.body_frames:
            for b,name in enumerate("ABC"):
                final = np.asarray(self.data.body_frames[name]["final_pose"])
                local_pose = poses[b] @ final
                origin = local_pose[:3,3]
                for j,color in enumerate(("#d74743","#279659","#3976d2")):
                    line = project(np.stack([origin,origin+local_pose[:3,j]*230]))
                    c.create_line(*line[:,:2].ravel(),fill=color,width=2,arrow="last",tags="body_axis")
                    c.create_text(line[1,0]+4,line[1,1]-7,text="XYZ"[j]+"_"+name,fill=color,tags="body_label")
        for name,(x,y,_) in zip(self.data.point_names,project(points)):
            tag = f"point_{name}"
            c.create_oval(x-4,y-4,x+4,y+4,fill="#f7d067" if self.point.get()==name else "#173648",outline="white",width=2,tags=tag)
            c.create_text(x+7,y-9,text=name,anchor="sw",fill="#173648",font=("Helvetica",11,"bold"),tags=tag)
            c.tag_bind(tag,"<Button-1>",lambda _,n=name:self.pick_point(n))
        c.create_text(12,12,text="A 机头   ·   B 机身   ·   C 机尾",anchor="nw",fill="#476779",font=("Helvetica",11))
        initial_reference=self.data.forward and stage<1
        self.target_toggle.configure(text='首次调姿目标点' if initial_reference else '阶段目标点')
        if self.show_targets.get():
            target_stage=1 if initial_reference else stage
            _,target_points=self.data.frame(split,index,target_stage,target=True)
            for name,(x,y,_) in zip(self.data.point_names,project(target_points)):
                # Draw last and larger than actual markers: visible even when coincident.
                c.create_polygon(x,y-10,x+10,y,x,y+10,x-10,y,fill='',outline='#ffffff',width=5,tags='target_halo')
                c.create_polygon(x,y-10,x+10,y,x,y+10,x-10,y,fill='',outline='#d36c00',width=2.5,tags=('target_point','target_'+name))
                c.create_text(x+13,y+12,text=name+' 目标',anchor='w',fill='#a34e00',font=('Helvetica',10),tags='target_label')
        exact = abs(stage-round(stage))<1e-7
        self.stage_caption.set(STATES[round(stage)] if exact else f"{STATES[int(stage)]} → {STATES[int(stage)+1]}（阶段间预览）")
        self.detail_caption.set(f"{split}_{index:05d} · mm" + ("" if exact else " · 插值"))
        if self.data.forward:
            titles = ("初始 · 随机摆放（无目标状态）", "首次调姿后", "前对合后", "后对合后 · 等待再调姿", "再调姿后")
            title = titles[round(stage)] if exact else titles[int(stage)]+" → "+titles[int(stage)+1]
            self.stage_caption.set(title+" · "+self.layer.get())
            self.detail_caption.set(f"{split}_{index:05d} · {self.layer.get()} · mm")
            if self.layer.get()=="阶段实际" and split not in self.data.actual:
                self.detail_caption.set("实际数据缺失 · 显示观测拟合")
            elif self.layer.get()=="阶段目标" and stage==0:
                self.detail_caption.set("初始状态 · 目标点为首次调姿参考")
        b = "ABC".index(self.part.get())
        h = poses[b]
        if self.data.body_frames:
            h = h @ np.asarray(self.data.body_frames[self.part.get()]["final_pose"])

        matrix = "R =\n"+"\n".join("["+" ".join(f"{v: .5f}" for v in row[:3])+"]" for row in h[:3])+"\n\nt / mm =\n["+", ".join(f"{v:.3f}" for v in h[:3,3])+"]"
        self.matrix.configure(state="normal"); self.matrix.delete("1.0","end"); self.matrix.insert("1.0",matrix); self.matrix.configure(state="disabled")
        def xyz(row): return "("+", ".join(f"{v:.3f}" for v in row)+")"
        for i,name in enumerate(self.data.point_names):
            observed = xyz(self.data.observed[split][index,round(stage),i]) if exact else "—"
            actual = xyz(self.data.actual[split][index,round(stage),i]) if exact and split in self.data.actual else "—"
            target = xyz(self.data.plans[split][index,round(stage),i]) if exact and (stage > 0 or not self.data.forward) else "—"
            self.table.item(name,values=(name,actual,observed,target))
