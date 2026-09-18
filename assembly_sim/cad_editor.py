"""CAD settings with real FreeCAD previews and editable body-local surface points."""
from dataclasses import asdict
from pathlib import Path
import copy
import queue
import uuid
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from .cad import CADGeometry, CADMotion, export_cad
from .forward import ForwardSettings, body_frames, target_definition
from .surface import surface_point, random_mesh_candidate
from .checkpoints import checkpoint_definition, as_config

# Label, explanation; all lengths are mm.
DIMENSIONS = {
    'nose_length': ('机头长度', '半椭球从尖端到机头/机身接口的轴向长度；必须 ≥ 接口半径。'),
    'radius': ('三段共用半径', '机身和机尾圆柱半径，也是机头底面半径；共用尺寸保证接口匹配。'),
    'body_length': ('机身长度', '两处装配接口之间的圆柱长度；世界原点独立固定，最终机身位姿可在目标页设置。'),
    'tail_length': ('机尾主筒长度', '从机身/机尾接口到尾端的圆柱长度，不含伸出的出气筒。'),
    'exhaust_radius': ('出气筒半径', '两个出气筒各自的半径；直径为此值的两倍。'),
    'exhaust_length': ('出气筒伸出长度', '沿 +X 方向超过主筒尾端的长度。'),
    'exhaust_offset': ('出气筒中心偏移', '两筒中心分别在 Y=±此值、Z=0；须大于筒半径，且两者之和小于共用半径。'),
    'fin_span': ('单侧尾翼外伸', '每片尾翼在 ±Y 方向伸出主筒侧面的距离。'),
    'fin_root_chord': ('尾翼根部长度', '尾翼连接主筒处沿 X 的长度；不得超过机尾主筒长度。'),
    'fin_tip_chord': ('尾翼端部长度', '尾翼最外端沿 X 的长度；与后掠量之和不得超过根部长度。'),
    'fin_sweep': ('尾翼后掠量', '外端前缘相对根部前缘向 +X 偏移的距离。'),
    'fin_thickness': ('尾翼厚度', '两片水平尾翼沿 Z 的厚度；须小于共用半径。'),
}
PART_FIELDS = {'A': ['nose_length'], 'B': ['radius', 'body_length'],
               'C': ['tail_length','exhaust_radius','exhaust_length','exhaust_offset','fin_span','fin_root_chord','fin_tip_chord','fin_sweep','fin_thickness']}
TARGET_FIELDS = {
    'front_gap_mm': ('前接口目标间隙 / mm', '首次调姿后机头与机身沿 X 保留的距离；前对合时闭合。'),
    'final_x_mm': ('最终 B · tX / mm', '机身局部原点在固定世界坐标 W 中的 X 位置。'),
    'final_y_mm': ('最终 B · tY / mm', '机身局部原点在 W 中的 Y 位置。'),
    'final_z_mm': ('最终 B · tZ / mm', '机身局部原点在 W 中的 Z 位置。'),
    'final_roll_deg': ('最终 B · 绕 X 角 / °', '滚转角；按 R=Rz(偏航) Ry(俯仰) Rx(滚转) 组成旋转矩阵。'),
    'final_pitch_deg': ('最终 B · 绕 Y 角 / °', '俯仰角；三段共享最终装配方向，对合间隙沿该方向。'),
    'final_yaw_deg': ('最终 B · 绕 Z 角 / °', '偏航角；这里只改变目标方向，世界坐标轴始终固定。'),
    'rear_gap_mm': ('后接口目标间隙 / mm', '首次调姿和前对合后，机尾与机身保留的距离；后对合时闭合。'),
}
RANDOM_FIELDS = {
    'initial_rotation_deg': ('初始旋转上限 / °', '每段初始方向随机；旋转角绝对值不超过此值，0 表示无初始旋转。'),
    'initial_translation_mm': ('初始平移范围 / mm', '相对首次调姿目标，各段每个轴独立在 ±此值内随机摆放；不作碰撞求解。'),
    'process_rotation_deg': ('工艺旋转标准差 / °', '每次执行时各旋转向量分量的高斯误差强度；平移和测量误差在主界面设置。'),
}


class PartPreview(tk.Canvas):
    def __init__(self, parent, part, on_pick):
        super().__init__(parent, background='#eaf1f6', highlightthickness=0, height=270)
        self.part, self.on_pick = part, on_pick
        self.vertices = self.triangles = None
        self.points, self.selected = [], None
        self.candidate = None
        self.face_colors = None
        self.extra_axes = []
        self.yaw, self.pitch, self.zoom = -.55, .42, 1.
        self.start = None
        self.job = None
        self.bind('<Configure>', lambda _: self.schedule())
        self.bind('<ButtonPress-1>', self.press)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<ButtonRelease-1>', self.release)
        self.bind('<MouseWheel>', self.wheel)
        self.bind('<Destroy>', self.destroyed)

    def destroyed(self, event):
        if event.widget == self and self.job is not None:
            self.after_cancel(self.job); self.job = None

    def schedule(self):
        if self.job is None: self.job = self.after_idle(self.draw)

    def set_mesh(self, mesh, origin):
        self.vertices = np.asarray(mesh['vertices'], float)-origin
        self.triangles = np.asarray(mesh['triangles'], int)
        self.schedule()

    def press(self, event):
        self.start = [event.x, event.y, event.x, event.y, False]

    def drag(self, event):
        if self.start is None: return
        dx,dy = event.x-self.start[2],event.y-self.start[3]
        if abs(event.x-self.start[0])+abs(event.y-self.start[1]) > 4: self.start[4] = True
        if not self.start[4]:return
        self.yaw += dx*.008; self.pitch = np.clip(self.pitch+dy*.008,-1.4,1.4)
        self.start[2:4] = [event.x,event.y]; self.schedule()

    def release(self, event):
        clicked = self.start is not None and not self.start[4]
        self.start = None
        if not clicked or self.vertices is None or not hasattr(self, 'projected'): return
        # Orthographic view ray in the local body frame; exact CAD intersection
        # is performed by the dialog worker, not snapped to tessellation vertices.
        camera_point=np.array([(event.x-self.screen_centre[0])/self.scale+self.view_centre[0],
                               (event.y-self.screen_centre[1])/self.scale+self.view_centre[1],0.])
        depth=max(np.linalg.norm(self.vertices,axis=1).max()*3,1000.)
        start=(camera_point+[0,0,depth]) @ self.camera
        end=(camera_point-[0,0,depth]) @ self.camera
        self.on_pick(dict(start=start,end=end))

    def wheel(self, event):
        self.zoom=float(np.clip(self.zoom*(1.1 if event.delta>0 else 1/1.1),.4,4)); self.schedule()

    def draw(self):
        self.job=None; self.delete('all')
        w,h=max(self.winfo_width(),1),max(self.winfo_height(),1)
        if self.vertices is None:
            self.create_text(w/2,h/2,text='正在生成 FreeCAD 实体预览…',fill='#537185'); return
        cy,sy,cp,sp=np.cos(self.yaw),np.sin(self.yaw),np.cos(self.pitch),np.sin(self.pitch)
        camera=np.array([[cy,-sy,0],[sy*sp,cy*sp,-cp],[sy*cp,cy*cp,sp]])
        camera_vertices=np.einsum("ij,kj->ik",self.vertices,camera)
        axis_length=max(np.ptp(self.vertices,axis=0).max()*.35,180)
        aids=np.vstack([np.zeros(3),np.eye(3)*axis_length]) @ camera.T
        if self.extra_axes:
            aids=np.vstack([aids,np.asarray([p for _,o,r in self.extra_axes for p in [o,*[o+r[:,i]*axis_length for i in range(3)]]]) @ camera.T])
        extent=np.ptp(np.vstack([camera_vertices,aids]),axis=0)
        centre=(np.vstack([camera_vertices,aids]).max(axis=0)+np.vstack([camera_vertices,aids]).min(axis=0))/2
        scale=min((w-80)/max(extent[0],1),(h-65)/max(extent[1],1))*self.zoom
        def project(points):
            result=np.einsum("ij,kj->ik",np.asarray(points),camera)
            result[:,:2]=(result[:,:2]-centre[:2])*scale+[w/2,h/2]
            return result
        self.camera,self.scale,self.view_centre,self.screen_centre=camera,scale,centre,np.array([w/2,h/2])
        self.projected=project(self.vertices)
        faces=self.projected[self.triangles]
        normals=np.cross(camera_vertices[self.triangles[:,1]]-camera_vertices[self.triangles[:,0]],camera_vertices[self.triangles[:,2]]-camera_vertices[self.triangles[:,0]])
        shade=.6+.4*np.abs(normals[:,2])/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
        base=np.array({'A':[69,167,196],'B':[102,138,197],'C':[220,163,100]}.get(self.part,[102,138,197]))
        for i in np.argsort(faces[:,:,2].mean(axis=1)):
            color='#%02x%02x%02x'%tuple(((base if self.face_colors is None else self.face_colors[i])*shade[i]).astype(int))
            self.create_polygon(*faces[i,:,:2].ravel(),fill=color,outline=color,tags='cad_mesh')
        axes=project(np.vstack([np.zeros(3),np.eye(3)*axis_length]))
        for i,color in enumerate(('#d74743','#279659','#3976d2')):
            self.create_line(*axes[0,:2],*axes[i+1,:2],fill=color,width=2,arrow='last',tags='local_axis')
            self.create_text(axes[i+1,0]+12,axes[i+1,1]-9,text='XYZ'[i]+'_'+self.part,fill=color,tags='local_label')
        self.create_text(axes[0,0],axes[0,1]+14,text='O_'+self.part+' (0,0,0)',fill='#243f53')
        for row in self.points:
            x,y,_=project([row['position']])[0]
            self.create_oval(x-4,y-4,x+4,y+4,fill='#f5cb52' if row['name']==self.selected else '#173648',outline='white',tags='checkpoint')
            self.create_text(x+8,y-9,text=row['name'],anchor='w',fill='#173648')
        if self.candidate is not None:
            x,y,_=project([self.candidate])[0]
            self.create_oval(x-8,y-8,x+8,y+8,outline='#df3f84',width=3,tags='candidate')
            self.create_line(x-13,y,x+13,y,fill='#df3f84',width=2,tags='candidate')
            self.create_line(x,y-13,x,y+13,fill='#df3f84',width=2,tags='candidate')
            self.create_text(x+14,y-15,text='待确认点',anchor='w',fill='#bd2167',tags='candidate')
        for name,origin,rotation in self.extra_axes:
            local_axes=project(np.vstack([origin,*[origin+rotation[:,i]*axis_length*.6 for i in range(3)]]))
            for i,color in enumerate(('#d74743','#279659','#3976d2')):
                self.create_line(*local_axes[0,:2],*local_axes[i+1,:2],fill=color,arrow='last',width=2,tags='body_axis')
                self.create_text(local_axes[i+1,0]+8,local_axes[i+1,1],text='XYZ'[i]+'_'+name,fill=color)
        self.create_text(10,12,text=('世界坐标 W' if self.part=='W' else '局部坐标')+' / mm · 红 X / 绿 Y / 蓝 Z',anchor='nw',fill='#426174')


class TargetPreview(PartPreview):
    def __init__(self,parent):
        super().__init__(parent,'W',lambda _:None)
        self.scene_meshes=None
        self.target_poses=None

    def release(self,event):self.start=None

    def set_targets(self,meshes,geometry,poses):
        if meshes is None:return
        self.scene_meshes=meshes;self.target_poses=poses.copy()
        origins=body_frames(geometry)[:,:3,3]
        vertices,triangles,colors=[],[],[];offset=0
        self.extra_axes=[]
        for b,mesh in enumerate(meshes):
            h=poses[b];local=np.asarray(mesh['vertices'])-origins[b]
            v=np.einsum('ij,kj->ik',local,h[:3,:3])+h[:3,3]
            t=np.asarray(mesh['triangles'],int)
            vertices.append(v);triangles.append(t+offset);offset+=len(v)
            colors.extend([[[69,167,196],[102,138,197],[220,163,100]][b]]*len(t))
            self.extra_axes.append(('ABC'[b],h[:3,3],h[:3,:3]))
        self.vertices=np.concatenate(vertices);self.triangles=np.concatenate(triangles)
        self.face_colors=np.asarray(colors);self.schedule()


class CADSettingsDialog(tk.Toplevel):
    def __init__(self, parent, values, on_save):
        super().__init__(parent)
        self.title('CAD 尺寸、测点与阶段目标')
        self.geometry('1260x850'); self.minsize(1080,760); self.transient(parent)
        self.on_save=on_save; self.busy=False; self.closed=False; self.messages=queue.Queue()
        self.geometry_vars={k:tk.StringVar(self,str(v)) for k,v in asdict(CADGeometry(**values.get('cad_geometry',{}))).items()}
        motion=asdict(CADMotion(**values.get('cad_motion',{})));motion.pop('initial_offset_mm',None)
        self.motion_vars={k:tk.StringVar(self,str(v)) for k,v in motion.items()}
        self.forward_vars={k:tk.StringVar(self,str(v)) for k,v in asdict(ForwardSettings(**values.get('forward_settings',{}))).items()}
        names,local,_=checkpoint_definition(CADGeometry(**values.get('cad_geometry',{})),values.get('cad_checkpoints',{}))
        self.points=as_config(names,local); self.mesh_geometry=None
        self.previews={}; self.tables={}; self.xyz={}
        self.surface_versions={p:0 for p in "ABC"};self.surface_pending=set()
        self.workspace=tempfile.TemporaryDirectory(prefix="aircraft-cad-editor-")
        self.mesh_directory=None;self.manifest=None
        self.status=tk.StringVar(self,'就绪')
        header=ttk.Frame(self,padding=(14,12));header.pack(fill='x')
        ttk.Label(header,text='按部件编辑尺寸与 checkpoint',font=('Helvetica',17,'bold')).pack(anchor='w')
        self.tabs=ttk.Notebook(self);self.tabs.pack(fill='both',expand=True,padx=12)
        for part,title in [('A','机头 A'),('B','机身 B'),('C','机尾 C')]: self.part_tab(part,title)
        self.target_tab()
        footer=ttk.Frame(self,padding=12);footer.pack(fill='x')
        ttk.Label(footer,textvariable=self.status,wraplength=960,foreground='#36576f').pack(side='left',fill='x',expand=True)
        ttk.Button(footer,text='取消',command=self.close).pack(side='right',padx=6)
        self.save_button=ttk.Button(footer,text='校验并保存',command=self.save);self.save_button.pack(side='right')
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.poll_job=self.after(100,self.poll)
        for var in [*self.geometry_vars.values(),*self.motion_vars.values(),*self.forward_vars.values()]:
            var.trace_add('write',lambda *_:self.update_targets())
        self.update_targets();self.refresh_preview()

    def field(self, parent, key, var, spec):
        title,help_text=spec[key]
        row=ttk.Frame(parent);row.pack(fill='x',pady=(8,1))
        ttk.Label(row,text=title).pack(side='left')
        ttk.Entry(row,textvariable=var,width=12).pack(side='right')
        ttk.Label(parent,text=help_text,wraplength=295,foreground='#5c7586',font=('Helvetica',11)).pack(fill='x',pady=(0,4))

    def part_tab(self, part, title):
        tab=ttk.Frame(self.tabs,padding=10);self.tabs.add(tab,text=title)
        left=ttk.Frame(tab,width=340);left.pack(side='left',fill='y',padx=(0,12));left.pack_propagate(False)
        scroll_canvas=tk.Canvas(left,highlightthickness=0,width=310)
        scroll=ttk.Scrollbar(left,command=scroll_canvas.yview);scroll.pack(side='right',fill='y')
        scroll_canvas.configure(yscrollcommand=scroll.set);scroll_canvas.pack(fill='both',expand=True)
        content=ttk.Frame(scroll_canvas);window=scroll_canvas.create_window(0,0,window=content,anchor='nw')
        content.bind('<Configure>',lambda _:scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all')))
        scroll_canvas.bind('<Configure>',lambda e:scroll_canvas.itemconfigure(window,width=e.width))
        ttk.Label(content,text='实体尺寸 / mm',font=('Helvetica',14,'bold')).pack(anchor='w',pady=5)
        for key in PART_FIELDS[part]: self.field(content,key,self.geometry_vars[key],DIMENSIONS)
        if part!='B':
            row=ttk.Frame(content);row.pack(fill='x',pady=8)
            ttk.Label(row,text='共用接口半径 / mm：').pack(side='left')
            ttk.Label(row,textvariable=self.geometry_vars['radius']).pack(side='left')
            ttk.Label(content,text='在「机身 B」中修改共用半径。三段接口同步变化。',wraplength=295,foreground='#5c7586').pack(anchor='w')
        ttk.Separator(content).pack(fill='x',pady=12)
        extent={'A':'X 从 −机头长度/2 到 +机头长度/2；尖端在 −X，接口在 +X。',
                'B':'X 从 −机身长度/2 到 +机身长度/2；两端分别连接机头与机尾。',
                'C':'主筒 X 从 −机尾长度/2 到 +机尾长度/2；出气筒再向 +X 伸出，双翼沿 ±Y。'}[part]
        ttk.Label(content,text='局部原点与方向',font=('Helvetica',13,'bold')).pack(anchor='w')
        ttk.Label(content,text=extent+'\n局部原点不随出气筒和尾翼尺寸变化。',wraplength=295,foreground='#5c7586').pack(anchor='w',pady=6)
        ttk.Button(content,text='应用尺寸并刷新三段预览',command=self.refresh_preview).pack(fill='x',pady=8)
        right=ttk.Frame(tab);right.pack(side='left',fill='both',expand=True)
        preview=PartPreview(right,part,lambda ray:self.request_point(part,ray));preview.pack(fill='both',expand=True)
        self.previews[part]=preview
        ttk.Label(right,text='拖动旋转 · 滚轮缩放 · 单击选点 · 粉色十字：候选点',foreground='#5c7586',wraplength=730).pack(anchor='w',pady=5)
        table_frame=ttk.Frame(right);table_frame.pack(fill='x')
        table=ttk.Treeview(table_frame,columns=('name','x','y','z'),show='headings',height=6,selectmode='browse')
        for key,label in [('name','checkpoint'),('x','局部 X / mm'),('y','局部 Y / mm'),('z','局部 Z / mm')]:
            table.heading(key,text=label);table.column(key,width=135,anchor='center')
        table.pack(side='left',fill='x',expand=True)
        ts=ttk.Scrollbar(table_frame,command=table.yview);ts.pack(side='right',fill='y');table.configure(yscrollcommand=ts.set)
        table.bind('<<TreeviewSelect>>',lambda _:self.select_point(part));self.tables[part]=table
        coords=ttk.Frame(right);coords.pack(fill='x',pady=8)
        self.xyz[part]=[tk.StringVar(self,'0') for _ in range(3)]
        for axis,var in zip('XYZ',self.xyz[part]):
            ttk.Label(coords,text=axis).pack(side='left');ttk.Entry(coords,textvariable=var,width=16).pack(side='left',padx=(3,10))
        actions=ttk.Frame(right);actions.pack(fill='x')
        for label,command in [('新增',lambda:self.edit_point(part,False)),('修改选中',lambda:self.edit_point(part,True)),('删除选中',lambda:self.delete_point(part)),('恢复该段默认点',lambda:self.reset_points(part))]:
            ttk.Button(actions,text=label,command=command).pack(side='left',padx=(0,6))
        random_bar=ttk.Frame(right);random_bar.pack(fill='x',pady=8)
        ttk.Button(random_bar,text='随机一个候选点',command=lambda:self.random_point(part)).pack(side='left')
        ttk.Button(random_bar,text='取消候选点',command=lambda:self.cancel_candidate(part)).pack(side='left',padx=8)
        for var in self.xyz[part]:var.trace_add('write',lambda *_,p=part:self.preview_candidate(p))
        self.refresh_points(part)

    def target_tab(self):
        tab=ttk.Frame(self.tabs,padding=14);self.tabs.add(tab,text='阶段目标与初始工况')
        left_wrap=ttk.Frame(tab,width=340);left_wrap.pack(side='left',fill='y',padx=(0,25));left_wrap.pack_propagate(False)
        sc=tk.Canvas(left_wrap,highlightthickness=0);bar=ttk.Scrollbar(left_wrap,command=sc.yview)
        bar.pack(side='right',fill='y');sc.pack(fill='both',expand=True);sc.configure(yscrollcommand=bar.set)
        left=ttk.Frame(sc);window=sc.create_window(0,0,window=left,anchor='nw')
        left.bind('<Configure>',lambda _:sc.configure(scrollregion=sc.bbox('all')))
        sc.bind('<Configure>',lambda e:sc.itemconfigure(window,width=e.width))
        ttk.Label(left,text='固定阶段目标',font=('Helvetica',14,'bold')).pack(anchor='w')
        for key,var in self.motion_vars.items():self.field(left,key,var,TARGET_FIELDS)
        ttk.Separator(left).pack(fill='x',pady=12)
        ttk.Label(left,text='初始工况与执行误差',font=('Helvetica',14,'bold')).pack(anchor='w')
        for key,var in self.forward_vars.items():self.field(left,key,var,RANDOM_FIELDS)
        right=ttk.Frame(tab);right.pack(side='left',fill='both',expand=True)
        ttk.Label(right,text='目标位姿预览 · 局部坐标 → 世界坐标',font=('Helvetica',14,'bold')).pack(anchor='w',pady=(0,10))
        stage_bar=ttk.Frame(right);stage_bar.pack(fill='x',pady=6)
        ttk.Label(stage_bar,text='预览阶段').pack(side='left')
        self.target_stage=tk.StringVar(self,'4 再调姿')
        box=ttk.Combobox(stage_bar,textvariable=self.target_stage,values=['1 首次调姿','2 前对合','3 后对合','4 再调姿'],state='readonly',width=16)
        box.pack(side='left',padx=8);box.bind('<<ComboboxSelected>>',lambda _:self.update_targets())
        self.target_preview=TargetPreview(right);self.target_preview.pack(fill='both',expand=True)
        ttk.Label(right,text='拖动旋转 · 滚轮缩放 · W 固定，A/B/C 坐标轴随目标位姿移动',foreground='#5c7586').pack(anchor='w')
        text_frame=ttk.Frame(right);text_frame.pack(fill='x',pady=8)
        self.target_text=tk.Text(text_frame,height=10,wrap='word',font=('Menlo',12),background='#f1f6fa',relief='flat',padx=12,pady=12,state='disabled')
        scroll=ttk.Scrollbar(text_frame,command=self.target_text.yview);scroll.pack(side='right',fill='y')
        self.target_text.configure(yscrollcommand=scroll.set);self.target_text.pack(side='left',fill='both',expand=True)

    def parameters(self, with_points=True):
        geometry=CADGeometry(**{k:float(v.get()) for k,v in self.geometry_vars.items()})
        motion=CADMotion(**{k:float(v.get()) for k,v in self.motion_vars.items()})
        forward=ForwardSettings(**{k:float(v.get()) for k,v in self.forward_vars.items()})
        if with_points: checkpoint_definition(geometry,self.points)
        m=asdict(motion);m.pop('initial_offset_mm',None)
        return dict(cad_geometry=asdict(geometry),cad_motion=m,forward_settings=asdict(forward),cad_checkpoints=copy.deepcopy(self.points))

    def update_targets(self):
        try:
            values=self.parameters(False);g=CADGeometry(**values['cad_geometry']);m=CADMotion(**values['cad_motion'])
            _,poses,_=target_definition(g,m)
            lines=['三段目标共用 R（局部 → W）：',*['  '+str([round(float(v),6) for v in row]) for row in poses[3,1,:3,:3]],'平移 t / mm：']
            for k,title in enumerate(['1 首次调姿：三段同轴，留前后间隙','2 前对合：机头到位，机尾保留间隙','3 后对合：三段到位','4 再调姿：与阶段 3 相同的完美目标']):
                lines.append(title)
                for b,part in enumerate('ABC'):lines.append(f"  {part}  t = ("+', '.join(f'{v:.3f}' for v in poses[k,b,:3,3])+')')
                lines.append('')
            text='\n'.join(lines)
            if self.manifest and self.mesh_geometry==values['cad_geometry']:
                self.target_preview.set_targets(self.manifest['meshes'],g,poses[int(self.target_stage.get()[0])-1])
            elif self.manifest:
                self.target_preview.vertices=None;self.target_preview.schedule()
        except (ValueError,TypeError) as exc:text=f'参数待修正：{exc}'
        self.target_text.configure(state='normal');self.target_text.delete('1.0','end');self.target_text.insert('1.0',text);self.target_text.configure(state='disabled')

    def refresh_points(self, part, selected=None):
        table=self.tables[part];table.delete(*table.get_children())
        for row in self.points[part]: table.insert('','end',iid=row['name'],values=(row['name'],*(f'{v:.6f}' for v in row['position'])))
        self.previews[part].points=self.points[part];self.previews[part].selected=selected;self.previews[part].schedule()
        if selected:table.selection_set(selected);table.see(selected)

    def select_point(self, part):
        selected=self.tables[part].selection()
        if not selected:return
        row=next(r for r in self.points[part] if r['name']==selected[0])
        self.pick(part,row['position'],message=False)
        self.previews[part].selected=selected[0];self.previews[part].schedule()

    def pick(self, part, xyz, message=True):
        self.surface_versions[part]+=1;self.surface_pending.discard(part)
        self._setting_fields=True
        try:
            for var,v in zip(self.xyz[part],xyz):var.set(f'{v:.12g}')
        finally:self._setting_fields=False
        self.previews[part].candidate=np.asarray(xyz,float);self.previews[part].schedule()
        if message:self.status.set(f'{part} 候选点已选取')

    def edit_point(self, part, update):
        if self.busy:return
        if part in self.surface_pending:self.status.set('请等待候选点计算完成后再确认');return
        try:
            xyz=[float(v.get()) for v in self.xyz[part]]
            if not np.isfinite(xyz).all():raise ValueError('坐标必须是有限数值')
            selected=self.tables[part].selection()
            if update:
                if not selected:raise ValueError('请先选中要修改的 checkpoint')
                name=selected[0]
            else:name=part+str(max([int(p['name'][1:]) for p in self.points[part]],default=0)+1)
            if any(p['name']!=name and np.allclose(p['position'],xyz,atol=1e-7,rtol=0) for p in self.points[part]):raise ValueError('该坐标已存在，不能重复添加')
            if update:next(p for p in self.points[part] if p['name']==name)['position']=xyz
            else:self.points[part].append(dict(name=name,position=xyz))
            self.cancel_candidate(part);self.refresh_points(part,name);self.status.set(f'{name} 已{"修改" if update else "新增"}；保存时统一检查 CAD 表面和配准条件。')
        except ValueError as exc:self.status.set(str(exc))

    def delete_point(self, part):
        if self.busy:return
        selected=self.tables[part].selection()
        if not selected:self.status.set('请先选择要删除的 checkpoint');return
        self.points[part]=[r for r in self.points[part] if r['name']!=selected[0]]
        self.refresh_points(part);self.status.set(f'{selected[0]} 已删除；当前 {len(self.points[part])} 点，保存时至少需 3 个不共线点。')

    def reset_points(self, part):
        if self.busy:return
        try:
            g=CADGeometry(**self.parameters(False)['cad_geometry']);names,local,_=checkpoint_definition(g)
            self.points[part]=as_config(names,local)[part];self.refresh_points(part)
            self.status.set(f'{part} 已恢复当前尺寸下的 3 个默认表面点。')
        except ValueError as exc:self.status.set(str(exc))

    def preview_candidate(self, part):
        if not getattr(self,'_setting_fields',False):
            self.surface_versions[part]+=1;self.surface_pending.discard(part)
        try:
            xyz=np.array([float(v.get()) for v in self.xyz[part]])
            self.previews[part].candidate=xyz if np.isfinite(xyz).all() else None
        except ValueError:self.previews[part].candidate=None
        self.previews[part].schedule()

    def cancel_candidate(self,part):
        self.surface_versions[part]+=1;self.surface_pending.discard(part)
        for v in self.xyz[part]:v.set('')
        self.previews[part].candidate=None;self.previews[part].schedule()
        self.status.set('已取消候选点')

    def request_point(self,part,query):
        if self.busy:return
        try:
            if self.parameters(False)['cad_geometry']!=self.mesh_geometry or self.mesh_directory is None:
                raise ValueError('尺寸已变化，请先应用尺寸并刷新预览，再选点')
        except ValueError as exc:self.status.set(str(exc));return
        for var in self.xyz[part]:var.set('')
        self.previews[part].candidate=None;self.previews[part].schedule()
        self.surface_versions[part]+=1;version=self.surface_versions[part]
        self.surface_pending.add(part);directory=self.mesh_directory
        self.status.set(f'正在计算 {part} 的表面候选点…')
        def worker():
            try:
                point=surface_point(directory/f'{part}_local.step',**query)
                self.messages.put(('surface',part,version,directory,point,None))
            except Exception as exc:self.messages.put(('surface',part,version,directory,None,str(exc)))
        threading.Thread(target=worker,daemon=True,name='cad-surface-pick').start()

    def random_point(self, part):
        if self.busy:return
        preview=self.previews[part]
        if preview.vertices is None:self.status.set('请等待模型预览完成');return
        point=random_mesh_candidate(preview.vertices,preview.triangles)
        self.request_point(part,dict(point=point))

    def refresh_preview(self):self.build(False)
    def save(self):self.build(True)

    def build(self, save):
        if self.busy:return
        try:values=self.parameters(save)
        except (ValueError,TypeError) as exc:self.status.set(f'参数无效：{exc}');return
        self.busy=True;self.save_button.configure(state='disabled')
        for part in 'ABC':self.surface_versions[part]+=1
        self.surface_pending.clear()
        self.status.set('FreeCAD 正在校验实体与 checkpoint 表面…' if save else 'FreeCAD 正在按尺寸生成三段预览…')
        def worker():
            try:
                directory=Path(self.workspace.name)/uuid.uuid4().hex
                manifest=export_cad(directory,CADGeometry(**values['cad_geometry']),values['cad_checkpoints'] if save else None)
                self.messages.put(('done',save,values,manifest,directory))
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True,name='cad-settings-preview').start()

    def poll(self):
        if self.closed:return
        try:
            item=self.messages.get_nowait()
            if item[0]=='surface':
                _,part,version,directory,point,error=item
                if version==self.surface_versions[part] and directory==self.mesh_directory:
                    self.surface_pending.discard(part)
                    if self.parameters(False)['cad_geometry']!=self.mesh_geometry:self.status.set('尺寸已变化，本次候选点已丢弃，请刷新预览。')
                    elif error:self.status.set(error)
                    elif point is None:self.status.set('此处未命中模型表面，请旋转模型后点击可见表面。')
                    else:self.pick(part,point)
                self.poll_job=self.after(30,self.poll);return
            self.busy=False;self.save_button.configure(state='normal')
            if item[0]=='error':self.status.set(item[1])
            else:
                _,save,values,manifest,directory=item
                current=self.parameters(False)
                if values!=current:
                    self.status.set('建模期间参数有变化，请重新刷新或保存；本次结果未应用。')
                elif save:
                    self.on_save(values);self.close();return
                else:
                    origins=body_frames(CADGeometry(**values['cad_geometry']))[:,:3,3]
                    for b,part in enumerate('ABC'):self.previews[part].set_mesh(manifest['meshes'][b],origins[b])
                    self.mesh_geometry=values['cad_geometry'];self.mesh_directory=directory;self.manifest=manifest
                    self.update_targets()
                    self.status.set('预览已更新')
        except queue.Empty:pass
        except (ValueError,TypeError) as exc:self.status.set(f'参数待修正：{exc}')
        self.poll_job=self.after(100,self.poll)

    def close(self):
        self.closed=True
        if self.poll_job:self.after_cancel(self.poll_job);self.poll_job=None
        self.destroy()
        self.workspace.cleanup()
