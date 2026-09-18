"""Standalone offline preview of FreeCAD meshes and the exported process plans."""
import json
from pathlib import Path
from .cad import CADGeometry, world_frame_spec


def write_preview(path, manifest, plans, poses, observed, forward=False, actual=None):
    count = min(len(plans), 20)
    payload = dict(actual=None if actual is None else actual[:count].tolist(), forward=forward, bodyFrames=manifest.get("body_frames",{}), meshes=manifest["meshes"], checkpoints=manifest["checkpoints"],
                   worldFrame=manifest.get("world_frame",world_frame_spec(CADGeometry(**manifest.get("geometry",{})))),
                   pointNames=manifest["point_names"], plans=plans[:count].tolist(),
                   poses=poses[:count].tolist(), observed=observed[:count].tolist(),
                   totalSamples=len(plans), engine=manifest["engine"])
    template = Path(__file__).with_name("cad_preview.html").read_text(encoding="utf-8")
    if forward:
        template = template.replace("从完整飞机，反推各阶段理想目标", "固定 CAD 目标，正向生成装配数据").replace("FreeCAD 逆向数据生成", "FreeCAD 正向仿真")
        start = template.index('<h2>逆向生成次序</h2>')
        end = template.index('<select id="part">',start)
        template = template[:start]+'<h2>正向生成</h2><p>三段随机初态 → 根据观测计算指令 → 加入整段刚体工艺误差 → 更新真实位姿 → 添加观测噪声。图中展示观测拟合位姿，表格同时记录固定目标。阶段 0 没有理想目标。</p><h2>观测拟合位姿（局部 → 世界）</h2>'+template[end:]
        template = template.replace('所有部件以完整装配坐标为参考。','各部件使用自己的局部坐标系。').replace('x_CAD','q_local').replace('正反向变换和执行动作','固定目标和正向执行动作').replace('example_plan.json','target_states.json').replace('彩色实体和实心点为阶段理想位置','彩色实体和实心点为观测拟合位置').replace('阶段理想 X','阶段目标 X')
        template = template.replace('▶ 逆向播放','▶ 正向播放').replace("value<=0", "value>=4").replace("value=4;playing", "value=0;playing").replace("value<=0)$('stage').value=4", "value>=4)$('stage').value=0")
        template = template.replace("Math.max(0,+$('stage').value-(now-lastTime)/1800)","Math.min(4,+$('stage').value+(now-lastTime)/1800)")
        template = template.replace("let mat=poses[+$('part').value];", "let b=+$('part').value,f=DATA.bodyFrames['ABC'[b]].final_pose,mat=poses[b].map((r,i)=>f[0].map((_,j)=>r.reduce((s,v,k)=>s+v*f[k][j],0)));")
        template = template.replace("[...p,...(exact?", "[...(exact&&DATA.actual?DATA.actual[sample][Math.round(stage)][i]:[null,null,null]),...(exact?")
        template = template.replace("...DATA.checkpoints[i]].map", "...(exact&&stage>0?DATA.plans[sample][Math.round(stage)][i]:[null,null,null])].map")
        template = template.replace("阶段目标 X", "阶段实际 X").replace("<th>观测 X</th>","<th>阶段观测 X</th>").replace("<th>最终目标 X</th>","<th>阶段目标 X</th>")
        template = template.replace('初始 · 调姿前固定参考位置','初始 · 随机摆放，无理想目标').replace('后对合目标 · 理想已到位，实际偏差待校正','后对合 · 观测拟合，等待再调姿').replace('再调姿后 · 最终完整，完美装配目标','再调姿后 · 观测拟合结果').replace('调姿目标 · 已对齐最终基准，仅保留间隙','首次调姿后 · 观测拟合结果')
        if actual is not None:
            template = template.replace("观测拟合", "阶段实际")
    Path(path).write_text(template.replace("__CAD_PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
