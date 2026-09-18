# CAD 建模、坐标读取与前端

默认 `cad_forward`：FreeCAD 实体 + 固定阶段目标 + 仅正向仿真。计算细节见 [流程说明](simulation_workflow.md)。

## 生成与查看

```bash
.venv/bin/python run_standard_pipeline.py --config configs/cad_forward.json
.venv/bin/python industrial_desktop_app.py --dataset-dir data/cad_general_demo
```

重复运行需更换 `--dataset-name`，不会覆盖既有数据。生成入口只保留当前 CAD 正向方式，数量只配置总流程数和测试占比。

CAD 参数对话框设置几何尺寸、前后固定间隙、初始随机范围和工艺旋转标准差。主界面设置工艺平移、观测标准差。`cad_motion.initial_offset_mm` 是旧版兼容参数，新版不使用。

内嵌三维页提供五阶段正向播放、样本选择、世界坐标轴和随部件移动的 A/B/C 局部轴。“阶段实际”读取仿真真实状态，“观测拟合”从测量点估计刚体位姿，“阶段目标”查看固定阶段设计位置。表格按阶段实际、阶段观测、阶段目标排列。阶段 0 显示随机初态，目标列为空；没有实际数据文件时实际列也为空，不拿估计值代替真值。右侧 R、t 为所选局部坐标系到 W 的绝对位姿，不是本阶段运动增量。阶段间插值没有测量记录。

离线 `cad_preview.html` 展示前 20 条测试样本的实际运动，表格比较阶段实际、阶段观测、阶段目标。它不需要联网或额外安装浏览器插件。

## CAD 产物

- `cad/aircraft.FCStd`：三段实体、配置的表面 checkpoint（默认共 9 个）、W 与 A/B/C 坐标轴。部件 Shape 使用局部坐标，Placement 设置最终位姿；局部轴和测点的 Placement 关联所属部件，可随部件移动。
- `cad/aircraft.step`：最终装配位置的三个实体，不含坐标轴和测点辅助对象。
- `cad/A.step`、`B.step`、`C.step` 和 STL：最终装配世界位置几何，供旧前端兼容。
- `cad/A_local.step`、`B_local.step`、`C_local.step`：各自局部坐标下的几何。
- `cad/geometry.json`：几何参数、表面验证、局部 checkpoint、装配世界 checkpoint、局部坐标系最终位姿，以及装配世界坐标下的三角网格。原点定义见流程说明。

机尾保留直径 180 mm、长度 190 mm 的双出气筒及两片水平尾翼。所有尺寸均可配置；FreeCAD 构建时检查三个实体有效、每段为单个实体、全部 checkpoint 位于对应表面。

## 数据契约

P 是三段 checkpoint 总数；metadata.json 与 target_states.json 的 point_names 指定数组顺序。每段至少 3 点，可以不同数量；不能再按固定每 3 点切片。

| 文件 | 内容与形状 |
|---|---|
| `target_states.json / .npz` | 固定 `local_checkpoints[P,3]`、`target_poses[4,3,4,4]`、`target_checkpoints[4,P,3]`；阶段 1–4，仅一份，不随样本变化 |
| `observations.npz` | train/test，各 `[N,5,P,3]`，阶段 0–4 |
| `observations.csv` 与训练/测试宽表 | 同一观测数据，便于其他程序读取 |
| `checkpoints.csv` | 局部坐标、观测世界坐标、阶段目标、最终完美目标；阶段 0 目标字段为空 |
| `process_model.json` | 坐标约定、公开目标、控制规则、误差含义及参数 |
| `evaluation/true_trajectories.npz` | 各 split 的真实 checkpoint `[N,5,P,3]` |
| `evaluation/state_actions.npz` | 各 split 的 `true_poses`、`estimated_poses` `[N,5,3,4,4]`；`commanded`、`executed`、`process_errors` `[N,4,3,4,4]`；`measurement_errors` `[N,5,P,3]` |
| `labels.json` | 两个集合全部流程的故障标签、source_id、阶段、执行部件和注入偏移 |
| `split_indices.npz` | 互不重叠的 train/test 原始流程编号 |
| `evaluation/generation_config.json` | 完整参数与随机种子 |

新版不输出逆向矩阵，也不将随机初态伪装成理想目标。内部 `load_plans` 为旧诊断接口保留五阶段数组适配，其中阶段 0 占位不参与诊断，不应作为目标导出。

```python
import json
import numpy as np
from pathlib import Path

root = Path('data/cad_general_demo')
names = json.loads((root / 'metadata.json').read_text())['point_names']
head_indices = [i for i, name in enumerate(names) if name.startswith('A')]
with np.load(root / 'target_states.npz') as z:
    q = z['local_checkpoints'][head_indices]  # 所有机头点，A 局部坐标
    T = z['target_poses'][0, 0]          # 首次调姿，A 的局部→世界位姿
    p_target = q @ T[:3, :3].T + T[:3, 3]
with np.load(root / 'evaluation/state_actions.npz') as z:
    H_cmd = z['test_commanded'][0, 0, 0]  # 第 0 条测试样本，首次调姿，A
    H_real = z['test_executed'][0, 0, 0]
    E = z['test_process_errors'][0, 0, 0]
    assert np.allclose(H_real, E @ H_cmd)
```

文件单位为 mm，旋转矩阵无量纲；参数中的角度单位为 deg。列向量形式 `p=Rq+t`，Python 批量行点形式 `q @ R.T + t`。

`TrialConfig.cad_checkpoints` 可配置各段局部点，例如 `{"A":[{"name":"A1","position":[150,272.956039,99.347874]}, ...]}`；实际使用应保存足够精度的坐标，建议从图形编辑器选点。省略某段时使用该段默认 3 点；空列表会因点数不足被拒绝。编号可不连续，不能重复，也不能跨部件使用。配置随设置预设和 `evaluation/generation_config.json` 保存，FreeCAD 产物与数据文件使用同一列表。


## 连续选点与可编辑最终位姿

模型点击使用视线和 FreeCAD BRep 的交点，取最靠近视点的一处，支持端面内部和曲面。随机按钮只给出一个候选点：先按网格三角面积选面并在面内连续采样，再投影到 FreeCAD 精确表面。点击和随机均先在模型上预览，只有新增或修改才写入 checkpoint 列表。

最终 B 的平移和三个旋转角保存在 `cad_motion.final_*`；三段最终位姿、各阶段间隙方向以及 FCStd/STEP/STL 均按此变换。世界 W 不动。各部件的 `_local.step` 始终表示局部几何，可直接用于表面选点；装配文件表示设置后的最终世界位姿。
