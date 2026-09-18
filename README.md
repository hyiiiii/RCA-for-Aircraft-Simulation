# 飞机装配 CAD 通用数据生成

只保留三段 CAD 刚体的正向仿真。输入总流程数和测试集占比，生成普通 train/test 两个集合，不创建健康校准集，也不自动训练或运行溯源算法。

```bash
.venv/bin/python generate_data.py --n-total 200 --test-fraction 0.2 --dataset-name my_dataset
.venv/bin/python run_standard_pipeline.py --config configs/cad_forward.json
.venv/bin/python industrial_desktop_app.py --dataset-dir data/cad_general_demo
```

`generate_data.py` 仅生成数据；`run_standard_pipeline.py` 额外导出已知装配依赖图和汇总，仍不运行诊断。桌面“生成数据”使用后一入口。旧生成方式选择和反向回放按钮已移除。

## 生成过程

初始随机摆放 → 首次调姿 → 前对合 → 后对合 → 整机再调姿。

每段有局部坐标系和可编辑的 CAD 表面 checkpoint（默认每段 3 个，各段点数可不同）。首次调姿对准固定设计基准并保留接口间隙。前对合使用 A/B 的观测移动 A；后对合使用 B/C 的观测移动 C；最后全部测点共同计算一次整机运动。执行时添加整段刚体工艺误差，再添加每点观测噪声。

正常和故障轨迹使用相同的生成规则，完整轨迹随机分到训练集或测试集。一条轨迹不会跨集合拆分。故障比例作用于全部流程，不再仅作用于测试集。训练集不是自动筛出的健康样本；具体算法怎样训练、校准、验证应在算法侧独立处理。

## 参数与数据

- `n_total`：总流程数；`test_fraction`：测试占比。测试数为 `floor(n_total*test_fraction+0.5)`，剩余为训练，要求两者非空。
- `fault_fraction`：全体中注入故障的比例，故障总数为 `round(n_total*fault_fraction)`。随机划分后各集合比例可有波动。
- `fault_stage`、`fault_part`：严格限定故障位置。阶段 1 可 A/B/C，2 仅 A，3 仅 C，4 仅 ABC。不相容组合报错，随机只在合法组合中选择。
- `fault_magnitude`：固定的额外平移偏差模长，方向随机；不是随机幅度。
- `process_sigma`、`measurement_sigma`、`forward_settings`：工艺平移、观测噪声、初态范围和工艺旋转误差。

`observations.npz` 包含 train/test，各 `[N,5,9,3]`；`train_observed.csv`、`test_observed.csv` 是宽表。`labels.json` 给出两个集合全部轨迹的故障标签和 source_id。`split_indices.npz` 保存原始流程编号，可检查分组互斥并做配对实验。

固定目标在 `target_states.json/.npz`；`evaluation/` 保存真实状态、估计位姿、指令矩阵、实际执行矩阵、工艺/观测误差与完整配置。`checkpoints.csv` 包含局部坐标、观测、阶段目标、最终目标。阶段 0 没有目标。

三维页可显示阶段实际、观测拟合或阶段目标；表格始终比较实际/观测/目标。实际是真值，观测拟合才是估计，目标是预定义的位置。前端读取真值仅用于查看，控制器只用观测。默认因果图只有已知结构，不会自动生成候选根因红线。

详见 [数学与物理流程](docs/simulation_workflow.md)、[坐标和文件读取](docs/cad_generation.md)、[参数与按钮说明](docs/ui_guide.md)。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python industrial_desktop_app.py --smoke-test --dataset-dir data/cad_general_demo
```

需要 Python 3.10+、NumPy、FreeCAD；桌面需要 Tkinter。当前为阶段端点的运动学仿真，不求解接触、变形和碰撞约束，阶段间播放插值不代表连续采样。

## 独立因果溯源

桌面“根因溯源”页：载入数据集 → 设置终检限度 → 开始溯源。生成数据与溯源是两个入口。溯源完成后可以选择测试轨迹，查看排名与因果图红线。

```bash
.venv/bin/python trace_root_cause.py --dataset-dir data/cad_general_demo --quality-limit 1
# 单条轨迹：
.venv/bin/python trace_root_cause.py --dataset-dir data/cad_general_demo --sample 12 --quality-limit 1
# 无训练标签或正常训练样本不足时，载入同模型的已保存参考：
.venv/bin/python trace_root_cause.py --dataset-dir data/cad_general_demo --reference /absolute/path/normal_reference.json
```

当前溯源方法需要健康参考。它使用训练集的正常/异常标签选择正常训练流程，在算法内部按固定规则分为估计误差分布和确定阈值两部分；不向原始数据写入 calibration，也不更改 train/test 划分。至少需要四条正常训练流程，样本很少时阈值估计会不稳定。测试标签不参与预测，只在事后计算评价指标。真实坐标和实际执行矩阵不参与诊断。

每次桌面溯源结果写入 `graph/<数据集名>/diagnosis_<时间>/`，包含健康参考、逐条结果、排名和图。生成文件保持原样。 [当前 CAD 的前对合数值计算](docs/front_mating_example.md)。

## 编辑 CAD 尺寸、测点与阶段目标

点击左侧「CAD 尺寸、测点与目标…」，按机头 A、机身 B、机尾 C 分页编辑。各页有尺寸说明、真实 FreeCAD 网格预览和局部 XYZ 坐标轴；可新增、修改、删除 checkpoint，或点击「随机一个候选点」。支持各段不同点数，每段需至少 3 个不共线点，保存前自动校验 CAD 表面。修改尺寸后先刷新预览，再检查已有测点是否仍适用。

「阶段目标与初始工况」页解释前后间隙、初始随机范围与工艺旋转误差，同时可修改最终机身 B 的平移和旋转角（R=Rz·Ry·Rx），实时展示各阶段目标模型、固定 W、局部坐标轴与 R、t。单击模型按视线精确求交，支持端面内部和曲面；点击与随机仅显示一个粉色候选点，新增/修改/取消由用户决定。保存后测点固定用于所有样本；改变点数会同步改变导出的数据、配准和因果图。

溯源操作与横向因果图在同一页，健康参考不再提供输入控件，由算法内部自动建立。详细步骤见 [界面说明](docs/ui_guide.md)。

独立「坐标」页已移除，阶段实际/观测/目标坐标仍可在「三维装配」页的表格和导出文件中读取。

根因图红线使用算法返回的传播路径，绿色边框单独标出该流程的真实根因。新增 `ground_truth.json` 供展示与事后对照，预测文件与标签分开保存。三维目标点采用置顶的橙色菱形；初始阶段可显示首次调姿目标作参照。所有显示开关使用勾选标记。
