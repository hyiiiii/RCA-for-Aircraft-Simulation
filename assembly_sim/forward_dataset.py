"""Storage contract for forward-only CAD trajectories."""
from dataclasses import asdict
import csv
import numpy as np
from .cad import CADGeometry, CADMotion, export_cad, freecad_command
from .model import ModelConfig
from .forward import ForwardSettings, target_definition, body_frames, simulate_forward, action_id, fault_choices


def create_forward_dataset(config, progress=None):
    from pathlib import Path
    from pipeline_core import write_json, _save_observations, _progress
    freecad_command()
    directory = Path(config.data_root).expanduser().resolve()/config.dataset_name
    directory.mkdir(parents=True,exist_ok=False)
    evaluation = directory/'evaluation'; evaluation.mkdir()
    geometry = CADGeometry(**config.cad_geometry); motion = CADMotion(**config.cad_motion)
    settings = ForwardSettings(**config.forward_settings)
    model = ModelConfig(config.process_sigma,config.measurement_sigma)
    _progress(progress,'FreeCAD 建模并建立 W、A、B、C 坐标系；固定各阶段目标')
    manifest = export_cad(directory/'cad',geometry,config.cad_checkpoints,motion)
    names = tuple(manifest['point_names'])
    local, targets, points = target_definition(geometry,motion,manifest['checkpoints'],names)
    spec = dict(version='5.0',controller_policy='body_relative',generation_mode='cad_forward',planning_mode='fixed_pose_targets',units='mm',
        point_names=list(names),part_order=list('ABC'),target_stages=[1,2,3,4],
        local_checkpoints=local.tolist(),target_poses=targets.tolist(),target_checkpoints=points.tolist(),
        world_frame=manifest['world_frame'],body_frames=manifest['body_frames'],
        convention='Column vectors: p_W=R_Wb q_b+t_Wb. H_real=E_W H_cmd; T_next=H_real T_previous.',
        initial_policy='Independent bounded random R,t around stage-1 poses; stage 0 has no target.',
        control_policy='Stage 1 fits A/B/C to fixed world targets. Stage 2: move A to observed B using fixed CAD relative pose D_BA. Stage 3: move C to observed B using D_BC. Stage 4: one shared fit of all configured points to the final world target.',
        target_policy='Published stage targets are fixed design/quality references. Mating command destinations follow the estimated B pose; fixed references are never overwritten by observations.',
        ideal_final_action=np.eye(4).tolist(),
        error_policy='Rigid-body SE(3) process error acts on the left in W (rotation about W origin); observation noise is added after true checkpoints are derived. Inactive bodies do not move.',
        limitations='Kinematic stage endpoints, no contact/collision/deformation model. Final shared motion cannot remove internal relative assembly errors.',
        audit_file='evaluation/state_actions.npz',
        audit_shapes='split_true_poses, split_estimated_poses: [sample,5,3,4,4]; split_commanded/executed/process_errors: [sample,4,3,4,4]',
        process_translation_sigma_mm=config.process_sigma,measurement_sigma_mm=config.measurement_sigma,
        forward_settings=asdict(settings))
    write_json(directory/'process_model.json',spec)
    # Single public target definition, independent of seed and sample count.
    write_json(directory/'target_states.json',{k:spec[k] for k in ('point_names','target_stages','part_order','local_checkpoints','target_poses','target_checkpoints','convention')})
    np.savez_compressed(directory/'target_states.npz',local_checkpoints=local,target_poses=targets,target_checkpoints=points)
    sample_seed,fault_seed,partition_seed=np.random.SeedSequence(config.seed).spawn(3)
    sample_seeds=sample_seed.spawn(config.n_total)
    fault_rng=np.random.default_rng(fault_seed)
    partition_rng=np.random.default_rng(partition_seed)
    permutation=partition_rng.permutation(config.n_total)
    indices=dict(train=permutation[config.n_test:],test=permutation[:config.n_test])
    attacked=set(fault_rng.choice(config.n_total,int(round(config.n_total*config.fault_fraction)),replace=False).tolist())
    allowed=fault_choices(config.fault_stage,config.fault_part)
    faults={}
    for i in sorted(attacked):
        stage=int(fault_rng.choice(list(allowed)))
        part=str(fault_rng.choice(allowed[stage]))
        direction=fault_rng.normal(size=3); direction/=np.linalg.norm(direction)
        faults[i]=dict(stage=stage,part=part,offset=(config.fault_magnitude*direction).tolist())
    observed,truth,audit,labels={},{},{},[]
    for split,source_ids in indices.items():
        _progress(progress,f'正向生成 {split}：{len(source_ids)} 条流程；训练和测试使用相同的工况/误差/故障规则')
        runs=[]
        for i,source_id in enumerate(source_ids):
            fault=faults.get(int(source_id))
            seed=int(sample_seeds[source_id].generate_state(1,dtype=np.uint64)[0])
            runs.append(simulate_forward(local,targets,points,model,settings,seed,fault,point_names=names))
            labels.append(dict(sample_id=f'{split}_{i:05d}',source_id=f'run_{source_id:05d}',split=split,
                injected=fault is not None,stage=fault['stage'] if fault else None,
                part=fault['part'] if fault else None,root=action_id(fault['stage'],fault['part']) if fault else None,
                offset=fault['offset'] if fault else None))
        observed[split]=np.stack([r.observed for r in runs]); truth[split]=np.stack([r.true for r in runs])
        for name,attr in [('true_poses','poses'),('estimated_poses','estimated_poses'),('commanded','commanded'),('executed','executed'),('process_errors','process_errors'),('measurement_errors','measurement_errors')]:
            audit[f'{split}_{name}']=np.stack([getattr(r,attr) for r in runs])
    np.savez_compressed(directory/'split_indices.npz',**indices)
    write_json(directory/'labels.json',labels)
    _save_observations(directory,observed,names)
    np.savez_compressed(evaluation/'true_trajectories.npz',**truth)
    np.savez_compressed(evaluation/'state_actions.npz',**audit)
    write_json(evaluation/'labels.json',labels); write_json(evaluation/'generation_config.json',asdict(config))
    with (directory/'checkpoints.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['sample_id','split','stage','point','local_x','local_y','local_z','observed_x','observed_y','observed_z','target_x','target_y','target_z','final_target_x','final_target_y','final_target_z'])
        for split,rows in observed.items():
            for i,row in enumerate(rows):
                for k in range(5):
                    for p,name in enumerate(names):
                        w.writerow([f'{split}_{i:05d}',split,k,name,*local[p],*row[k,p],*(points[k-1,p] if k else ['','','']),*points[3,p]])
    write_json(directory/'metadata.json',dict(schema_version=5,controller_policy='body_relative',generation_mode='cad_forward',planning_mode='fixed_pose_targets',
        units='mm',shape='sample,stage,point,xyz',splits={k:len(v) for k,v in observed.items()},point_names=list(names),stages=list(range(5)),
        process_sigma=config.process_sigma,measurement_sigma=config.measurement_sigma,n_total=config.n_total,test_fraction=config.test_fraction,
        split_policy="Random split of complete trajectories, independent of simulation RNG; no calibration subset",
        fault_policy="Exactly round(n_total*fault_fraction) trajectories across the whole dataset; each split fraction can fluctuate",
        purpose='General-purpose train/test dataset with labels. No training, calibration or diagnosis is performed. evaluation/ contains simulator truth and actions.'))
    # Viewer may show simulation truth; controller and diagnosis never consume it.
    from .cad_view import CADViewData
    from .cad_preview import write_preview
    view=CADViewData(directory)
    write_preview(directory/'cad_preview.html',manifest,view.plans['test'],view.actual_poses['test'],observed['test'],forward=True,actual=truth['test'])
    return directory
