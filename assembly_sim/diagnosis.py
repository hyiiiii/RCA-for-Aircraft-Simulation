"""Algorithm-side reference preparation for general labelled train/test data.

Only training-set normal/abnormal labels select reference samples. Test labels
are not inputs to the reference or per-trajectory diagnosis.
"""
import json
from pathlib import Path
import numpy as np
from .tracing import fit_reference
from .model import POINT_NAMES


def prepare_reference(directory, values, plans, metadata, reference_file=None):
    directory=Path(directory)
    names = tuple(metadata.get('point_names', POINT_NAMES))
    mode=metadata.get('generation_mode','feedback')
    policy=metadata.get('controller_policy','fixed_world')
    if reference_file is not None:
        reference=json.loads(Path(reference_file).read_text(encoding='utf-8'))
        if reference.get('generation_mode','feedback')!=mode or reference.get('controller_policy','fixed_world')!=policy:
            raise ValueError('健康参考与当前数据的控制方式不一致')
        nominal=np.asarray(reference.get('nominal_coordinates'),float)
        if plans and (nominal.shape!=(5,len(names),3) or not np.allclose(nominal[1:],plans['test'][0,1:],atol=1e-8,rtol=0)):
            raise ValueError('健康参考与当前 CAD 阶段目标不一致')
        if tuple(reference.get("point_names", POINT_NAMES)) != names:
            raise ValueError("健康参考的测点编号与当前数据不一致")
        return reference
    if 'calibration' in values:
        return fit_reference(values['train'],values['calibration'],
            train_planned=plans.get('train'),calibration_planned=plans.get('calibration'),
            generation_mode=mode,controller_policy=policy,point_names=names)
    path=directory/'labels.json'
    if not path.exists():
        raise ValueError('缺少训练集正常/异常标签 labels.json，算法无法自动建立正常参考；不会用测试数据建立参考。')
    labels=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(labels,list):
        raise ValueError('labels.json 必须是标签列表')
    expected={f'train_{i:05d}':i for i in range(len(values['train']))}
    selected={}
    for row in labels:
        if not isinstance(row,dict):
            raise ValueError('Invalid label row')
        sample=row.get('sample_id','')
        if not isinstance(sample,str) or not sample.startswith('train_'):
            continue  # No test-label fields are used for reference preparation.
        if sample not in expected or sample in selected or row.get('split')!='train' or type(row.get('injected')) is not bool:
            raise ValueError('训练集标签缺失、重复或格式无效')
        selected[sample]=row['injected']
    if set(selected)!=set(expected):
        raise ValueError('训练集标签未完整对应训练样本')
    indices=np.array([i for name,i in expected.items() if not selected[name]],dtype=int)
    if len(indices)<4:
        raise ValueError(f'正常训练样本只有 {len(indices)} 条，当前方法至少需要 4 条。请使用含更多正常训练样本的数据集。')
    # This partition belongs to the algorithm run, not the dataset generator.
    shuffled=np.random.default_rng(0).permutation(indices)
    cut=min(max(2,int(.7*len(shuffled))),len(shuffled)-2)
    fit_ids,threshold_ids=shuffled[:cut],shuffled[cut:]
    p=plans.get('train')
    reference=fit_reference(values['train'][fit_ids],values['train'][threshold_ids],
        train_planned=p[fit_ids] if p is not None else None,
        calibration_planned=p[threshold_ids] if p is not None else None,
        generation_mode=mode,controller_policy=policy,point_names=names)
    reference.update(reference_source=str(directory),
        input_policy='Training observations and public targets; training injected flag selects normal reference samples. No test labels, true coordinates or executed actions are used.',
        reference_selection='Normal training samples; deterministic 70/30 internal fit/threshold split, at least 2 each; dataset unchanged',
        fit_sample_ids=[f'train_{i:05d}' for i in fit_ids],
        threshold_sample_ids=[f'train_{i:05d}' for i in threshold_ids])
    return reference


def evaluation_labels(directory, n_test):
    """Optional post-diagnosis evaluation. Missing labels never block inference."""
    directory=Path(directory)
    path=directory/'labels.json'
    if not path.exists():
        path=directory/'evaluation/labels.json'
    if not path.exists():
        return None
    labels=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(labels,list):
        return None
    lookup={}
    for row in labels:
        if not isinstance(row,dict):
            return None
        name=row.get('sample_id','')
        if not isinstance(name,str) or not name.startswith('test_'):
            continue
        if name in lookup or type(row.get('injected')) is not bool or 'root' not in row:
            return None
        lookup[name]=row
    expected=[f'test_{i:05d}' for i in range(n_test)]
    if set(lookup)!=set(expected):
        return None
    return [lookup[name] for name in expected]


def truth_annotations(directory):
    """Display/evaluation only. Never passed into reference fitting or inference."""
    directory=Path(directory)
    path=directory/'labels.json'
    if not path.exists():path=directory/'evaluation/labels.json'
    try:rows=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}
    if not isinstance(rows,list):return {}
    annotations={};duplicates=set()
    for row in rows:
        if not isinstance(row,dict):continue
        name=row.get('sample_id')
        if not isinstance(name,str) or type(row.get('injected')) is not bool:continue
        if name in annotations:duplicates.add(name)
        root=row.get('root')
        annotations[name]=dict(injected=row['injected'],root=root if isinstance(root,str) else None)
    for name in duplicates:annotations.pop(name,None)
    return annotations
