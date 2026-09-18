"""Stable checkpoint identities and body-local geometry, independent of point count."""
import re
import numpy as np
from .model import POINT_NAMES


def point_groups(point_names=POINT_NAMES):
    names = tuple(point_names)
    if any(not isinstance(n, str) or not re.fullmatch(r'[ABC][1-9][0-9]*', n) for n in names) or len(set(names)) != len(names):
        raise ValueError('测点名称必须唯一，格式为 A1、B1、C1 等正整数编号')
    groups = [np.array([i for i, n in enumerate(names) if n[0] == part], dtype=int) for part in 'ABC']
    if any(len(g) < 3 for g in groups):
        raise ValueError('机头、机身、机尾各需至少 3 个 checkpoint')
    return groups


def validate_local(point_names, local):
    groups = point_groups(point_names)
    local = np.asarray(local, float)
    if local.shape != (len(point_names), 3) or not np.isfinite(local).all():
        raise ValueError('checkpoint 必须是有限的三维局部坐标')
    for part, group in zip('ABC', groups):
        p = local[group]
        if len(np.unique(p, axis=0)) != len(p):
            raise ValueError(f'{part} 段存在重复坐标')
        if np.linalg.matrix_rank(p - p.mean(axis=0), tol=1e-7) < 2:
            raise ValueError(f'{part} 段的 checkpoint 不能全部共线')
    return local


def checkpoint_definition(geometry, config=None):
    from .cad import checkpoint_coordinates
    from .forward import body_frames
    origins = body_frames(geometry)[:, :3, 3]
    defaults = checkpoint_coordinates(geometry).reshape(3, 3, 3) - origins[:, None, :]
    config = {} if config is None else config
    if not isinstance(config, dict) or set(config) - set('ABC'):
        raise ValueError('cad_checkpoints 必须按 A、B、C 分组')
    names, local = [], []
    for b, part in enumerate('ABC'):
        rows = config.get(part, [dict(name=f'{part}{i+1}', position=p.tolist()) for i, p in enumerate(defaults[b])])
        if not isinstance(rows, list):
            raise ValueError(f'{part} checkpoint 必须是列表')
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'name', 'position'} or not isinstance(row['name'], str) or not row['name'].startswith(part):
                raise ValueError(f'{part} checkpoint 格式：name、position（局部 x/y/z）')
            names.append(row['name']); local.append(row['position'])
    local = validate_local(names, local)
    assembled = local.copy()
    for b, group in enumerate(point_groups(names)):
        assembled[group] += origins[b]
    return tuple(names), local, assembled


def as_config(names, local):
    return {part: [dict(name=n, position=np.asarray(p).tolist()) for n, p in zip(names, local) if n[0] == part] for part in 'ABC'}
