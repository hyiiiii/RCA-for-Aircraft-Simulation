"""Generate dependency edges from the fixed process, never from data discovery.

Coordinates denote physical positions. Independent exogenous process errors,
measurement-noise inputs and fixed design targets are omitted from the drawing.
Observation Y is a noisy measurement of each position and is used by the controller.
The adjacency convention matches the reference project: adj[child, parent].
"""
from __future__ import annotations

import csv
import json
from collections import deque
from html import escape
from pathlib import Path

from .model import POINT_NAMES


def build_graph(forward=False, controller_policy="body_relative", point_names=POINT_NAMES) -> dict:
    from .checkpoints import point_groups
    point_groups(point_names)
    nodes, edges = [], []
    for stage in range(5):
        for point in point_names:
            nodes.append(dict(id=f"s{stage}_{point}", kind="coordinate", stage=stage,
                              point=point, part=point[0], dimension=3,
                              value_type="physical_position_mm", label=f"{point} / 阶段 {stage}"))
    actions = [
        ("h1_A", 1, "首次调姿：移动 A", "A", "A"),
        ("h1_B", 1, "首次调姿：移动 B", "B", "B"),
        ("h1_C", 1, "首次调姿：移动 C", "C", "C"),
        ("h2_front", 2, "前对合：移动 A", "A" if forward and controller_policy == "fixed_world" else "AB", "A"),
        ("h3_rear", 3, "后对合：移动 C", "C" if forward and controller_policy == "fixed_world" else "BC", "C"),
        ("h4_final", 4, "再调姿：移动整机", "ABC", "ABC"),
    ]
    for name, stage, label, inputs, moved in actions:
        nodes.append(dict(id=name, kind="action", stage=stage, label=label,
                          dimension=6, value_type="SE(3)", input_parts=inputs, moved_parts=moved))
        for point in point_names:
            if point[0] in inputs:
                edges.append(dict(source=f"s{stage-1}_{point}", target=name, kind="localization"))
            if point[0] in moved:
                edges.append(dict(source=name, target=f"s{stage}_{point}", kind="motion"))
    for stage in range(1, 5):
        for point in point_names:
            edges.append(dict(source=f"s{stage-1}_{point}", target=f"s{stage}_{point}", kind="state"))
    graph = dict(nodes=nodes, edges=edges, adjacency_convention="adj[child,parent]",
                 source="specified_structural_equations", learned=False,
                 omitted="independent exogenous errors, measurement nodes, fixed design targets",
                 coordinate_semantics="true physical positions; diagnostics use noisy observations",
                 action_semantics="compute transform from preceding localization measurements and execute it")
    validate_graph(graph)
    return graph


def validate_graph(graph: dict) -> None:
    ids = [n["id"] for n in graph["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate graph node")
    degree = dict.fromkeys(ids, 0)
    outgoing = {n: [] for n in ids}
    undirected = {n: set() for n in ids}
    for e in graph["edges"]:
        a, b = e["source"], e["target"]
        if a not in degree or b not in degree:
            raise ValueError("Graph edge refers to unknown node")
        degree[b] += 1
        outgoing[a].append(b)
        undirected[a].add(b)
        undirected[b].add(a)
    queue = deque(n for n in ids if degree[n] == 0)
    seen = []
    while queue:
        n = queue.popleft()
        seen.append(n)
        for child in outgoing[n]:
            degree[child] -= 1
            if degree[child] == 0:
                queue.append(child)
    if len(seen) != len(ids):
        raise ValueError("The process graph contains a directed cycle")
    connected, pending = set(), [ids[0]]
    while pending:
        n = pending.pop()
        if n not in connected:
            connected.add(n)
            pending.extend(undirected[n] - connected)
    if len(connected) != len(ids):
        raise ValueError("The process graph is not weakly connected")


def ancestors(graph: dict, node: str, include_self: bool = False) -> set[str]:
    parents = {n["id"]: [] for n in graph["nodes"]}
    if node not in parents:
        raise ValueError(f"Unknown observed node: {node}")
    for e in graph["edges"]:
        parents[e["target"]].append(e["source"])
    found, stack = set(), list(parents[node])
    while stack:
        parent = stack.pop()
        if parent not in found:
            found.add(parent)
            stack.extend(parents[parent])
    if include_self:
        found.add(node)
    return found


def directed_path(graph: dict, source: str | None, target: str | None) -> list[str]:
    if not source or not target:
        return []
    queue = deque([[source]])
    seen = {source}
    outgoing = {n["id"]: [] for n in graph["nodes"]}
    for e in graph["edges"]:
        outgoing[e["source"]].append(e["target"])
    while queue:
        path = queue.popleft()
        if path[-1] == target:
            return path
        for child in outgoing.get(path[-1], []):
            if child not in seen:
                seen.add(child)
                queue.append(path + [child])
    return []


def inference_path_edges(graph, path):
    """Use exactly the path returned by diagnosis, not all root-to-entry routes."""
    if not path:return set()
    if not isinstance(path,(list,tuple)) or any(not isinstance(n,str) for n in path):
        raise ValueError('Invalid inferred path')
    edges=set(zip(path,path[1:]))
    existing={(e['source'],e['target']) for e in graph['edges']}
    if not edges<=existing:raise ValueError('Inferred path contains an unknown edge')
    return edges


def save_graph(directory: Path, graph: dict | None = None,
               highlight_source: str | None = None, target: str | None = None, *, true_root=None, inferred_path=None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    graph = graph or build_graph()
    (directory / "fixed_graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = [n["id"] for n in graph["nodes"]]
    edge_set = {(e["source"], e["target"]) for e in graph["edges"]}
    with (directory / "adjacency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["child\\parent"] + ids)
        writer.writerows([[child] + [int((parent, child) in edge_set) for parent in ids] for child in ids])
    with (directory / "edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "kind"])
        writer.writeheader()
        writer.writerows(graph["edges"])
    render_svg(directory / "causal_graph.svg", graph, highlight_source, target, true_root=true_root, inferred_path=inferred_path)


def graph_layout(graph, include_initial=True):
    """Time runs left to right; bodies have separate bands of variable height."""
    first = 0 if include_initial else 1
    names = [n['point'] for n in graph['nodes'] if n['kind'] == 'coordinate' and n['stage'] == first]
    positions, bands = {}, {}
    y = 75
    for part in 'ABC':
        points = [p for p in names if p[0] == part]
        height = max(110, len(points)*32+40)
        bands[part] = (y, height)
        for stage in range(first, 5):
            x = 125+(stage-first)*290
            for i, point in enumerate(points):
                positions[f's{stage}_{point}'] = (x, y+32+i*32)
        y += height+20
    for node in graph['nodes']:
        if node['kind'] != 'action' or node['stage'] <= first:
            continue
        moved = node['moved_parts']
        centers = [bands[p][0]+bands[p][1]/2 for p in moved]
        positions[node['id']] = (125+(node['stage']-first)*290-145, sum(centers)/len(centers))
    return positions, bands, 250+(4-first)*290, y+55


def render_svg(path: Path, graph: dict, source=None, target=None, *, true_root=None, inferred_path=None) -> None:
    positions, bands, width, height = graph_layout(graph)
    chosen = directed_path(graph, source, target) if inferred_path is None else inferred_path
    highlighted = inference_path_edges(graph, chosen)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<title>装配因果图</title><desc>红色为算法推断的传播路径；绿色标记真实根因。</desc>',
              '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>',
              '<style>text{font-family:Arial,"Hiragino Sans GB",sans-serif;fill:#294c65;font-size:12px}</style>',
              f'<rect width="{width}" height="{height}" fill="white"/>']
    for stage, label in enumerate(['初始', '首次调姿后', '前对合后', '后对合后', '再调姿后']):
        chunks.append(f'<text x="{125+stage*290}" y="34" text-anchor="middle">{stage} · {label}</text>')
    for part, (y, h) in bands.items():
        chunks.append(f'<rect x="25" y="{y}" width="{width-50}" height="{h}" rx="8" fill="#f4f8fb"/><text x="35" y="{y+18}">{part} {dict(A="机头",B="机身",C="机尾")[part]}</text>')
    nodes = {n['id']:n for n in graph['nodes']}
    for edge in sorted(graph['edges'], key=lambda e:(e['source'],e['target']) in highlighted):
        a,b = edge['source'],edge['target']
        x,y = positions[a]; xx,yy = positions[b]
        x += 54 if nodes[a]['kind']=='action' else 8
        xx -= 54 if nodes[b]['kind']=='action' else 8
        red = (a,b) in highlighted
        edge_class = "inferred-path" if red else "structural-edge"
        color = '#cf4238' if red else '#aebfca' if edge['kind']=='state' else '#608cb1'
        chunks.append(f'<path class="{edge_class}" data-source="{escape(a)}" data-target="{escape(b)}" d="M{x} {y} C{(x+xx)/2} {y} {(x+xx)/2} {yy} {xx} {yy}" fill="none" stroke="{color}" stroke-width="{3 if red else 1}" marker-end="url(#arrow)"/>')
    for name,(x,y) in positions.items():
        node=nodes[name]; color='#cf4238' if name in chosen else '#294c65'
        if node['kind']=='action':
            label={1:'调姿 '+node['moved_parts'],2:'前对合 A',3:'后对合 C',4:'再调姿 ABC'}[node['stage']]
            chunks.append(f'<rect x="{x-54}" y="{y-20}" width="108" height="40" rx="8" fill="#e7f2fa" stroke="{color}"/><text x="{x}" y="{y-3}" text-anchor="middle">{label}</text><text x="{x}" y="{y+13}" text-anchor="middle">R, t</text>')
        else:
            chunks.append(f'<circle cx="{x}" cy="{y}" r="8" fill="white" stroke="{color}"/><text x="{x+12}" y="{y+4}">{escape(node["point"])}</text>')
    if true_root in positions:
        x,y=positions[true_root];dx,dy=(59,25) if nodes[true_root]['kind']=='action' else (13,13)
        chunks.append(f'<g class="true-root"><rect x="{x-dx}" y="{y-dy}" width="{2*dx}" height="{2*dy}" fill="none" stroke="#18865b" stroke-width="3"/><text x="{x}" y="{y-dy-7}" text-anchor="middle" style="fill:#18865b">真实根因</text></g>')
    if source in positions:
        x,y=positions[source]
        chunks.append(f'<text x="{x}" y="{y+36}" text-anchor="middle" style="fill:#cf4238">推断根因</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks), encoding='utf-8')
