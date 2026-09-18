"""Exact CAD queries and continuous candidate sampling for the editor."""
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from .cad import freecad_command


def surface_point(shape_file, *, start=None, end=None, point=None):
    with tempfile.TemporaryDirectory(prefix='cad-surface-') as tmp:
        root=Path(tmp);output=root/'point.json';request=root/'request.json'
        data=dict(shape=str(Path(shape_file).resolve()),output=str(output))
        if point is None:data.update(start=np.asarray(start).tolist(),end=np.asarray(end).tolist())
        else:data['point']=np.asarray(point).tolist()
        request.write_text(json.dumps(data))
        result=subprocess.run([*freecad_command(),str(Path(__file__).with_name('freecad_surface.py')),str(request)],capture_output=True,text=True,timeout=30)
        if result.returncode or not output.exists():
            raise ValueError('CAD 表面选点失败：'+(result.stderr or result.stdout)[-300:])
        value=json.loads(output.read_text())
        return None if value is None else np.asarray(value,float)


def random_mesh_candidate(vertices, triangles, rng=None):
    """Area-weighted mesh face + uniform triangle interior, then CAD projection."""
    rng=np.random.default_rng() if rng is None else rng
    faces=np.asarray(vertices)[triangles]
    area=np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)
    face=faces[rng.choice(len(faces),p=area/area.sum())]
    u,v=rng.random(2);u=np.sqrt(u)
    return (1-u)*face[0]+u*(1-v)*face[1]+u*v*face[2]
