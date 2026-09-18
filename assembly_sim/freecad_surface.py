"""FreeCAD-side exact ray intersection / projection; no GUI required."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.executable).resolve().parent.parent/'lib'))
import FreeCAD as App
import Part


def query(data):
    shape=Part.read(data['shape'])
    if 'start' in data:
        start,end=App.Vector(*data['start']),App.Vector(*data['end'])
        vertices=shape.section(Part.makeLine(start,end)).Vertexes
        if not vertices:return None
        p=min((v.Point for v in vertices),key=lambda v:(v-start).Length)
    else:
        vertex=Part.Vertex(App.Vector(*data['point']))
        distance,pairs,_=min((face.distToShape(vertex) for face in shape.Faces),key=lambda result:result[0])
        p=pairs[0][0]
    return [p.x,p.y,p.z]

if __name__=='__main__':
    request=Path(sys.argv[1]);data=json.loads(request.read_text())
    Path(data['output']).write_text(json.dumps(query(data)))
