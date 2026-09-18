"""Run with FreeCAD's Python / FreeCADCmd (stdlib + FreeCAD only)."""
import json
import math
import os
from pathlib import Path
import sys

# The macOS bundle's Python does not automatically expose its FreeCAD modules.
bundle_lib = Path(sys.executable).resolve().parent.parent / "lib"
sys.path.insert(0, str(bundle_lib))
import FreeCAD as App
import Part
import Mesh


def build(request):
    output = request.parent
    data = json.loads(request.read_text(encoding="utf-8"))
    g = data["geometry"]
    r, nose, body, tail = (g[k] for k in ("radius", "nose_length", "body_length", "tail_length"))
    v = App.Vector
    origin = v(-body/2, 0, 0)
    # Quarter ellipse + two radii, revolved about X: exact half ellipsoid.
    ellipse = Part.Ellipse(origin, nose, r)
    arc = ellipse.toShape(math.pi/2, math.pi)
    wire = Part.Wire([arc, Part.makeLine(v(-body/2-nose, 0, 0), origin),
                      Part.makeLine(origin, v(-body/2, r, 0))])
    head = Part.Face(wire).revolve(origin, v(1, 0, 0), 360)
    middle = Part.makeCylinder(r, body, origin, v(1, 0, 0))
    rear = Part.makeCylinder(r, tail, v(body/2, 0, 0), v(1, 0, 0))
    for y in (-g["exhaust_offset"], g["exhaust_offset"]):
        # Small overlap makes each decoration a robust union with the tail.
        outer = Part.makeCylinder(g["exhaust_radius"], g["exhaust_length"]+1,
                                  v(body/2+tail-1, y, 0), v(1, 0, 0))
        inner = Part.makeCylinder(g["exhaust_radius"]*0.65, g["exhaust_length"],
                                  v(body/2+tail, y, 0), v(1, 0, 0))
        rear = rear.fuse(outer).cut(inner)
    # Two symmetric swept horizontal fins. Roots overlap the cylinder so each
    # fin and both exhausts belong to the same rigid C solid.
    fin_start = body/2+tail-g["fin_root_chord"]
    for sign in (-1,1):
        root_y,tip_y = sign*r*0.85,sign*(r+g["fin_span"])
        z = -g["fin_thickness"]/2
        outline = [v(fin_start,root_y,z),v(fin_start+g["fin_root_chord"],root_y,z),
                   v(fin_start+g["fin_sweep"]+g["fin_tip_chord"],tip_y,z),
                   v(fin_start+g["fin_sweep"],tip_y,z)]
        face = Part.Face(Part.makePolygon(outline+[outline[0]]))
        rear = rear.fuse(face.extrude(v(0,0,g["fin_thickness"])))
    shapes = [head, middle, rear.removeSplitter()]
    def placement(rows):
        matrix=App.Matrix(*[float(x) for row in rows for x in row])
        return App.Placement(matrix)
    transform=placement(data.get('cad_transform',[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    for shape in shapes: shape.Placement=transform.multiply(shape.Placement)
    doc = App.newDocument("AircraftAssembly")
    objects, meshes, diagnostics = [], [], []
    for index, (name, label, shape) in enumerate(zip("ABC", ["A 机头 · 半椭球", "B 机身 · 圆柱", "C 机尾 · 双出气筒与双尾翼"], shapes)):
        if not shape.isValid() or len(shape.Solids) != 1:
            raise ValueError(f"Invalid CAD solid {name}")
        obj = doc.addObject("PartDesign::Feature", name)
        obj.Label = label
        local_shape = shape.copy()
        local_frame=placement(data["body_frames"][name]["final_pose"])
        local_shape.transformShape(local_frame.inverse().toMatrix(), True)
        # Keep the local BRep location inside an identity outer container.
        # Assigning Feature.Placement must not replace the shape's own location.
        local_shape = Part.makeCompound([local_shape])
        obj.Shape = local_shape
        obj.Placement = local_frame
        local_shape.exportStep(str(output / f"{name}_local.step"))
        for key, value in g.items():
            obj.addProperty("App::PropertyLength", key, "Geometry")
            setattr(obj, key, value)
        objects.append(obj)
        shape.exportStep(str(output / f"{name}.step"))
        vertices, triangles = shape.tessellate(2.0)
        mesh = Mesh.Mesh([[vertices[j] for j in face] for face in triangles])
        mesh.write(str(output / f"{name}.stl"))
        meshes.append(dict(part=name, vertices=[[p.x, p.y, p.z] for p in vertices], triangles=triangles))
        distances = []
        for point_name, xyz in ((n, p) for n, p in zip(data["point_names"], data["checkpoints"]) if n[0] == name):
            vertex = Part.Vertex(v(*xyz))
            # Distance to faces, not to the filled solid: verify surface membership.
            distance = min(face.distToShape(vertex)[0] for face in shape.Faces)
            if distance > 1e-6:
                raise ValueError(f"测点 {point_name} 不在所属部件 CAD 表面（距离 {distance:.6g} mm）")
            distances.append(distance)
            point = doc.addObject("PartDesign::Feature", point_name)
            point.Label = point_name
            point.Shape = Part.Vertex(local_frame.inverse().multVec(v(*xyz)))
            point.setExpression("Placement", "<<"+label+">>.Placement")
            point.addProperty("App::PropertyVector", "LocalCoordinate", "Checkpoint")
            point.LocalCoordinate = local_frame.inverse().multVec(v(*xyz))
            point.addProperty("App::PropertyVector", "PerfectTarget", "Checkpoint")
            point.PerfectTarget = v(*xyz)
            point.addProperty("App::PropertyString", "AircraftPart", "Checkpoint")
            point.AircraftPart = name
        frame_group = doc.addObject("App::DocumentObjectGroup", "Frame_"+name)
        frame_group.Label = "刚体坐标系 "+name+"（随部件移动）"
        for axis_name, direction_xyz in zip("XYZ", [(1,0,0),(0,1,0),(0,0,1)]):
            direction = v(*direction_xyz)
            axis_obj = doc.addObject("PartDesign::Feature", axis_name+"_"+name)
            axis_obj.Shape = Part.makeCylinder(3,180,v(0,0,0),direction).fuse(Part.makeCone(9,0,20,direction*180,direction))
            axis_obj.setExpression("Placement", "<<"+label+">>.Placement")
            frame_group.addObject(axis_obj)
        diagnostics.append(dict(part=name, valid=True, solids=len(shape.Solids), volume_mm3=shape.Volume,
                                bounding_box_mm=dict(x_min=shape.BoundBox.XMin,x_max=shape.BoundBox.XMax,
                                                     y_min=shape.BoundBox.YMin,y_max=shape.BoundBox.YMax,
                                                     z_min=shape.BoundBox.ZMin,z_max=shape.BoundBox.ZMax),
                                surface_distances_mm=distances))
    # World aids stay outside A/B/C. Only the three aircraft objects are
    # exported to STEP/STL, so axes can never become part of the training mesh.
    world = data["world_frame"]
    group = doc.addObject("App::DocumentObjectGroup","WorldCoordinates")
    group.Label = "世界坐标系 W（固定，不随飞机移动）"
    for axis in world["axes"]:
        direction,length = v(*axis["direction"]),axis["length"]
        arrow = Part.makeCylinder(4,length-30,v(0,0,0),direction).fuse(
            Part.makeCone(12,0,30,direction*(length-30),direction))
        obj = doc.addObject("PartDesign::Feature",axis["name"])
        obj.Label = axis["name"]+" positive / world fixed"
        obj.Shape = arrow
        group.addObject(obj)
        annotation = doc.addObject("App::Annotation",axis["name"]+"_Label")
        annotation.LabelText = [axis["name"]]
        annotation.Position = direction*(length+30)
        group.addObject(annotation)
    annotation = doc.addObject("App::Annotation","WorldOriginLabel")
    annotation.LabelText = ["O_W (0, 0, 0) mm"]
    annotation.Position = v(0,0,0)
    group.addObject(annotation)
    doc.recompute()
    doc.saveAs(str(output / "aircraft.FCStd"))
    Part.export(objects, str(output / "aircraft.step"))
    manifest = dict(engine="FreeCAD " + ".".join(App.Version()[:3]), units="mm", geometry=g,
                    world_frame=world, body_frames=data["body_frames"], geometry_revision="local-frames-v3",
                    local_checkpoints=data["local_checkpoints"],
                    mesh_frame="assembled_world", local_step_files=["A_local.step","B_local.step","C_local.step"],
                    point_names=data["point_names"], checkpoints=data["checkpoints"],
                    validation=diagnostics, meshes=meshes)
    (output / "geometry.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    App.closeDocument(doc.Name)


if __name__ == "__main__":
    build(Path(os.environ["AIRCRAFT_CAD_REQUEST"]))
