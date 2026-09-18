"""Regressions for continuous CAD picking and a freely placed final assembly."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from assembly_sim.cad import CADGeometry,CADMotion,export_cad,freecad_command
from assembly_sim.surface import surface_point,random_mesh_candidate
from assembly_sim.forward import target_definition,final_body_pose,body_frames,simulate_forward,ForwardSettings
from assembly_sim.model import ModelConfig
from assembly_sim.cad_view import CADViewData
from pipeline_core import TrialConfig,run_pipeline,analyze_dataset


class SurfaceTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.manifest=export_cad(cls.root/'cad',CADGeometry())

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_ray_hits_face_interiors_sides_fins_and_exhausts(self):
        # Local coordinates: none of these is restricted to mesh vertices/rims.
        examples=[('A',[1500,23,44],[-1500,23,44],[300,23,44]),
                  ('B',[-1500,23,44],[1500,23,44],[-600,23,44]),
                  ('B',[1500,23,44],[-1500,23,44],[600,23,44]),
                  ('B',[12,1500,40],[12,-1500,40],[12,np.sqrt(300**2-40**2),40]),
                  ('C',[-1500,23,44],[1500,23,44],[-300,23,44]),
                  ('C',[-200,1500,40],[-200,-1500,40],[-200,np.sqrt(300**2-40**2),40]),
                  ('C',[200,500,1500],[200,500,-1500],[200,500,9]),
                  ('C',[400,150,1500],[400,150,-1500],[400,150,90])]
        for part,start,end,expected in examples:
            with self.subTest(part=part,expected=expected):
                actual=surface_point(self.root/f'cad/{part}_local.step',start=start,end=end)
                np.testing.assert_allclose(actual,expected,atol=1e-6)
        self.assertIsNone(surface_point(self.root/'cad/A_local.step',start=[1500,900,900],end=[-1500,900,900]))

    def test_random_candidate_is_continuous_and_passes_exact_surface_validation(self):
        config={};origins=body_frames(CADGeometry())[:,:3,3];rng=np.random.default_rng(90)
        for b,part in enumerate('ABC'):
            mesh=self.manifest['meshes'][b];vertices=np.asarray(mesh['vertices'])-origins[b]
            points=[]
            for i in range(3):
                candidate=random_mesh_candidate(vertices,np.asarray(mesh['triangles']),rng)
                q=surface_point(self.root/f'cad/{part}_local.step',point=candidate)
                self.assertGreater(np.linalg.norm(vertices-q,axis=1).min(),1e-6)
                points.append(dict(name=f'{part}{i+1}',position=q.tolist()))
            config[part]=points
        validated=export_cad(self.root/'random_surface',CADGeometry(),config)
        self.assertTrue(all(max(p['surface_distances_mm'])<1e-6 for p in validated['validation']))

    def test_rotated_translated_targets_preserve_local_geometry_and_mating(self):
        m=CADMotion(final_x_mm=220,final_y_mm=-150,final_z_mm=350,final_roll_deg=10,final_pitch_deg=-20,final_yaw_deg=35)
        g=final_body_pose(m);local,poses,points=target_definition(motion=m)
        q0,h0,p0=target_definition()
        np.testing.assert_allclose(local,q0,atol=1e-9)
        np.testing.assert_allclose(poses,g@h0,atol=1e-9)
        np.testing.assert_allclose(points,np.einsum('kpi,ji->kpj',p0,g[:3,:3])+g[:3,3],atol=1e-9)
        np.testing.assert_allclose(poses[3,1],g)
        for b in range(3):np.testing.assert_allclose(np.linalg.inv(poses[3,1])@poses[3,b],h0[3,b],atol=1e-9)
        run=simulate_forward(local,poses,points,ModelConfig(0,0),ForwardSettings(process_rotation_deg=0),7)
        np.testing.assert_allclose(run.true[1:],points,atol=1e-8)
        np.testing.assert_allclose(run.commanded[3],np.tile(np.eye(4),(3,1,1)),atol=1e-8)
        from dataclasses import asdict
        result=run_pipeline(TrialConfig(dataset_name='pose',n_total=60,cad_motion=asdict(m),data_root=str(self.root/'data'),graph_root=str(self.root/'graph')))
        d=Path(result['dataset_dir']);view=CADViewData(d)
        for stage in range(1,5):
            _,shown=view.frame('test',0,stage,target=True)
            np.testing.assert_allclose(shown,points[stage-1],atol=1e-8)
        analysis=analyze_dataset(d,self.root/'diagnosis');self.assertEqual(analysis['metrics']['n_test'],12)
        # Opening FCStd or STEP must show the same world pose as the viewer mesh.
        script=self.root/'check_export.py'
        script.write_text('''import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(sys.executable).resolve().parent.parent/'lib'))
import FreeCAD as App,Part
root=Path(sys.argv[1]);doc=App.openDocument(str(root/'aircraft.FCStd'))
manifest=json.loads((root/'geometry.json').read_text())
for name in 'ABC':
    actual=doc.getObject(name).Shape.Solids[0]
    expected=Part.read(str(root/f'{name}.step')).Solids[0]
    assert (actual.CenterOfMass-expected.CenterOfMass).Length<1e-6
    assert abs(actual.Volume/expected.Volume-1)<1e-7
    for i,p in enumerate(manifest['point_names']):
        if p[0]==name:
            q=doc.getObject(p).Shape.Vertexes[0].Point
            assert (q-App.Vector(*manifest['checkpoints'][i])).Length<1e-6
''')
        checked=subprocess.run([*freecad_command(),str(script),str(d/'cad')],capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
