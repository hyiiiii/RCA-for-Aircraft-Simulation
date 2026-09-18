"""Variable checkpoint counts must preserve geometry, mating and diagnosis contracts."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from assembly_sim.cad import CADGeometry, export_cad
from assembly_sim.checkpoints import checkpoint_definition, as_config, point_groups
from assembly_sim.forward import body_frames, target_definition, world_points, commands, rigid
from assembly_sim.cad_view import CADViewData
from assembly_sim.graph import build_graph, graph_layout
from pipeline_core import TrialConfig, run_pipeline, analyze_dataset, load_dataset


class VariableCheckpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        g=CADGeometry();manifest=export_cad(cls.root/'preview',g);origins=body_frames(g)[:,:3,3]
        cls.config={}
        for b,(part,count) in enumerate(zip('ABC',[4,5,6])):
            vertices=np.unique(np.asarray(manifest['meshes'][b]['vertices'])-origins[b],axis=0)
            sample=vertices[np.random.default_rng(13+b).choice(len(vertices),count,replace=False)]
            # Missing A1 and nonsequential IDs expose accidental global defaults.
            cls.config[part]=[dict(name=f'{part}{2*i+2}',position=p.tolist()) for i,p in enumerate(sample)]
        cls.generated=run_pipeline(TrialConfig(dataset_name='variable',n_total=60,fault_fraction=.25,
            cad_checkpoints=cls.config,data_root=str(cls.root/'data'),graph_root=str(cls.root/'graph')))
        cls.dataset=Path(cls.generated['dataset_dir'])

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_unequal_groups_still_follow_observed_body(self):
        names,local,assembled=checkpoint_definition(CADGeometry(),self.config)
        _,poses,goals=target_definition(checkpoints=assembled,point_names=names)
        initial=poses[0].copy();initial[1]=rigid([0,0,.02],[1,2,3])
        before=world_points(local,initial,names)
        h=commands(before,2,goals,point_names=names);after=world_points(before,h,names)
        a,b,c=point_groups(names)
        np.testing.assert_allclose(after[a],goals[3,a]@initial[1,:3,:3].T+initial[1,:3,3],atol=1e-8)
        np.testing.assert_array_equal(after[np.r_[b,c]],before[np.r_[b,c]])
        # Names, not array position or contiguous triplets, determine membership.
        order=np.random.default_rng(8).permutation(len(names));shuffled=tuple(names[i] for i in order)
        hh=commands(before[order],2,goals[:,order],point_names=shuffled)
        np.testing.assert_allclose(hh,h,atol=1e-8)

    def test_exports_viewer_and_diagnosis_use_same_fifteen_points(self):
        d=self.dataset;names=tuple(json.loads((d/'metadata.json').read_text())['point_names'])
        observed=load_dataset(d)
        self.assertEqual(observed['test'].shape,(12,5,15,3))
        with (d/'observations.csv').open() as f:self.assertEqual(sum(1 for _ in csv.DictReader(f)),60*5*15)
        view=CADViewData(d);self.assertEqual(view.point_names,names)
        _,points=view.frame('test',0,4,actual=True)
        np.testing.assert_allclose(points,view.actual['test'][0,4],atol=1e-8)
        _,targets=view.frame('test',0,1,target=True)
        np.testing.assert_allclose(targets,view.plans['test'][0,1],atol=1e-8)
        result=analyze_dataset(d,self.root/'diagnosis')
        reference=json.loads((self.root/'diagnosis/normal_reference.json').read_text())
        self.assertEqual(reference['point_names'],list(names));self.assertEqual(np.shape(reference['center']),(4,15,3))
        traces=json.loads(Path(result['files']['traces']).read_text())
        self.assertEqual(set(traces[0]['final_errors_mm']),set(names))
        graph=build_graph(forward=True,point_names=names)
        positions,_,width,height=graph_layout(graph)
        self.assertEqual(set(positions),{n['id'] for n in graph['nodes']})
        self.assertGreater(positions['s4_A2'][0],positions['s0_A2'][0])
        self.assertIn("'ABC'.indexOf(DATA.pointNames[i][0])",(d/'cad_preview.html').read_text())

    def test_invalid_points_rejected_before_generation(self):
        g=CADGeometry();names,local,_=checkpoint_definition(g);default=as_config(names,local)
        invalid=[]
        too_few=copy.deepcopy(default);too_few['A'].pop();invalid.append(too_few)
        duplicates=copy.deepcopy(default);duplicates['B'][1]['position']=duplicates['B'][0]['position'];invalid.append(duplicates)
        collinear=copy.deepcopy(default)
        for i,row in enumerate(collinear['C']):row['position']=[i,0,0]
        invalid.append(collinear)
        bad_name=copy.deepcopy(default);bad_name['A'][0]['name']='B9';invalid.append(bad_name)
        for config in invalid:
            with self.assertRaises(ValueError):TrialConfig(cad_checkpoints=config)
        inside=copy.deepcopy(default);inside['A'][0]['position']=[0,0,0]
        with self.assertRaisesRegex(RuntimeError,'CAD 表面'):
            export_cad(self.root/'off_surface',g,inside)
