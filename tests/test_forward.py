"""Physical and storage contracts for the forward-only CAD model."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from assembly_sim.cad import CADGeometry, CADMotion, checkpoint_coordinates
from assembly_sim.forward import ForwardSettings, target_definition, simulate_forward, world_points, commands, rigid
from assembly_sim.model import ModelConfig
from assembly_sim.cad_view import CADViewData
from pipeline_core import TrialConfig, run_pipeline, analyze_dataset


class ForwardPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.local,self.poses,self.points=target_definition()

    def run_sample(self,seed=21,process=0,sensor=0,rotation=0,fault=None):
        return simulate_forward(self.local,self.poses,self.points,ModelConfig(process,sensor),ForwardSettings(process_rotation_deg=rotation),seed,fault)

    def test_local_frames_fixed_targets_and_random_initial_condition(self):
        np.testing.assert_allclose(world_points(self.local,self.poses[3]),checkpoint_coordinates(CADGeometry()))
        np.testing.assert_array_equal(self.poses[2],self.poses[3])
        np.testing.assert_array_equal(self.poses[0,:,:3,:3],np.tile(np.eye(3),(3,1,1)))
        np.testing.assert_allclose(self.poses[0,:,0,3],[-1150,0,1150])
        a,b=self.run_sample(),self.run_sample(seed=22)
        self.assertFalse(np.allclose(a.poses[0],b.poses[0]))
        self.assertFalse(np.allclose(a.poses[0,0,:3,:3],a.poses[0,1,:3,:3]))
        np.testing.assert_array_equal(a.poses,self.run_sample().poses)
        np.testing.assert_allclose(a.true[1:],self.points,atol=1e-9)
        np.testing.assert_allclose(a.commanded[3],np.tile(np.eye(4),(3,1,1)),atol=1e-9)

    def test_se3_process_error_preserves_rigidity_and_uses_observations(self):
        a=self.run_sample(process=.02,sensor=.03,rotation=.02)
        local=self.local.reshape(3,3,3)
        expected=np.linalg.norm(local[:,:,None]-local[:,None,:],axis=-1)
        for k in range(5):
            parts=a.true[k].reshape(3,3,3)
            np.testing.assert_allclose(np.linalg.norm(parts[:,:,None]-parts[:,None,:],axis=-1),expected,atol=1e-9)
            np.testing.assert_allclose(a.true[k],world_points(self.local,a.poses[k]),atol=1e-9)
        np.testing.assert_allclose(a.executed,a.process_errors@a.commanded,atol=1e-10)
        np.testing.assert_allclose(a.poses[1:],a.executed@a.poses[:-1],atol=1e-10)
        np.testing.assert_allclose(a.commanded[0],self.poses[0]@np.linalg.inv(a.estimated_poses[0]),atol=1e-9)
        np.testing.assert_array_equal(a.poses[2,1:],a.poses[1,1:])
        np.testing.assert_array_equal(a.poses[3,:2],a.poses[2,:2])
        for b in (1,2):
            np.testing.assert_array_equal(a.commanded[3,b],a.commanded[3,0])
            np.testing.assert_array_equal(a.executed[3,b],a.executed[3,0])
        clean_sensor=self.run_sample(process=.02,sensor=0,rotation=.02)
        np.testing.assert_array_equal(a.poses[0],clean_sensor.poses[0])
        self.assertGreater(np.linalg.norm(a.poses[1]-clean_sensor.poses[1]),1e-5)

    def test_fault_is_a_body_action_and_global_correction_preserves_internal_error(self):
        a=self.run_sample(fault=dict(stage=3,part='C',offset=[0,4,0]))
        np.testing.assert_allclose(a.true[3,6:]-self.points[2,6:],np.tile([0,4,0],(3,1)),atol=1e-9)
        self.assertGreater(np.linalg.norm(a.commanded[3]-np.eye(4)),1e-4)
        self.assertLess(np.linalg.norm(a.true[4]-self.points[3]),np.linalg.norm(a.true[3]-self.points[3]))
        self.assertGreater(np.linalg.norm(a.true[4]-self.points[3]),1)
        np.testing.assert_allclose(np.linalg.norm(a.true[4,:,None]-a.true[4,None,:],axis=-1),np.linalg.norm(a.true[3,:,None]-a.true[3,None,:],axis=-1),atol=1e-9)

    def test_mating_commands_follow_measured_body_and_restore_graph_inputs(self):
        from assembly_sim.graph import build_graph
        disturbance = rigid([.01,-.02,.03],[3,-5,2])
        for stage,b in ((2,0),(3,2)):
            previous=self.points[stage-2].copy()
            altered=previous.copy()
            altered[3:6]=previous[3:6] @ disturbance[:3,:3].T + disturbance[:3,3]
            h=commands(altered,stage,self.points)
            baseline=commands(previous,stage,self.points)
            self.assertGreater(np.linalg.norm(h[b]-baseline[b]),1)
            moved=world_points(altered,h)
            sl=slice(3*b,3*b+3)
            np.testing.assert_allclose(moved[sl],self.points[3,sl] @ disturbance[:3,:3].T+disturbance[:3,3],atol=1e-9)
            np.testing.assert_array_equal(moved[3:6],altered[3:6])
            # Old files can still be interpreted with their original controller.
            old=commands(altered,stage,self.points,controller_policy="fixed_world")
            np.testing.assert_allclose(old[b],baseline[b],atol=1e-9)
        edges={(e['source'],e['target']) for e in build_graph(forward=True)['edges']}
        for i in range(1,4):
            self.assertIn((f's1_B{i}','h2_front'),edges)
            self.assertIn((f's2_B{i}','h3_rear'),edges)

    def test_invalid_sampling_parameters(self):
        for kw in ({'initial_rotation_deg':181},{'initial_translation_mm':-1},{'process_rotation_deg':float('nan')},{'initial_rotation_deg':True}):
            with self.assertRaises(ValueError): ForwardSettings(**kw)
        with self.assertRaises(ValueError): TrialConfig(forward_settings={'typo':1})


class ForwardPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.config=TrialConfig(dataset_name='forward',n_total=20,test_fraction=.2,
            process_sigma=0,measurement_sigma=0,forward_settings={'process_rotation_deg':0},
            fault_fraction=1,fault_stage=3,fault_magnitude=4,
            data_root=cls.temp.name+'/data',graph_root=cls.temp.name+'/graph')
        cls.result=run_pipeline(cls.config); cls.directory=Path(cls.result['dataset_dir'])

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_output_separates_targets_truth_observations_and_actions(self):
        d=self.directory
        self.assertFalse((d/'planned_transforms.npz').exists())
        with np.load(d/'target_states.npz') as targets:
            self.assertEqual(targets['target_poses'].shape,(4,3,4,4))
            np.testing.assert_array_equal(targets['target_poses'][2],targets['target_poses'][3])
        with np.load(d/'evaluation/state_actions.npz') as audit:
            self.assertEqual(audit['test_commanded'].shape,(4,4,3,4,4))
            np.testing.assert_allclose(audit['test_executed'],audit['test_process_errors']@audit['test_commanded'],atol=1e-9)
        with (d/'checkpoints.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertTrue(all(r['target_x']=='' for r in rows if r['stage']=='0'))
        self.assertTrue(all(r['target_x']!='' for r in rows if r['stage']!='0'))
        view=CADViewData(d)
        _, actual_points=view.frame('test',0,3,actual=True)
        np.testing.assert_allclose(actual_points,view.actual['test'][0,3],atol=1e-9)
        self.assertEqual(json.loads((d/'metadata.json').read_text())['controller_policy'],'body_relative')
        labels=json.loads((d/'evaluation/labels.json').read_text())
        self.assertEqual({r['root'] for r in labels if r['injected']},{'h3_rear'})
        self.assertEqual(self.result['task'],'generation_only')
        self.assertNotIn('metrics',self.result)
        self.assertEqual({r['split'] for r in labels},{'train','test'})
        manifest=json.loads((d/'cad/geometry.json').read_text())
        self.assertEqual(set(manifest['body_frames']),set('ABC'))
        self.assertEqual(len(manifest['local_checkpoints']),9)
        self.assertTrue((d/'cad/A_local.step').is_file())

    def test_view_and_diagnosis_work_without_evaluation_files(self):
        d=self.directory; (d/'evaluation').rename(d/'hidden')
        try:
            view=CADViewData(d)
            h,p=view.frame('test',0,0)
            np.testing.assert_allclose(p,view.observed['test'][0,0],atol=1e-9)
            target_h,target_p=view.frame('test',0,1,target=True)
            np.testing.assert_allclose(target_p,view.plans['test'][0,1],atol=1e-9)
            self.assertEqual(set(view.poses),{'train','test'})
            self.assertFalse(view.actual)
            with self.assertRaisesRegex(ValueError,'正常训练样本'):
                analyze_dataset(d,Path(self.temp.name)/'without_truth')
        finally: (d/'hidden').rename(d/'evaluation')
