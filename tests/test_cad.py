"""CAD surface, reverse-motion and plan-conditioned diagnosis invariants."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from assembly_sim.cad import CADGeometry, CADMotion, checkpoint_coordinates, freecad_command, make_plan
from assembly_sim.model import Fault, ModelConfig, apply_transforms, simulate
from assembly_sim.tracing import fit_reference, innovations, trace_trajectory
from pipeline_core import TrialConfig, analyze_dataset, load_dataset, load_plans, run_pipeline


class ReversePlanTests(unittest.TestCase):
    def setUp(self):
        self.geometry = CADGeometry()
        self.motion = CADMotion()
        self.plan = make_plan(self.geometry, self.motion, 81)

    def test_reverse_motion_keeps_fixed_groups_and_reaches_complete_target(self):
        p = self.plan
        np.testing.assert_array_equal(p.checkpoints[4], checkpoint_coordinates(self.geometry))
        np.testing.assert_array_equal(p.checkpoints[3, :6], p.checkpoints[2, :6])
        np.testing.assert_array_equal(p.checkpoints[2, 3:], p.checkpoints[1, 3:])
        for b in (1, 2):
            np.testing.assert_array_equal(p.poses[3, b], p.poses[3, 0])
        for k in range(4):
            for transforms, source, target in ((p.reverse[k], p.checkpoints[k+1], p.checkpoints[k]),
                                               (p.forward[k], p.checkpoints[k], p.checkpoints[k+1])):
                np.testing.assert_allclose(apply_transforms(source, transforms[:, :3, :3], transforms[:, :3, 3]), target, atol=1e-10)
            for b in range(3):
                np.testing.assert_allclose(p.reverse[k, b]@p.forward[k, b], np.eye(4), atol=1e-10)
                np.testing.assert_allclose(p.forward[k, b,:3,:3].T@p.forward[k, b,:3,:3], np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(p.forward[k,b,:3,:3]), 1)
        for b in range(3):
            group = p.checkpoints[:, b*3:b*3+3]
            distances = np.linalg.norm(group[:,:,None]-group[:,None,:],axis=-1)
            np.testing.assert_allclose(distances, np.broadcast_to(distances[4],distances.shape), atol=1e-10)

    def test_targets_are_fixed_aligned_and_final_ideal_action_is_identity(self):
        p,m=self.plan,self.motion
        for seed in (0,1,82,100):
            np.testing.assert_array_equal(make_plan(self.geometry,m,seed).poses,p.poses)
        np.testing.assert_array_equal(p.poses[:,:,:3,:3],np.tile(np.eye(3),(5,3,1,1)))
        np.testing.assert_array_equal(p.checkpoints[1,:,1:],p.checkpoints[4,:,1:])
        np.testing.assert_array_equal(p.poses[1,:,:3,3],[[-m.front_gap_mm,0,0],[0,0,0],[m.rear_gap_mm,0,0]])
        np.testing.assert_array_equal(p.checkpoints[3],p.checkpoints[4])
        np.testing.assert_array_equal(p.forward[3],np.tile(np.eye(4),(3,1,1)))
        np.testing.assert_array_equal(p.reverse[3],np.tile(np.eye(4),(3,1,1)))

    def test_final_actual_adjustment_is_caused_by_forward_errors(self):
        clean=simulate(ModelConfig(0,0),planned=self.plan.checkpoints)
        np.testing.assert_allclose(clean.rotations[3],np.tile(np.eye(3),(3,1,1)),atol=1e-12)
        np.testing.assert_allclose(clean.translations[3],0,atol=1e-10)
        faulty=simulate(ModelConfig(0,0),fault=Fault(3,"C1",(0,2,0)),planned=self.plan.checkpoints)
        self.assertGreater(np.linalg.norm(faulty.rotations[3]-np.eye(3)),1e-6)
        self.assertGreater(np.linalg.norm(faulty.translations[3]),1e-6)
        before=np.linalg.norm(faulty.true[3]-self.plan.checkpoints[4])
        after=np.linalg.norm(faulty.true[4]-self.plan.checkpoints[4])
        self.assertLess(after,before)
        np.testing.assert_array_equal(faulty.nominal,self.plan.checkpoints)

    def test_forward_feedback_matches_reverse_plan_without_noise_and_faults(self):
        p = self.plan
        healthy = simulate(ModelConfig(0,0), planned=p.checkpoints)
        np.testing.assert_allclose(healthy.true,p.checkpoints,atol=1e-10)
        np.testing.assert_allclose(healthy.rotations,p.forward[:,:,:3,:3],atol=1e-10)
        np.testing.assert_allclose(healthy.translations,p.forward[:,:,:3,3],atol=1e-10)
        faulty = simulate(ModelConfig(0,0),fault=Fault(3,"C1",(0,2,0)),planned=p.checkpoints)
        expected = np.zeros((4,9,3)); expected[2,6]=[0,2,0]
        np.testing.assert_allclose(innovations(faulty.observed,p.checkpoints),expected,atol=1e-10)
        self.assertGreater(np.linalg.norm(faulty.true[4,4]-healthy.true[4,4]),0.01)
        np.testing.assert_array_equal(p.checkpoints,self.plan.checkpoints)

    def test_invalid_geometry_and_random_bounds_are_rejected(self):
        for kwargs in ({"radius":0},{"nose_length":100},{"exhaust_offset":5},{"radius":float('nan')},{"tail_length":True},
                       {"fin_root_chord":700},{"fin_sweep":400},{"fin_thickness":400}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): CADGeometry(**kwargs)
        for kwargs in ({"front_gap_mm":0},{"rear_gap_mm":-1},{"initial_offset_mm":-1},{"initial_offset_mm":float('inf')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): CADMotion(**kwargs)
        for kwargs in ({"generation_mode":"wrong"},{"cad_motion":{"typo":1}},{"cad_geometry":[]}):
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)): TrialConfig(**kwargs)


class FreeCADPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: freecad_command()
        except RuntimeError as exc: raise unittest.SkipTest(str(exc))
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = TrialConfig(dataset_name="cad_test",n_total=20,test_fraction=.2,
                                 process_sigma=0,measurement_sigma=0,fault_fraction=0,
                                 data_root=cls.temp.name+"/data",graph_root=cls.temp.name+"/graph")
        cls.result = run_pipeline(cls.config)
        cls.directory = Path(cls.result['dataset_dir'])
        cls.values, cls.plans = load_dataset(cls.directory), load_plans(cls.directory)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_native_cad_surfaces_and_complete_exports(self):
        g = CADGeometry()
        manifest = json.loads((self.directory/'cad/geometry.json').read_text())
        self.assertIn('FreeCAD',manifest['engine'])
        self.assertAlmostEqual(manifest['validation'][0]['volume_mm3'],2/3*np.pi*g.nose_length*g.radius**2,places=4)
        self.assertAlmostEqual(manifest['validation'][1]['volume_mm3'],np.pi*g.radius**2*g.body_length,places=4)
        for item in manifest['validation']:
            self.assertTrue(item['valid']); self.assertEqual(item['solids'],1)
            self.assertLess(max(item['surface_distances_mm']),1e-6)
        tail = manifest['validation'][2]['bounding_box_mm']
        self.assertAlmostEqual(tail['y_min'],-g.radius-g.fin_span)
        self.assertAlmostEqual(tail['y_max'],g.radius+g.fin_span)
        self.assertAlmostEqual(tail['x_max'],g.body_length/2+g.tail_length+g.exhaust_length)
        self.assertEqual(manifest['world_frame']['origin'],[0,0,0])
        self.assertTrue(manifest['world_frame']['fixed'])
        self.assertEqual([a['name'] for a in manifest['world_frame']['axes']],['X_W','Y_W','Z_W'])
        for name in ('aircraft.FCStd','aircraft.step','A.step','B.step','C.step','A.stl','B.stl','C.stl'):
            self.assertGreater((self.directory/'cad'/name).stat().st_size,100)
        for name in ('checkpoints.csv','target_states.npz','target_states.json','labels.json','cad_preview.html'):
            self.assertTrue((self.directory/name).is_file())
        self.assertNotIn('__CAD_PAYLOAD__',(self.directory/'cad_preview.html').read_text())



if __name__ == '__main__': unittest.main()
