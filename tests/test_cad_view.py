"""Viewer data must use the exported transforms and preserve rigid interpolation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from assembly_sim.cad import CADGeometry, CADMotion, checkpoint_coordinates, make_plan
from assembly_sim.cad_view import CADViewData, interpolate_pose
from assembly_sim.model import POINT_NAMES


class CADViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'cad').mkdir()
        self.q = checkpoint_coordinates(CADGeometry())
        manifest = dict(point_names=list(POINT_NAMES),checkpoints=self.q.tolist(),
                        meshes=[dict(part=part,vertices=self.q[i*3:i*3+3].tolist(),triangles=[[0,1,2]])
                                for i,part in enumerate('ABC')])
        (self.root/'cad/geometry.json').write_text(json.dumps(manifest))
        # Include more samples than the standalone HTML's 20-sample preview cap.
        plans = [make_plan(CADGeometry(),CADMotion(),seed) for seed in range(21)]
        self.points = np.stack([p.checkpoints for p in plans])
        self.poses = np.stack([p.poses for p in plans])
        np.savez(self.root/'planned_checkpoints.npz',**{s:self.points for s in ('train','calibration','test')})
        np.savez(self.root/'planned_transforms.npz',**{s+'_poses':self.poses for s in ('train','calibration','test')})
        np.savez(self.root/'observations.npz',**{s:self.points+0.03 for s in ('train','calibration','test')})

    def tearDown(self): self.temp.cleanup()

    def test_all_samples_and_discrete_stages_match_exported_coordinates(self):
        data = CADViewData(self.root)
        for split in ('train','calibration','test'):
            for sample in (0,20):
                for stage in range(5):
                    poses,points = data.frame(split,sample,stage)
                    np.testing.assert_array_equal(poses,self.poses[sample,stage])
                    np.testing.assert_allclose(points,self.points[sample,stage],atol=1e-10)
        self.assertFalse((self.root/'evaluation').exists())

    def test_intermediate_motion_is_rigid_including_half_turn(self):
        a = np.tile(np.eye(4),(3,1,1)); b=a.copy()
        b[:,:3,:3] = np.diag([-1,-1,1]); b[:,:3,3]=[10,20,30]
        h = interpolate_pose(a,b,0.5)
        np.testing.assert_allclose(h[:,:3,3],[[5,10,15]]*3)
        np.testing.assert_allclose(h[:,:3,:3]@h[:,:3,:3].swapaxes(-1,-2),np.tile(np.eye(3),(3,1,1)),atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(h[:,:3,:3]),1)
        np.testing.assert_array_equal(interpolate_pose(a,b,0),a)
        np.testing.assert_array_equal(interpolate_pose(a,b,1),b)

    def test_loaded_preview_interpolation_preserves_part_distances(self):
        data=CADViewData(self.root)
        for stage in (0.5,1.25,2.9,3.5):
            _,points=data.frame('test',4,stage)
            for b in range(3):
                q,p=self.q[b*3:b*3+3],points[b*3:b*3+3]
                np.testing.assert_allclose(np.linalg.norm(q[:,None]-q[None,:],axis=-1),np.linalg.norm(p[:,None]-p[None,:],axis=-1),atol=1e-10)

    def test_world_reference_does_not_follow_assembly_motion(self):
        data=CADViewData(self.root)
        frame=json.dumps(data.world_frame,sort_keys=True)
        for stage in (0,1,2,3,4): data.frame('test',0,stage)
        self.assertEqual(json.dumps(data.world_frame,sort_keys=True),frame)
        self.assertTrue(data.world_frame['fixed'])
        np.testing.assert_array_equal(data.world_frame['origin'],[0,0,0])

    def test_mismatched_plan_fails_instead_of_showing_wrong_geometry(self):
        np.savez(self.root/'planned_checkpoints.npz',**{s:self.points+1 for s in ('train','calibration','test')})
        with self.assertRaisesRegex(ValueError,'不匹配'): CADViewData(self.root)

    def test_invalid_stage_sample_and_rotation_fail(self):
        data=CADViewData(self.root)
        for args in (('test',21,0),('test',-1,0),('test',0,5),('test',0,float('nan')),('other',0,0)):
            with self.subTest(args=args),self.assertRaises(ValueError): data.frame(*args)
        self.poses[0,0,0,0,0]=2
        np.savez(self.root/'planned_transforms.npz',**{s+'_poses':self.poses for s in ('train','calibration','test')})
        with self.assertRaisesRegex(ValueError,'刚体'): CADViewData(self.root)


if __name__=='__main__': unittest.main()
