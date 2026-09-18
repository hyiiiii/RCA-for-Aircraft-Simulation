"""Independent dataset diagnosis: reference selection, input isolation and mating math."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pipeline_core import TrialConfig,run_pipeline,analyze_dataset,load_dataset,load_plans
from assembly_sim.diagnosis import prepare_reference
from assembly_sim.forward import target_definition,commands,world_points,rigid,fit_pose


class IndependentDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.generated=run_pipeline(TrialConfig(dataset_name='generic',n_total=60,test_fraction=.3,
            fault_fraction=.5,fault_stage=3,fault_part='C',fault_magnitude=4,
            data_root=str(cls.root/'data'),graph_root=str(cls.root/'graph')))
        cls.dataset=Path(cls.generated['dataset_dir'])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_existing_train_test_dataset_runs_without_rewriting_data(self):
        d=self.dataset
        before={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file()}
        result=analyze_dataset(d,self.root/'diagnosis',quality_limit=1)
        after={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(result['task'],'diagnosis')
        self.assertEqual(result['metrics']['n_test'],18)
        self.assertEqual(result['metrics']['root_top1_on_detected'],1)
        reference=json.loads((self.root/'diagnosis/normal_reference.json').read_text())
        fit,threshold=set(reference['fit_sample_ids']),set(reference['threshold_sample_ids'])
        self.assertFalse(fit&threshold)
        labels={r['sample_id']:r for r in json.loads((d/'labels.json').read_text())}
        self.assertTrue(all(s.startswith('train_') and not labels[s]['injected'] for s in fit|threshold))
        self.assertFalse((d/'calibration_normal.csv').exists())

    def test_test_labels_and_audit_truth_cannot_change_predictions(self):
        d=self.dataset;path=d/'labels.json';original=path.read_text()
        (d/'evaluation').rename(d/'hidden_audit')
        try:
            baseline=analyze_dataset(d,self.root/'before')
            rows=json.loads(original)
            for row in rows:
                if row['split']=='test':
                    row.update(injected=not row['injected'],root='h1_B',stage=1,part='B',offset=[999,999,999])
            path.write_text(json.dumps(rows))
            changed=analyze_dataset(d,self.root/'changed')
            self.assertEqual(Path(baseline['files']['traces']).read_text(),Path(changed['files']['traces']).read_text())
            path.write_text(json.dumps([r for r in rows if r['split']=='train']))
            missing=analyze_dataset(d,self.root/'missing')
            self.assertEqual(Path(baseline['files']['traces']).read_text(),Path(missing['files']['traces']).read_text())
            self.assertIn('evaluation',missing['metrics'])
            # No labels at all is supported with an explicitly supplied saved reference.
            path.unlink()
            external=analyze_dataset(d,self.root/'external',reference_file=self.root/'before/normal_reference.json')
            self.assertEqual(Path(baseline['files']['traces']).read_text(),Path(external['files']['traces']).read_text())
        finally:
            path.write_text(original);(d/'hidden_audit').rename(d/'evaluation')

    def test_missing_and_duplicate_training_labels_fail_explicitly(self):
        d=self.dataset;path=d/'labels.json';original=path.read_text()
        try:
            rows=json.loads(original);train=[r for r in rows if r['split']=='train']
            path.write_text(json.dumps(rows+[train[0]]))
            with self.assertRaisesRegex(ValueError,'训练集标签'):
                prepare_reference(d,load_dataset(d),load_plans(d),json.loads((d/'metadata.json').read_text()))
            for row in rows:
                if row['split']=='train':row['injected']=True
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'至少需要 4'):
                analyze_dataset(d,self.root/'too_few')
        finally:path.write_text(original)


class MatingExampleTests(unittest.TestCase):
    def test_example_matches_current_cad_and_local_pose_formula(self):
        local,targets,goal=target_definition()
        poses=targets[0].copy();poses[1]=rigid([0,0,np.deg2rad(1)],[0,2,0])
        before=world_points(local,poses);h=commands(before,2,goal)
        np.testing.assert_allclose(h[0,:3,3],[249.961923789,6.363101609,0],atol=1e-8)
        after=world_points(before,h)
        np.testing.assert_allclose(after[0],[-754.649510938,261.825162871,99.347873938],atol=1e-6)
        np.testing.assert_array_equal(after[3:],before[3:])
        dba=np.linalg.inv(targets[3,1])@targets[3,0]
        ta=fit_pose(local[:3],before[:3]);tb=fit_pose(local[3:6],before[3:6])
        np.testing.assert_allclose(h[0],tb@dba@np.linalg.inv(ta),atol=1e-10)
        # At B=I, t_B=0 the same setup is simply +250 mm X, no rotation.
        baseline=commands(goal[0],2,goal)
        np.testing.assert_allclose(baseline[0,:3,:3],np.eye(3),atol=1e-10)
        np.testing.assert_allclose(baseline[0,:3,3],[250,0,0],atol=1e-10)
