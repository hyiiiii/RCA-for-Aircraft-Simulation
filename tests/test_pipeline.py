"""General dataset: counts, no algorithm dependency, strict faults and reproducibility."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pipeline_core import TrialConfig, run_pipeline, load_dataset
from assembly_sim.forward import fault_choices


class GeneralDatasetTests(unittest.TestCase):
    def test_config_rounding_and_strict_fault_locations(self):
        c=TrialConfig(n_total=11,test_fraction=.3)
        self.assertEqual((c.n_train,c.n_test),(8,3))
        for kw in ({'n_total':1},{'n_total':True},{'test_fraction':0},{'test_fraction':1},
                   {'test_fraction':float('nan')},{'fault_stage':2,'fault_part':'B'},
                   {'fault_stage':4,'fault_part':'A'},{'fault_stage':True},
                   {'fault_fraction':-1},{'process_sigma':-1}):
            with self.subTest(kw=kw),self.assertRaises(ValueError): TrialConfig(**kw)
        self.assertEqual(fault_choices(None,'B'),{1:('B',)})
        self.assertEqual(fault_choices(None,'A'),{1:('A',),2:('A',)})
        self.assertEqual(fault_choices(4,'ABC'),{4:('ABC',)})
        for obsolete in ('n_calibration','n_train','generation_mode','fault_point','quality_limit'):
            with self.assertRaises(TypeError): TrialConfig(**{obsolete:1})

    def test_split_trajectory_integrity_reproducibility_and_no_training(self):
        with tempfile.TemporaryDirectory() as temp:
            c=TrialConfig(dataset_name='general',n_total=20,test_fraction=.3,fault_fraction=.5,
                fault_stage=1,fault_part='B',data_root=temp+'/data',graph_root=temp+'/graph')
            with patch('pipeline_core.fit_reference',side_effect=AssertionError('generator trained an algorithm')):
                result=run_pipeline(c)
            d=Path(result['dataset_dir']); values=load_dataset(d)
            self.assertEqual(set(values),{'train','test'})
            self.assertEqual(values['train'].shape,(14,5,9,3));self.assertEqual(values['test'].shape,(6,5,9,3))
            self.assertFalse((d/'calibration_normal.csv').exists())
            self.assertTrue((d/'train_observed.csv').exists())
            self.assertFalse((Path(result['graph_dir'])/'normal_reference.json').exists())
            with np.load(d/'split_indices.npz') as indices:
                self.assertEqual(set(indices['train']) & set(indices['test']),set())
                self.assertEqual(set(indices['train'])|set(indices['test']),set(range(20)))
            labels=json.loads((d/'labels.json').read_text())
            injected=[x for x in labels if x['injected']]
            self.assertEqual(len(injected),10)
            self.assertTrue(all(x['stage']==1 and x['part']=='B' for x in injected))
            np.testing.assert_allclose([np.linalg.norm(x['offset']) for x in injected],c.fault_magnitude)
            self.assertGreater(len({tuple(x['offset']) for x in injected}),1)
            # Changing only train/test ratio never changes the generated physical runs.
            other=run_pipeline(replace(c,dataset_name='other',test_fraction=.5))
            other_d=Path(other['dataset_dir']); other_values=load_dataset(other_d)
            def source_map(folder,arrays):
                with np.load(folder/'split_indices.npz') as z:
                    return {int(source):arrays[split][i] for split in arrays for i,source in enumerate(z[split])}
            one,two=source_map(d,values),source_map(other_d,other_values)
            for i in one: np.testing.assert_array_equal(one[i],two[i])
            with self.assertRaises(FileExistsError): run_pipeline(c)

    def test_all_faults_honor_whole_assembly_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            result=run_pipeline(TrialConfig(dataset_name='all',n_total=4,test_fraction=.5,
                fault_fraction=1,fault_stage=4,fault_part='ABC',data_root=temp+'/d',graph_root=temp+'/g'))
            rows=json.loads((Path(result['dataset_dir'])/'labels.json').read_text())
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['injected'] and r['stage']==4 and r['part']=='ABC' for r in rows))
