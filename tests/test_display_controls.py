"""UI path semantics, ground-truth isolation and visible checkbox overlays."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from assembly_sim.graph import build_graph,directed_path,inference_path_edges,render_svg
from assembly_sim.diagnosis import truth_annotations


class GraphDisplayTests(unittest.TestCase):
    def test_red_edges_are_exact_inferred_path_with_independent_truth(self):
        graph=build_graph(forward=True);path=directed_path(graph,'h1_A','s4_B1')
        expected=set(zip(path,path[1:]))
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'graph.svg'
            render_svg(target,graph,'h1_A','s4_B1',true_root='h3_rear',inferred_path=path)
            root=ET.fromstring(target.read_text())
            drawn={(n.attrib['data-source'],n.attrib['data-target']) for n in root.iter() if n.attrib.get('class')=='inferred-path'}
            self.assertEqual(drawn,expected)
            self.assertTrue(any(n.attrib.get('class')=='true-root' for n in root.iter()))
            self.assertNotIn('并非已确认',target.read_text())
            render_svg(target,graph,'h1_A','s4_B1',inferred_path=[])
            self.assertNotIn('class="inferred-path"',target.read_text())
        with self.assertRaises(ValueError):inference_path_edges(graph,['h1_A','s4_B1'])

    def test_missing_truth_differs_from_healthy_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.assertEqual(truth_annotations(d),{})
            rows=[dict(sample_id='test_00000',injected=False,root=None),dict(sample_id='test_00001',injected=True,root='h3_rear')]
            (d/'labels.json').write_text(json.dumps(rows))
            truth=truth_annotations(d)
            self.assertFalse(truth['test_00000']['injected']);self.assertEqual(truth['test_00001']['root'],'h3_rear')
            (d/'labels.json').write_text(json.dumps(rows+[rows[1]]))
            self.assertNotIn('test_00001',truth_annotations(d))


@unittest.skipUnless(os.environ.get('CAD_GUI_TEST')=='1','requires native Tk display')
class DisplayControlTests(unittest.TestCase):
    def test_toggles_targets_and_root_annotations(self):
        import tkinter as tk
        from tkinter import ttk
        from industrial_desktop_app import SimulationApp
        from pipeline_core import TrialConfig,run_pipeline,analyze_dataset
        with tempfile.TemporaryDirectory() as tmp:
            generated=run_pipeline(TrialConfig(dataset_name='display',n_total=40,fault_fraction=.5,fault_stage=3,fault_part='C',fault_magnitude=4,data_root=tmp+'/data',graph_root=tmp+'/graph'))
            result=analyze_dataset(generated['dataset_dir'],Path(tmp)/'diagnosis')
            root=tk.Tk();errors=[]
            try:
                app=SimulationApp(root,persist=False);root.report_callback_exception=lambda *args:errors.append(args)
                app.show_result(result);app.tabs.select(app.graph_tab);root.update()
                example=result['example'];expected=inference_path_edges(app.graph,example['path'])
                drawn={t for item in app.canvas.find_withtag('inferred_path') for t in app.canvas.gettags(item) if '->' in t}
                self.assertEqual(drawn,{a+'->'+b for a,b in expected})
                self.assertEqual(len(app.canvas.find_withtag('true_root')),2)
                self.assertIn('真实根因：h3_rear',app.diagnosis_caption.get())
                app.truth_annotations[example['sample_id']]=dict(injected=True,root='h1_B');app.draw_graph()
                self.assertEqual({t for item in app.canvas.find_withtag('inferred_path') for t in app.canvas.gettags(item) if '->' in t},drawn)
                app.initial_toggle.invoke();root.update();self.assertFalse(app.show_initial.get())
                self.assertFalse(app.initial_toggle.instate(['selected']))
                app.initial_toggle.invoke();root.update();self.assertTrue(app.show_initial.get())
                view=app.cad_tab;app.tabs.select(view);root.update()
                view.layer.set('阶段目标');view.set_stage(4);view.show_targets.set(False);root.update()
                self.assertEqual(view.canvas.find_withtag('target_point'),())
                view.target_toggle.invoke();root.update()
                target_items=view.canvas.find_withtag('target_point')
                self.assertEqual(len(target_items),len(view.data.point_names))
                self.assertGreater(min(target_items),max(view.canvas.find_withtag('point_A1')))
                self.assertTrue(view.target_toggle.instate(['selected']))
                view.target_toggle.invoke();root.update();self.assertEqual(view.canvas.find_withtag('target_point'),())
                view.set_stage(0);view.target_toggle.invoke();root.update()
                self.assertEqual(view.target_toggle.cget('text'),'首次调姿目标点')
                self.assertEqual(len(view.canvas.find_withtag('target_point')),len(view.data.point_names))
                self.assertEqual(view.table.item('A1')['values'][-1],'—')
                view.world_toggle.invoke();root.update()
                self.assertFalse(view.show_world.get());self.assertEqual(view.canvas.find_withtag('world_axis'),())
                view.world_toggle.invoke();root.update();self.assertEqual(len(view.canvas.find_withtag('world_axis')),3)
                self.assertEqual(len(view.canvas.find_withtag('body_axis')),9)
                layout=str(ttk.Style(root).layout('TCheckbutton'))
                self.assertIn('Checked.indicator',layout)
                checked=root._checkmark_images[1]
                self.assertEqual(checked.get(8,13),(255,255,255));self.assertNotEqual(checked.get(5,5),(255,255,255))
                self.assertFalse(errors,errors)
            finally:root.destroy()
