"""Opt-in real Tk/FreeCAD editor workflow: CAD_GUI_TEST=1 python -m unittest discover ..."""
import copy
import json
import os
from pathlib import Path
import time
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('CAD_GUI_TEST')=='1','requires native Tk display')
class CADEditorTests(unittest.TestCase):
    def test_edit_preview_validation_save_and_cancel(self):
        import tkinter as tk
        from assembly_sim.cad_editor import CADSettingsDialog
        from industrial_desktop_app import SimulationApp
        root=tk.Tk();errors=[]
        try:
            app=SimulationApp(root,persist=False)
            root.report_callback_exception=lambda *args:errors.append(args)
            app.tabs.select(app.diagnosis_tab);root.update()
            self.assertIs(app.graph_tab,app.diagnosis_tab)
            titles=[app.tabs.tab(t,'text') for t in app.tabs.tabs()]
            self.assertIn('根因溯源',titles);self.assertNotIn('坐标',titles);self.assertNotIn('因果图',titles)
            self.assertFalse(hasattr(app,'diagnosis_reference'))
            self.assertGreater(app.canvas.winfo_height(),250)
            saved=[];dialog=CADSettingsDialog(root,{},saved.append)
            def wait_build():
                deadline=time.monotonic()+30
                while (dialog.busy or dialog.surface_pending) and time.monotonic()<deadline:
                    root.update();time.sleep(.01)
                root.update();self.assertFalse(dialog.busy,dialog.status.get())
            wait_build()
            original=copy.deepcopy(dialog.points)
            self.assertIsNotNone(dialog.mesh_geometry,dialog.status.get())
            for part,count in zip('ABC',[4,5,6]):
                dialog.tabs.select('ABC'.index(part));root.update()
                preview=dialog.previews[part]
                self.assertGreater(preview.winfo_height(),170)
                self.assertEqual(len(preview.find_withtag('local_axis')),3)
                self.assertGreater(len(preview.find_withtag('cad_mesh')),10)
                from types import SimpleNamespace
                preview.yaw=np.pi/2 if part=='A' else 0;preview.pitch=0;preview.schedule();root.update()
                expected=np.array([300,23,44]) if part=='A' else np.array([12 if part=='B' else -200,np.sqrt(300**2-40**2),40])
                screen=(expected @ preview.camera.T)[:2]
                screen=(screen-preview.view_centre[:2])*preview.scale+preview.screen_centre
                event=SimpleNamespace(x=screen[0],y=screen[1])
                before=copy.deepcopy(dialog.points[part]);preview.press(event);preview.release(event);wait_build()
                np.testing.assert_allclose(preview.candidate,expected,atol=1e-6)
                self.assertEqual(dialog.points[part],before)
                for _ in range(count-3):
                    before=copy.deepcopy(dialog.points[part]);dialog.random_point(part);wait_build()
                    self.assertEqual(dialog.points[part],before)
                    self.assertIsNotNone(preview.candidate,dialog.status.get())
                    self.assertGreater(len(preview.find_withtag('candidate')),0)
                    self.assertGreater(np.linalg.norm(preview.vertices-preview.candidate,axis=1).min(),1e-5)
                    dialog.edit_point(part,False);root.update()
                self.assertEqual(len(dialog.points[part]),count)
                # Selected point remains unchanged until an explicit modify operation.
                name=dialog.tables[part].selection()[0];before=copy.deepcopy(dialog.points[part])
                dialog.random_point(part);wait_build();self.assertEqual(dialog.points[part],before)
                dialog.cancel_candidate(part);root.update()
                self.assertIsNone(preview.candidate);self.assertEqual(dialog.points[part],before)
                dialog.random_point(part);wait_build();candidate=preview.candidate.copy()
                dialog.edit_point(part,True);root.update()
                self.assertEqual(len(dialog.points[part]),count)
                np.testing.assert_allclose(next(r['position'] for r in dialog.points[part] if r['name']==name),candidate)
            dialog.tabs.select(3);dialog.motion_vars['final_yaw_deg'].set('90');dialog.motion_vars['final_x_mm'].set('400');root.update()
            self.assertEqual(len(dialog.target_preview.find_withtag('body_axis')),9)
            np.testing.assert_allclose(dialog.target_preview.target_poses[1,:3,3],[400,0,0])
            np.testing.assert_allclose(dialog.target_preview.target_poses[1,:3,0],[0,1,0],atol=1e-12)
            self.assertGreater(dialog.target_preview.winfo_height(),160)
            # Add a real surface point, modify it, delete it; IDs of remaining points stay stable.
            part='A';dialog.tabs.select(0);root.update()
            dialog.pick(part,[300,40,80]);dialog.edit_point(part,False);root.update()
            name=dialog.tables[part].selection()[0];self.assertEqual(len(dialog.points[part]),5)
            dialog.pick(part,[300,55,80]);dialog.edit_point(part,True);root.update()
            np.testing.assert_allclose(dialog.points[part][-1]['position'],[300,55,80])
            dialog.delete_point(part);root.update();self.assertEqual(len(dialog.points[part]),4)
            self.assertNotIn(name,dialog.tables[part].get_children())
            # Interior point must not reach the callback; restore and save the valid edit.
            valid=copy.deepcopy(dialog.points)
            dialog.points['A'][0]['position']=[0,0,0];dialog.save();wait_build()
            self.assertEqual(saved,[]);self.assertIn('CAD 表面',dialog.status.get())
            dialog.points=valid;dialog.save();wait_build()
            self.assertEqual(len(saved),1);self.assertEqual([len(saved[0]['cad_checkpoints'][p]) for p in 'ABC'],[4,5,6])
            other=CADSettingsDialog(root,saved[0],saved.append)
            other.geometry_vars['body_length'].set('1250');other.close();root.update()
            self.assertEqual(len(saved),1);self.assertEqual(saved[0]['cad_geometry']['body_length'],1200)
            self.assertFalse(errors,errors)
        finally:root.destroy()
