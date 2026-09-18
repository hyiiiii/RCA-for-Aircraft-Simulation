"""Desktop helper checks; no Tk display or GUI initialization required."""
import json
from pathlib import Path
import tempfile
import unittest

from assembly_sim.graph import build_graph
from industrial_desktop_app import (
    DEFAULTS, ancestor_path_edges, atomic_write_json, preview_graph,
    validate_parameters,
)


class DesktopContractTests(unittest.TestCase):
    def test_parameters_match_pipeline_contract(self):
        values = validate_parameters(dict(DEFAULTS, fault_stage="1", fault_part="B"))
        self.assertEqual(values["fault_stage"], 1)
        self.assertEqual(values["fault_part"], "B")
        self.assertIsInstance(values["n_total"], int)
        self.assertIsInstance(values["test_fraction"], float)
        self.assertTrue(Path(values["data_root"]).is_absolute())
        self.assertIsNone(validate_parameters(DEFAULTS)["fault_stage"])

    def test_invalid_input_rejected_before_run(self):
        for key, value in (("dataset_name", "../outside"), ("seed", "-1"),
                           ("n_total", "1"), ("test_fraction", "1"), ("process_sigma", "nan"),
                           ("test_fraction", "0"), ("fault_fraction", "1.1"),
                           ("fault_part", "C9"), ("fault_stage", "5")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_parameters(dict(DEFAULTS, **{key: value}))

    def test_cross_point_trace_only_follows_existing_actions(self):
        graph = preview_graph()
        path = ancestor_path_edges(graph, "s3_C1", "s4_B2")
        self.assertEqual(path, {("s3_C1", "h4_final"), ("h4_final", "s4_B2")})
        self.assertEqual(ancestor_path_edges(graph, "s2_C1", "s2_B2"), set())
        self.assertEqual(ancestor_path_edges(graph, "s4_B2", "s3_C1"), set())

    def test_preview_uses_the_structural_graph(self):
        preview = {(edge["source"], edge["target"], edge["kind"]) for edge in preview_graph()["edges"]}
        official = {(edge["source"], edge["target"], edge["kind"]) for edge in build_graph(forward=True)["edges"]}
        self.assertEqual(preview, official)
        self.assertIn(("s3_C1", "h4_final", "localization"), official)
        self.assertIn(("h4_final", "s4_B2", "motion"), official)

    def test_atomic_ledger_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "history.json"
            atomic_write_json(path, {"状态": "完成", "runs": [1]})
            atomic_write_json(path, {"状态": "完成", "runs": [1, 2]})
            self.assertEqual(json.loads(path.read_text())["runs"], [1, 2])
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
