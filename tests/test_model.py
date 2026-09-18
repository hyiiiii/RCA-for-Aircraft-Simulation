"""Tests of physical feedback semantics, not just the implementation layout."""

import json
import unittest

import numpy as np

from assembly_sim.model import (
    Fault,
    ModelConfig,
    POINT_NAMES,
    apply_transforms,
    model_spec,
    nominal_trajectory,
    predict_next,
    rigid_fit,
    simulate,
    stage_transforms,
)


class ModelTests(unittest.TestCase):
    def test_rigid_fit_recovers_proper_rotation_translation(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        theta = 0.43
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        translation = np.array([4.0, -2.0, 7.0])
        fitted_r, fitted_t = rigid_fit(points, points @ rotation.T + translation)
        np.testing.assert_allclose(fitted_r, rotation, atol=1e-12)
        np.testing.assert_allclose(fitted_t, translation, atol=1e-12)
        np.testing.assert_allclose(fitted_r.T @ fitted_r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted_r), 1.0)
        reflected = points * np.array([-1.0, 1.0, 1.0])
        fitted_r, _ = rigid_fit(points, reflected)
        self.assertAlmostEqual(np.linalg.det(fitted_r), 1.0)

    def test_noiseless_healthy_reproduces_analytical_targets(self):
        run = simulate(ModelConfig(0, 0), seed=0)
        self.assertEqual(run.observed.shape, (5, 9, 3))
        self.assertEqual(run.rotations.shape, (4, 3, 3, 3))
        self.assertEqual(run.translations.shape, (4, 3, 3))
        np.testing.assert_allclose(run.true, nominal_trajectory(), atol=1e-10)
        np.testing.assert_array_equal(run.observed, run.true)
        # The normal relative locations remain separated; A and B are not fit
        # to have coincident checkpoint coordinates.
        np.testing.assert_allclose(run.true[2, :3] - run.true[2, 3:6], [[-500, 0, 0]] * 3, atol=1e-10)

    def test_locked_groups_share_the_same_transform(self):
        run = simulate(ModelConfig(), seed=4, fault=Fault(1, "B1", (0.7, -0.4, 0.3)))
        # After front mating, A and B remain fixed together during rear mating.
        np.testing.assert_array_equal(run.rotations[2, 0], run.rotations[2, 1])
        np.testing.assert_array_equal(run.translations[2, 0], run.translations[2, 1])
        # Once all parts are connected, all three execute exactly the same H4.
        for part in (1, 2):
            np.testing.assert_array_equal(run.rotations[3, 0], run.rotations[3, part])
            np.testing.assert_array_equal(run.translations[3, 0], run.translations[3, part])
        moved = apply_transforms(run.true[3], run.rotations[3], run.translations[3])
        before = np.linalg.norm(run.true[3, :, None] - run.true[3, None, :], axis=-1)
        after = np.linalg.norm(moved[:, None] - moved[None, :], axis=-1)
        np.testing.assert_allclose(after, before, atol=1e-10)

    def test_c1_fault_changes_other_parts_through_actual_feedback(self):
        config = ModelConfig(0, 0)
        healthy = simulate(config, seed=2)
        faulty = simulate(config, seed=2, fault=Fault(3, "C1", (0.0, 1.2, 0.0)))
        np.testing.assert_array_equal(faulty.true[:3], healthy.true[:3])
        b2 = POINT_NAMES.index("B2")
        c1 = POINT_NAMES.index("C1")
        np.testing.assert_array_equal(faulty.true[3, b2], healthy.true[3, b2])
        np.testing.assert_allclose(faulty.true[3, c1] - healthy.true[3, c1], [0, 1.2, 0], atol=1e-12)
        self.assertGreater(np.linalg.norm(faulty.true[4, b2] - healthy.true[4, b2]), 0.01)
        self.assertGreater(np.linalg.norm(faulty.translations[3] - healthy.translations[3]), 0.01)

    def test_paired_runs_reuse_all_exogenous_noise(self):
        config = ModelConfig(0.02, 0.03)
        fault = Fault(2, "A2", (0.0, 0.0, 1.2))
        healthy = simulate(config, seed=55)
        faulty = simulate(config, seed=55, fault=fault)
        np.testing.assert_array_equal(faulty.true[:2], healthy.true[:2])
        np.testing.assert_allclose(faulty.observed - faulty.true, healthy.observed - healthy.true, atol=1e-12)
        for stage in range(1, 5):
            additions = []
            for run in (healthy, faulty):
                prediction = apply_transforms(run.true[stage - 1], run.rotations[stage - 1], run.translations[stage - 1])
                additions.append(run.true[stage] - prediction)
            expected = np.zeros((9, 3))
            if stage == fault.stage:
                expected[POINT_NAMES.index(fault.point)] = fault.offset
            np.testing.assert_allclose(additions[1] - additions[0], expected, atol=1e-12)

    def test_measurement_error_propagates_only_through_actions(self):
        run = simulate(ModelConfig(0, 0.1), seed=77)
        healthy = simulate(ModelConfig(0, 0), seed=77)
        np.testing.assert_array_equal(run.true[0], healthy.true[0])
        self.assertGreater(np.linalg.norm(run.observed[0] - run.true[0]), 0.1)
        # No process additions: every later true state is precisely the action
        # applied to preceding truth, never to preceding noisy observations.
        for stage in range(1, 5):
            expected = apply_transforms(run.true[stage - 1], run.rotations[stage - 1], run.translations[stage - 1])
            np.testing.assert_allclose(run.true[stage], expected, atol=1e-12)
        # Inactive B's true points do not inherit noise at front mating.
        np.testing.assert_array_equal(run.true[2, 3:6], run.true[1, 3:6])
        # Positioning still reacts to noise, so later truth differs from ideal.
        self.assertGreater(np.linalg.norm(run.true[1] - healthy.true[1]), 0.01)

    def test_coordinate_prediction_identifies_new_fault_not_propagated_effect(self):
        run = simulate(ModelConfig(0, 0), fault=Fault(3, "C1", (0.0, 1.2, 0.0)))
        residuals = np.stack([run.observed[k] - predict_next(run.observed[k - 1], k) for k in range(1, 5)])
        expected = np.zeros((4, 9, 3))
        expected[2, POINT_NAMES.index("C1")] = [0, 1.2, 0]
        np.testing.assert_allclose(residuals, expected, atol=1e-10)

    def test_bad_inputs_are_rejected(self):
        for kwargs in ({"process_sigma": -1}, {"measurement_sigma": float("nan")}, {"quality_limit": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        for args in ((0, "A1", (0, 0, 1)), (1, "Z1", (0, 0, 1)), (1, "A1", (0, 1))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Fault(*args)
        with self.assertRaises(ValueError):
            stage_transforms(np.zeros((8, 3)), 2)
        with self.assertRaises(ValueError):
            stage_transforms(np.full((9, 3), np.nan), 2)
        with self.assertRaises(ValueError):
            rigid_fit(np.zeros((3, 3)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            simulate(ModelConfig(), seed=-1)
        with self.assertRaises(ValueError):
            stage_transforms(nominal_trajectory()[0], 1.5)
        with self.assertRaises(ValueError):
            apply_transforms(nominal_trajectory()[0], np.zeros((3, 3, 3)), np.zeros((3, 3)))

    def test_spec_is_json_serializable_and_nominal_does_not_alias(self):
        json.dumps(model_spec(), ensure_ascii=False, allow_nan=False)
        first = nominal_trajectory()
        first[:] = 0
        self.assertGreater(np.linalg.norm(nominal_trajectory()), 1)


if __name__ == "__main__":
    unittest.main()
