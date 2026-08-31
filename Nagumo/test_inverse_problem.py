"""Dependency-light tests plus an opt-in FEniCSx repeatability test."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

try:
    from . import params
    from .inverse_problem import (
        ExactForwardCache,
        build_regularized_residual,
        load_observation_data,
    )
except ImportError:
    import params  # type: ignore[no-redef]
    from inverse_problem import (  # type: ignore[no-redef]
        ExactForwardCache,
        build_regularized_residual,
        load_observation_data,
    )


class InverseUtilitiesTest(unittest.TestCase):
    def test_observation_layout(self) -> None:
        path = Path(__file__).resolve().parent / "observations_coarse.npy"
        observations = load_observation_data(path)
        self.assertEqual(observations.shape, (params.num_steps, 5))
        self.assertEqual(observations.dtype, np.float64)
        self.assertTrue(np.isfinite(observations).all())
        self.assertAlmostEqual(params.observation_times[0], params.dt)
        self.assertAlmostEqual(params.observation_times[-1], params.T)

    def test_residual_order_and_regularization(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.5, 1.0], [4.0, 4.25]])
        residual, data, regularization = build_regularized_residual(
            predicted,
            observed,
            theta=(0.8, 0.6),
            theta_ref=(1.0, 1.0),
            lambda_reg=0.25,
        )
        np.testing.assert_allclose(data, [0.5, -1.0, 1.0, 0.25])
        np.testing.assert_allclose(regularization, [-0.1, -0.2])
        np.testing.assert_allclose(residual, [0.5, -1.0, 1.0, 0.25, -0.1, -0.2])
        expected_objective = 0.5 * np.dot(residual, residual)
        self.assertAlmostEqual(expected_objective, 1.18125)

    def test_exact_forward_cache(self) -> None:
        calls: list[tuple[float, float]] = []

        def forward(a1: float, a2: float) -> np.ndarray:
            calls.append((a1, a2))
            return np.array([[a1, a2]])

        cache = ExactForwardCache(forward)
        first, first_hit = cache.predict((0.8, 0.6))
        second, second_hit = cache.predict((0.8, 0.6))
        _, nearby_hit = cache.predict((np.nextafter(0.8, 1.0), 0.6))
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertFalse(nearby_hit)
        self.assertEqual(cache.solve_count, 2)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(first, second)


@unittest.skipUnless(
    os.environ.get("NAGUMO_RUN_FEM_TESTS") == "1",
    "Set NAGUMO_RUN_FEM_TESTS=1 in a working FEniCSx environment",
)
class ForwardIntegrationTest(unittest.TestCase):
    def test_repeatability_and_shape(self) -> None:
        try:
            from .forward_model import NagumoForwardModel
        except ImportError:
            from forward_model import NagumoForwardModel

        model = NagumoForwardModel(params.observation_points)
        first = model.solve(1.0, 1.0, params.observation_times)
        second = model.solve(1.0, 1.0, params.observation_times)
        self.assertEqual(first.shape, (params.num_steps, len(params.observation_points)))
        self.assertTrue(model.last_initial_reset_verified)
        np.testing.assert_allclose(first, second, rtol=1.0e-12, atol=1.0e-14)


if __name__ == "__main__":
    unittest.main()
