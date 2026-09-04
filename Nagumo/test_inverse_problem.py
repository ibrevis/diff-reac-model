"""Dependency-light tests plus an opt-in FEniCSx repeatability test."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import params
    from .inverse_problem import (
        ExactForwardCache,
        add_observation_noise,
        build_regularized_residual,
        load_observation_data,
        plot_results,
    )
except ImportError:
    import params  # type: ignore[no-redef]
    from inverse_problem import (  # type: ignore[no-redef]
        ExactForwardCache,
        add_observation_noise,
        build_regularized_residual,
        load_observation_data,
        plot_results,
    )


class InverseUtilitiesTest(unittest.TestCase):
    def test_zero_observation_noise_is_unchanged(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        noisy = add_observation_noise(observed, noise_std=0.0, noise_seed=42)
        np.testing.assert_array_equal(noisy, observed)

    def test_observation_noise_is_seeded_and_does_not_modify_input(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = observed.copy()
        first = add_observation_noise(observed, noise_std=0.1, noise_seed=42)
        second = add_observation_noise(observed, noise_std=0.1, noise_seed=42)
        different = add_observation_noise(observed, noise_std=0.1, noise_seed=43)

        np.testing.assert_array_equal(observed, original)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))

    def test_observation_noise_rejects_invalid_standard_deviation(self) -> None:
        observed = np.ones((2, 2))
        for noise_std in (-0.1, np.nan, np.inf, -np.inf):
            with self.subTest(noise_std=noise_std):
                with self.assertRaises(ValueError):
                    add_observation_noise(observed, noise_std, noise_seed=42)

    def test_observation_layout(self) -> None:
        path = Path(__file__).resolve().parent / "observations_coarse.npy"
        observations = load_observation_data(path)
        self.assertEqual(observations.shape, (params.num_steps, 5))
        self.assertEqual(observations.dtype, np.float64)
        self.assertTrue(np.isfinite(observations).all())
        self.assertAlmostEqual(params.observation_times[0], params.dt)
        self.assertAlmostEqual(params.observation_times[-1], params.T)

    def test_optimization_history_plots_true_coefficients(self) -> None:
        from matplotlib.axes import Axes

        times = np.array([0.1, 0.2])
        observed = np.zeros((2, len(params.observation_points)))
        predicted = np.ones_like(observed)
        history = np.zeros((2, 9))
        history[:, 0] = (1.0, 2.0)
        history[:, 1] = (1.0, 0.8)
        history[:, 2] = (1.0, 0.6)
        history[:, 6] = (1.0, 0.5)
        true_theta = np.array([params.a1, params.a2])
        horizontal_lines: list[tuple[float, dict[str, object]]] = []
        original_axhline = Axes.axhline

        def recording_axhline(axis, y=0, xmin=0, xmax=1, **kwargs):
            horizontal_lines.append((float(y), kwargs.copy()))
            return original_axhline(axis, y, xmin, xmax, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch.object(Axes, "axhline", new=recording_axhline):
                plot_results(
                    output_dir,
                    times,
                    observed,
                    predicted,
                    history,
                    true_theta,
                )
            self.assertTrue((output_dir / "optimization_history.png").is_file())

        true_lines = [
            (value, options)
            for value, options in horizontal_lines
            if options.get("linestyle") == "--"
        ]
        self.assertEqual([value for value, _ in true_lines], [params.a1, params.a2])
        self.assertEqual([options["color"] for _, options in true_lines], ["C0", "C1"])
        self.assertEqual(
            [options["label"] for _, options in true_lines],
            ["true a1", "true a2"],
        )

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
