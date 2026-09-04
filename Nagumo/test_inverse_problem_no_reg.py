"""Dependency-light tests for the unregularized Nagumo inverse problem."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from .inverse_problem_no_reg import (
        DEFAULT_OUTPUT_DIR,
        CHI_SQUARE_2_95,
        ExactForwardCache,
        ResidualEvaluator,
        build_argument_parser,
        build_data_residual,
        compute_fisher_ellipse_geometry,
        compute_fisher_information,
        plot_fisher_confidence_ellipse,
        plot_results,
    )
except ImportError:
    from inverse_problem_no_reg import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_DIR,
        CHI_SQUARE_2_95,
        ExactForwardCache,
        ResidualEvaluator,
        build_argument_parser,
        build_data_residual,
        compute_fisher_ellipse_geometry,
        compute_fisher_information,
        plot_fisher_confidence_ellipse,
        plot_results,
    )


class UnregularizedInverseUtilitiesTest(unittest.TestCase):
    def _render_history_plot(
        self, confidence_intervals_95: np.ndarray
    ) -> list[tuple[float, float, dict[str, object]]]:
        from matplotlib.axes import Axes

        times = np.array([0.1, 0.2])
        observed = np.zeros((2, 5))
        predicted = np.ones_like(observed)
        history = np.zeros((2, 7))
        history[:, 0] = (1.0, 2.0)
        history[:, 1] = (1.0, 0.8)
        history[:, 2] = (1.0, 0.6)
        history[:, 4] = (10.0, 1.0)
        spans: list[tuple[float, float, dict[str, object]]] = []
        original_axhspan = Axes.axhspan

        def recording_axhspan(axis, ymin, ymax, **kwargs):
            spans.append((float(ymin), float(ymax), kwargs.copy()))
            return original_axhspan(axis, ymin, ymax, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch.object(Axes, "axhspan", new=recording_axhspan):
                plot_results(
                    output_dir,
                    times,
                    observed,
                    predicted,
                    history,
                    true_theta=np.array([0.8, 0.6]),
                    confidence_intervals_95=confidence_intervals_95,
                )
            self.assertTrue((output_dir / "optimization_history.png").is_file())
        return spans

    def test_data_residual_has_no_parameter_terms(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.5, 1.0], [4.0, 4.25]])

        residual = build_data_residual(predicted, observed, noise_std=0.5)

        np.testing.assert_allclose(residual, [1.0, -2.0, 2.0, 0.5])
        self.assertEqual(residual.size, observed.size)

    def test_evaluator_objective_is_data_misfit_only(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.5, 1.0], [4.0, 4.25]])
        cache = ExactForwardCache(lambda _a1, _a2: predicted)
        evaluator = ResidualEvaluator(
            observed, noise_std=0.5, cache=cache, print_diagnostics=False
        )

        residual = evaluator((0.8, 0.6))
        history = evaluator.history_array()

        self.assertAlmostEqual(history[0, 4], 0.5 * np.dot(residual, residual))
        self.assertEqual(history.shape, (1, 7))

    def test_invalid_noise_standard_deviation_is_rejected(self) -> None:
        values = np.ones((2, 2))
        for noise_std in (0.0, -0.1, np.nan, np.inf, -np.inf):
            with self.subTest(noise_std=noise_std):
                with self.assertRaises(ValueError):
                    build_data_residual(values, values, noise_std)

    def test_central_fisher_information_and_confidence_intervals(self) -> None:
        def forward(a1: float, a2: float) -> np.ndarray:
            return np.array([[a1 + 2.0 * a2], [3.0 * a1 - a2]])

        cache = ExactForwardCache(forward)
        theta = np.array([0.8, 0.6])
        result = compute_fisher_information(
            cache,
            theta,
            noise_std=0.5,
            lower_bounds=(0.1, 0.1),
            upper_bounds=(2.0, 2.0),
            relative_step=1.0e-4,
        )

        expected_sensitivities = np.array([[1.0, 2.0], [3.0, -1.0]])
        expected_fisher = np.array([[40.0, -4.0], [-4.0, 20.0]])
        expected_covariance = np.linalg.inv(expected_fisher)
        expected_errors = np.sqrt(np.diag(expected_covariance))
        expected_intervals = np.column_stack(
            (theta - 1.96 * expected_errors, theta + 1.96 * expected_errors)
        )
        np.testing.assert_allclose(result.sensitivity_matrix, expected_sensitivities)
        np.testing.assert_allclose(result.fisher_matrix, expected_fisher)
        np.testing.assert_allclose(result.covariance_matrix, expected_covariance)
        np.testing.assert_allclose(result.standard_errors, expected_errors)
        np.testing.assert_allclose(result.confidence_intervals_95, expected_intervals)
        self.assertEqual(result.rank, 2)
        self.assertEqual(cache.solve_count, 4)

    def test_fisher_ellipse_geometry_uses_inverse_eigenvalue_scaling(self) -> None:
        center = np.array([0.8, 0.6])
        geometry = compute_fisher_ellipse_geometry(
            np.diag([4.0, 1.0]), center
        )

        self.assertIsNotNone(geometry)
        assert geometry is not None
        np.testing.assert_array_equal(geometry.center, center)
        np.testing.assert_allclose(geometry.eigenvalues, [1.0, 4.0])
        np.testing.assert_allclose(
            geometry.semiaxis_lengths,
            [np.sqrt(CHI_SQUARE_2_95), np.sqrt(CHI_SQUARE_2_95 / 4.0)],
        )
        self.assertAlmostEqual(abs(geometry.angle_degrees) % 180.0, 90.0)

    def test_fisher_ellipse_plot_centers_on_estimate_and_marks_truth(self) -> None:
        from matplotlib.axes import Axes
        from matplotlib.patches import Ellipse

        estimated = np.array([0.76, 0.615])
        truth = np.array([0.8, 0.6])
        ellipse_centers: list[np.ndarray] = []
        scatter_points: list[tuple[float, float, str]] = []
        original_ellipse = Ellipse
        original_scatter = Axes.scatter

        def recording_ellipse(*args, **kwargs):
            ellipse_centers.append(np.asarray(kwargs["xy"], dtype=np.float64))
            return original_ellipse(*args, **kwargs)

        def recording_scatter(axis, x, y, *args, **kwargs):
            scatter_points.append((float(x), float(y), str(kwargs.get("label"))))
            return original_scatter(axis, x, y, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                patch("matplotlib.patches.Ellipse", new=recording_ellipse),
                patch.object(Axes, "scatter", new=recording_scatter),
            ):
                output_path = plot_fisher_confidence_ellipse(
                    Path(temporary_dir),
                    estimated,
                    truth,
                    np.diag([4.0, 1.0]),
                )
            self.assertTrue(output_path.is_file())

        self.assertEqual(len(ellipse_centers), 1)
        np.testing.assert_array_equal(ellipse_centers[0], estimated)
        self.assertEqual(
            scatter_points,
            [
                (estimated[0], estimated[1], "estimated parameters"),
                (truth[0], truth[1], "true parameters"),
            ],
        )

    def test_rank_deficient_fisher_plot_annotates_missing_ellipse(self) -> None:
        from matplotlib.axes import Axes
        from matplotlib.patches import Ellipse

        ellipse_calls: list[object] = []
        annotations: list[str] = []
        original_ellipse = Ellipse
        original_text = Axes.text

        def recording_ellipse(*args, **kwargs):
            ellipse_calls.append((args, kwargs))
            return original_ellipse(*args, **kwargs)

        def recording_text(axis, x, y, text, *args, **kwargs):
            annotations.append(str(text))
            return original_text(axis, x, y, text, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                patch("matplotlib.patches.Ellipse", new=recording_ellipse),
                patch.object(Axes, "text", new=recording_text),
            ):
                output_path = plot_fisher_confidence_ellipse(
                    Path(temporary_dir),
                    estimated_theta=np.array([0.8, 0.6]),
                    true_theta=np.array([0.8, 0.6]),
                    fisher_matrix=np.ones((2, 2)),
                )
            self.assertTrue(output_path.is_file())

        self.assertEqual(ellipse_calls, [])
        self.assertIn("95% Fisher confidence ellipse unavailable", annotations)

    def test_fisher_difference_step_shrinks_near_bounds(self) -> None:
        cache = ExactForwardCache(
            lambda a1, a2: np.array([[a1 + a2, a1 - a2]])
        )
        result = compute_fisher_information(
            cache,
            theta=(0.1002, 1.0),
            noise_std=0.005,
            lower_bounds=(0.1, 0.1),
            upper_bounds=(2.0, 2.0),
            relative_step=0.1,
        )

        self.assertAlmostEqual(result.difference_steps[0], 0.0001)
        self.assertAlmostEqual(result.difference_steps[1], 0.1)

    def test_rank_deficient_fisher_has_unavailable_intervals(self) -> None:
        cache = ExactForwardCache(lambda a1, a2: np.array([[a1 + a2]]))
        result = compute_fisher_information(
            cache,
            theta=(0.8, 0.6),
            noise_std=0.005,
            lower_bounds=(0.1, 0.1),
            upper_bounds=(2.0, 2.0),
            relative_step=1.0e-3,
        )

        self.assertEqual(result.rank, 1)
        self.assertTrue(np.isnan(result.covariance_matrix).all())
        self.assertTrue(np.isnan(result.standard_errors).all())
        self.assertTrue(np.isnan(result.confidence_intervals_95).all())

    def test_history_plot_includes_confidence_bands(self) -> None:
        intervals = np.array([[0.75, 0.85], [0.55, 0.65]])

        spans = self._render_history_plot(intervals)

        np.testing.assert_allclose(
            [(lower, upper) for lower, upper, _ in spans], intervals
        )
        self.assertEqual([options["color"] for _, _, options in spans], ["C0", "C1"])
        self.assertEqual(
            [options["label"] for _, _, options in spans],
            ["a1 95% CI", "a2 95% CI"],
        )
        self.assertEqual([options["alpha"] for _, _, options in spans], [0.15, 0.15])
        self.assertEqual([options["zorder"] for _, _, options in spans], [0, 0])

    def test_history_plot_omits_unavailable_confidence_bands(self) -> None:
        spans = self._render_history_plot(np.full((2, 2), np.nan))

        self.assertEqual(spans, [])

    def test_parser_omits_regularization_options(self) -> None:
        parser = build_argument_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--lambda-reg", option_strings)
        self.assertNotIn("--a1-ref", option_strings)
        self.assertNotIn("--a2-ref", option_strings)
        args = parser.parse_args([])
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(args.noise_std, 0.005)
        self.assertEqual(args.fisher_diff_step, 1.0e-3)


if __name__ == "__main__":
    unittest.main()
