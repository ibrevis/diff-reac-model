"""Recover Nagumo diffusion coefficients without parameter regularization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from . import params
except ImportError:
    import params  # type: ignore[no-redef]


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OBSERVATIONS = SCRIPT_DIR / "observations_coarse.npy"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inverse_output_no_reg"
HISTORY_COLUMNS = (
    "evaluation",
    "a1",
    "a2",
    "weighted_data_norm",
    "objective",
    "forward_solves",
    "cache_hit",
)
CONFIDENCE_Z_95 = 1.96
CHI_SQUARE_2_95 = 5.991464547107979


def load_observation_data(
    path: str | Path,
    *,
    expected_times: Sequence[float] = params.observation_times,
    expected_points: Sequence[Sequence[float]] = params.observation_points,
) -> np.ndarray:
    """Load and validate the known time-major observation layout."""
    observation_path = Path(path)
    if not observation_path.is_file():
        hint = ""
        if observation_path.name == "observation_coarse.npy":
            hint = f" Did you mean '{DEFAULT_OBSERVATIONS}'?"
        raise FileNotFoundError(f"Observation file not found: {observation_path}.{hint}")

    observed = np.load(observation_path, allow_pickle=False)
    expected_shape = (len(expected_times), len(expected_points))
    if observed.ndim != 2:
        raise ValueError(
            "Observations must be a 2D time-by-sensor array; "
            f"got shape {observed.shape}"
        )
    if observed.shape != expected_shape:
        raise ValueError(
            f"Observations have shape {observed.shape}; expected {expected_shape} "
            "with rows=time and columns=sensors"
        )
    if not np.issubdtype(observed.dtype, np.floating):
        raise TypeError(f"Observations must be floating point; got {observed.dtype}")
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observations contain non-finite values")
    return np.asarray(observed, dtype=np.float64)


def add_observation_noise(
    observed: np.ndarray,
    noise_std: float,
    noise_seed: int,
) -> np.ndarray:
    """Return observations with reproducible independent Gaussian noise."""
    _validate_noise_std(noise_std)

    values = np.asarray(observed, dtype=np.float64)
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=values.shape)
    return values + noise


def build_data_residual(
    predicted: np.ndarray,
    observed: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    """Return the flattened residual weighted by the known noise deviation."""
    _validate_noise_std(noise_std)
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)

    if predicted.shape != observed.shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} does not match observations "
            f"{observed.shape}"
        )
    if predicted.ndim != 2:
        raise ValueError("Predicted and observed arrays must be two-dimensional")
    if not np.all(np.isfinite(predicted)):
        raise FloatingPointError("Predicted observations contain non-finite values")
    if not np.all(np.isfinite(observed)):
        raise FloatingPointError("Observed values contain non-finite values")
    return (predicted - observed).ravel(order="C") / noise_std


def _validate_noise_std(noise_std: float) -> None:
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError(f"noise_std must be finite and strictly positive; got {noise_std}")


class ExactForwardCache:
    """Cache forward predictions using exact float tuple keys."""

    def __init__(self, forward: Callable[[float, float], np.ndarray]) -> None:
        self.forward = forward
        self._cache: dict[tuple[float, float], np.ndarray] = {}
        self.solve_count = 0

    def predict(self, theta: Sequence[float]) -> tuple[np.ndarray, bool]:
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"theta must contain two finite values; got {values}")
        if np.any(values <= 0.0):
            raise ValueError(f"Diffusion coefficients must be positive; got {values}")
        key = (float(values[0]), float(values[1]))
        if key in self._cache:
            return self._cache[key].copy(), True

        try:
            prediction = np.asarray(self.forward(*key), dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(
                f"Forward solve failed for a1={key[0]:.16g}, "
                f"a2={key[1]:.16g}: {exc}"
            ) from exc
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError(
                f"Forward solve returned non-finite data for theta={key}"
            )
        self._cache[key] = prediction.copy()
        self.solve_count += 1
        return prediction, False


@dataclass(frozen=True)
class FisherInformationResult:
    sensitivity_matrix: np.ndarray
    difference_steps: np.ndarray
    fisher_matrix: np.ndarray
    rank: int
    condition_number: float
    covariance_matrix: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals_95: np.ndarray


@dataclass(frozen=True)
class FisherEllipseGeometry:
    center: np.ndarray
    eigenvalues: np.ndarray
    semiaxis_lengths: np.ndarray
    angle_degrees: float


def compute_fisher_ellipse_geometry(
    fisher_matrix: np.ndarray,
    center: Sequence[float],
) -> FisherEllipseGeometry | None:
    """Return the 95% joint Fisher ellipse geometry, if it is identifiable."""
    matrix = np.asarray(fisher_matrix, dtype=np.float64)
    center_values = np.asarray(center, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"fisher_matrix must have shape (2, 2); got {matrix.shape}")
    if center_values.shape != (2,):
        raise ValueError(f"ellipse center must have shape (2,); got {center_values.shape}")
    if not np.all(np.isfinite(center_values)):
        raise ValueError("ellipse center must be finite")
    if not np.all(np.isfinite(matrix)):
        return None

    symmetric_matrix = 0.5 * (matrix + matrix.T)
    if np.linalg.matrix_rank(symmetric_matrix) < 2:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        return None

    semiaxis_lengths = np.sqrt(CHI_SQUARE_2_95 / eigenvalues)
    major_axis_vector = eigenvectors[:, 0]
    angle_degrees = float(
        np.degrees(np.arctan2(major_axis_vector[1], major_axis_vector[0]))
    )
    return FisherEllipseGeometry(
        center=center_values.copy(),
        eigenvalues=eigenvalues,
        semiaxis_lengths=semiaxis_lengths,
        angle_degrees=angle_degrees,
    )


def compute_fisher_information(
    cache: ExactForwardCache,
    theta: Sequence[float],
    noise_std: float,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    relative_step: float,
) -> FisherInformationResult:
    """Compute local Fisher information using bound-aware central differences."""
    _validate_noise_std(noise_std)
    values = np.asarray(theta, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if any(array.shape != (2,) for array in (values, lower, upper)):
        raise ValueError("theta and Fisher bounds must each contain two values")
    if not all(np.all(np.isfinite(array)) for array in (values, lower, upper)):
        raise ValueError("theta and Fisher bounds must be finite")
    if np.any(upper <= lower) or np.any(values < lower) or np.any(values > upper):
        raise ValueError("theta must lie within valid Fisher bounds")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("Fisher relative_step must be finite and positive")

    scales = np.maximum(1.0, np.abs(values))
    requested_steps = relative_step * scales
    distances_to_bounds = np.minimum(values - lower, upper - values)
    difference_steps = np.minimum(requested_steps, 0.5 * distances_to_bounds)
    minimum_steps = 64.0 * np.finfo(np.float64).eps * scales
    invalid = difference_steps <= minimum_steps
    if np.any(invalid):
        names = ", ".join(("a1", "a2")[index] for index in np.flatnonzero(invalid))
        raise RuntimeError(
            "Cannot form a numerically meaningful central Fisher difference for "
            f"{names}; the recovered value is too close to a bound"
        )

    sensitivity_columns: list[np.ndarray] = []
    prediction_shape: tuple[int, ...] | None = None
    for index, step in enumerate(difference_steps):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        prediction_plus, _ = cache.predict(plus)
        prediction_minus, _ = cache.predict(minus)
        if prediction_plus.shape != prediction_minus.shape:
            raise RuntimeError("Central Fisher predictions have inconsistent shapes")
        if prediction_shape is None:
            prediction_shape = prediction_plus.shape
        elif prediction_plus.shape != prediction_shape:
            raise RuntimeError("Fisher predictions changed shape between parameters")
        derivative = (prediction_plus - prediction_minus) / (2.0 * step)
        sensitivity_columns.append(derivative.ravel(order="C"))

    sensitivity_matrix = np.column_stack(sensitivity_columns)
    fisher_matrix = (sensitivity_matrix.T @ sensitivity_matrix) / noise_std**2
    rank = int(np.linalg.matrix_rank(fisher_matrix))
    condition_number = float(np.linalg.cond(fisher_matrix))

    covariance_matrix = np.full((2, 2), np.nan, dtype=np.float64)
    standard_errors = np.full(2, np.nan, dtype=np.float64)
    confidence_intervals_95 = np.full((2, 2), np.nan, dtype=np.float64)
    if rank == 2:
        covariance_matrix = np.linalg.inv(fisher_matrix)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance_matrix), 0.0))
        confidence_intervals_95[:, 0] = values - CONFIDENCE_Z_95 * standard_errors
        confidence_intervals_95[:, 1] = values + CONFIDENCE_Z_95 * standard_errors

    return FisherInformationResult(
        sensitivity_matrix=sensitivity_matrix,
        difference_steps=difference_steps,
        fisher_matrix=fisher_matrix,
        rank=rank,
        condition_number=condition_number,
        covariance_matrix=covariance_matrix,
        standard_errors=standard_errors,
        confidence_intervals_95=confidence_intervals_95,
    )


@dataclass
class ResidualEvaluator:
    observed: np.ndarray
    noise_std: float
    cache: ExactForwardCache
    print_diagnostics: bool = True

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=np.float64)
        _validate_noise_std(self.noise_std)
        self.history: list[list[float]] = []

    def __call__(self, theta: Sequence[float]) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        predicted, cache_hit = self.cache.predict(values)
        residual = build_data_residual(predicted, self.observed, self.noise_std)
        data_norm = float(np.linalg.norm(residual))
        objective = 0.5 * data_norm**2
        evaluation = len(self.history) + 1
        self.history.append(
            [
                float(evaluation),
                float(values[0]),
                float(values[1]),
                data_norm,
                objective,
                float(self.cache.solve_count),
                float(cache_hit),
            ]
        )
        if self.print_diagnostics:
            print(
                f"eval {evaluation}:\n"
                f"    a1 = {values[0]:.12g}\n"
                f"    a2 = {values[1]:.12g}\n"
                f"    ||r_data / sigma|| = {data_norm:.12e}\n"
                f"    J           = {objective:.12e}\n"
                f"    forward solves = {self.cache.solve_count}"
                + (" (cache hit)" if cache_hit else "")
            )
        return residual

    def history_array(self) -> np.ndarray:
        if not self.history:
            return np.empty((0, len(HISTORY_COLUMNS)), dtype=np.float64)
        return np.asarray(self.history, dtype=np.float64)


def _relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(right.ravel(order="C")))
    numerator = float(np.linalg.norm((left - right).ravel(order="C")))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else np.inf
    return numerator / denominator


def run_preflight_checks(
    model,
    observed: np.ndarray,
    observation_times: np.ndarray,
    initial_guess: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    true_theta: np.ndarray | None = None,
) -> dict[str, float]:
    """Run forward repeatability, sensitivity, and synthetic-data checks."""
    print("Running forward-model preflight checks...")
    first = model.solve(*initial_guess, observation_times)
    second = model.solve(*initial_guess, observation_times)
    if first.shape != observed.shape:
        raise RuntimeError(
            f"Preflight prediction shape {first.shape} != observations {observed.shape}"
        )
    if not np.all(np.isfinite(first)):
        raise FloatingPointError("Preflight prediction contains non-finite values")
    repeatability_error = float(np.max(np.abs(first - second)))
    if not np.allclose(first, second, rtol=1.0e-12, atol=1.0e-14):
        raise RuntimeError(
            "Forward repeatability check failed: maximum absolute difference "
            f"is {repeatability_error:.6e}"
        )
    if not model.last_initial_reset_verified:
        raise RuntimeError("Forward model did not verify its initial-state reset")

    relative_steps = (1.0e-2, 3.0e-3, 1.0e-3)
    derivatives: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for index in (0, 1):
        for relative_step in relative_steps:
            h = relative_step * max(1.0, abs(float(initial_guess[index])))
            if initial_guess[index] + h >= upper_bounds[index]:
                h = -h
            trial = initial_guess.copy()
            trial[index] += h
            if trial[index] <= 0.0:
                raise RuntimeError("Unable to form a positive preflight perturbation")
            perturbed = model.solve(*trial, observation_times)
            derivatives[index].append((perturbed - first) / h)

    sensitivity_a1 = float(np.linalg.norm(derivatives[0][-1]))
    sensitivity_a2 = float(np.linalg.norm(derivatives[1][-1]))
    threshold = 100.0 * np.finfo(np.float64).eps * max(1.0, np.linalg.norm(first))
    if sensitivity_a1 <= threshold or sensitivity_a2 <= threshold:
        raise RuntimeError(
            "One or both diffusion coefficients have no measurable observation "
            f"sensitivity: {sensitivity_a1=:.6e}, {sensitivity_a2=:.6e}"
        )
    sensitivity_difference = float(
        np.linalg.norm(derivatives[0][-1] - derivatives[1][-1])
    )
    if sensitivity_difference <= threshold:
        raise RuntimeError("The a1 and a2 observation sensitivities are indistinguishable")

    fd_change_a1 = _relative_difference(derivatives[0][-1], derivatives[0][-2])
    fd_change_a2 = _relative_difference(derivatives[1][-1], derivatives[1][-2])
    report = {
        "repeatability_max_abs": repeatability_error,
        "sensitivity_a1_norm": sensitivity_a1,
        "sensitivity_a2_norm": sensitivity_a2,
        "sensitivity_difference_norm": sensitivity_difference,
        "fd_relative_change_a1": fd_change_a1,
        "fd_relative_change_a2": fd_change_a2,
    }

    if true_theta is not None:
        truth_prediction = model.solve(*true_theta, observation_times)
        truth_relative_error = _relative_difference(truth_prediction, observed)
        report["truth_data_relative_error"] = truth_relative_error
        if truth_relative_error > 1.0e-8:
            raise RuntimeError(
                "The configured synthetic parameters do not reproduce the coarse "
                f"observations (relative error {truth_relative_error:.6e})"
            )

    print("Preflight checks passed:")
    for key, value in report.items():
        print(f"    {key}: {value:.12e}")
    return report


def plot_results(
    output_dir: Path,
    times: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    history: np.ndarray,
    true_theta: np.ndarray,
    confidence_intervals_95: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.arange(observed.shape[1]))

    fig, axis = plt.subplots(figsize=(10, 6))
    for index, (label, color) in enumerate(
        zip(params.observation_point_labels, colors)
    ):
        axis.plot(times, predicted[:, index], color=color, label=f"fitted {label}")
        axis.plot(
            times,
            observed[:, index],
            linestyle="none",
            marker=".",
            markersize=2.5,
            alpha=0.55,
            color=color,
            label=f"observed {label}",
        )
    axis.set(xlabel="Time", ylabel="u", title="Observed and recovered-model sensors")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "observations_vs_fit.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    residuals = predicted - observed
    for index, (label, color) in enumerate(
        zip(params.observation_point_labels, colors)
    ):
        axis.plot(times, residuals[:, index], color=color, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Time", ylabel="Predicted - observed", title="Sensor residuals")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    evaluations = history[:, 0]
    axes[0].semilogy(evaluations, np.maximum(history[:, 4], np.finfo(float).tiny))
    axes[0].set(ylabel="Objective", title="Optimization history")
    axes[0].grid(alpha=0.25)
    confidence_intervals_95 = np.asarray(
        confidence_intervals_95, dtype=np.float64
    )
    if confidence_intervals_95.shape != (2, 2):
        raise ValueError(
            "confidence_intervals_95 must have shape (2, 2); "
            f"got {confidence_intervals_95.shape}"
        )
    for index, color in enumerate(("C0", "C1")):
        interval = confidence_intervals_95[index]
        if np.all(np.isfinite(interval)):
            axes[1].axhspan(
                interval[0],
                interval[1],
                color=color,
                alpha=0.15,
                zorder=0,
                label=f"a{index + 1} 95% CI",
            )
    axes[1].plot(evaluations, history[:, 1], marker=".", color="C0", label="a1")
    axes[1].plot(evaluations, history[:, 2], marker=".", color="C1", label="a2")
    axes[1].axhline(
        true_theta[0], color="C0", linestyle="--", label="true a1"
    )
    axes[1].axhline(
        true_theta[1], color="C1", linestyle="--", label="true a2"
    )
    axes[1].set(xlabel="Residual evaluation", ylabel="Coefficient")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimization_history.png", dpi=180)
    plt.close(fig)


def plot_fisher_confidence_ellipse(
    output_dir: Path,
    estimated_theta: np.ndarray,
    true_theta: np.ndarray,
    fisher_matrix: np.ndarray,
) -> Path:
    """Plot the recovered parameters and their joint 95% Fisher ellipse."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse

    estimated_theta = np.asarray(estimated_theta, dtype=np.float64)
    true_theta = np.asarray(true_theta, dtype=np.float64)
    if true_theta.shape != (2,) or not np.all(np.isfinite(true_theta)):
        raise ValueError("true_theta must contain two finite values")
    geometry = compute_fisher_ellipse_geometry(fisher_matrix, estimated_theta)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 7))
    if geometry is not None:
        ellipse = Ellipse(
            xy=geometry.center,
            width=2.0 * geometry.semiaxis_lengths[0],
            height=2.0 * geometry.semiaxis_lengths[1],
            angle=geometry.angle_degrees,
            facecolor="C0",
            edgecolor="C0",
            alpha=0.2,
            linewidth=1.5,
            label="95% Fisher confidence ellipse",
        )
        axis.add_patch(ellipse)
    else:
        axis.text(
            0.5,
            0.05,
            "95% Fisher confidence ellipse unavailable",
            transform=axis.transAxes,
            horizontalalignment="center",
            color="C3",
        )

    axis.scatter(
        estimated_theta[0],
        estimated_theta[1],
        marker="X",
        s=90,
        color="C0",
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        label="estimated parameters",
    )
    axis.scatter(
        true_theta[0],
        true_theta[1],
        marker="*",
        s=140,
        color="C3",
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        label="true parameters",
    )
    axis.autoscale_view()
    axis.margins(0.15)
    axis.set(
        xlabel=r"$a_1$",
        ylabel=r"$a_2$",
        title="95% joint Fisher confidence ellipse",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output_path = output_dir / "fisher_confidence_ellipse.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_results(
    output_dir: Path,
    result,
    *,
    initial_guess: np.ndarray,
    noise_std: float,
    noise_seed: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    predicted: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    history: np.ndarray,
    forward_solve_count: int,
    preflight_report: dict[str, float],
    fisher_result: FisherInformationResult,
    true_theta: np.ndarray | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    weighted_data_residual = build_data_residual(predicted, observed, noise_std)
    data_residual = weighted_data_residual * noise_std
    result_path = output_dir / "inverse_result.npz"
    payload: dict[str, np.ndarray | float | int | bool | str] = {
        "a1": float(result.x[0]),
        "a2": float(result.x[1]),
        "initial_guess": initial_guess,
        "noise_std": float(noise_std),
        "noise_seed": int(noise_seed),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "predicted_observations": predicted,
        "observed_observations": observed,
        "observation_times": times,
        "observation_points": np.asarray(params.observation_points),
        "optimization_cost": float(result.cost),
        "optimization_status": int(result.status),
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
        "forward_solve_count": int(forward_solve_count),
        "final_data_residual_norm": float(np.linalg.norm(data_residual)),
        "final_weighted_data_residual_norm": float(
            np.linalg.norm(weighted_data_residual)
        ),
        "fisher_sensitivity_matrix": fisher_result.sensitivity_matrix,
        "fisher_difference_steps": fisher_result.difference_steps,
        "fisher_information_matrix": fisher_result.fisher_matrix,
        "fisher_rank": fisher_result.rank,
        "fisher_condition_number": fisher_result.condition_number,
        "parameter_covariance_matrix": fisher_result.covariance_matrix,
        "parameter_standard_errors": fisher_result.standard_errors,
        "parameter_confidence_intervals_95": fisher_result.confidence_intervals_95,
        "history": history,
        "history_columns": np.asarray(HISTORY_COLUMNS),
    }
    for key, value in preflight_report.items():
        payload[f"preflight_{key}"] = float(value)
    if true_theta is not None:
        payload["true_parameters"] = true_theta
        payload["relative_error_a1"] = abs(result.x[0] - true_theta[0]) / abs(
            true_theta[0]
        )
        payload["relative_error_a2"] = abs(result.x[1] - true_theta[1]) / abs(
            true_theta[1]
        )
    np.savez(result_path, **payload)
    return result_path


def run_inverse_problem(args: argparse.Namespace):
    try:
        from .forward_model import NagumoForwardModel
    except ImportError:
        from forward_model import NagumoForwardModel

    clean_observed = load_observation_data(args.observations)
    observed = add_observation_noise(
        clean_observed,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
    )
    times = np.asarray(params.observation_times, dtype=np.float64)
    initial_guess = np.asarray((args.a1_init, args.a2_init), dtype=np.float64)
    lower_bounds = np.asarray((args.a1_lower, args.a2_lower), dtype=np.float64)
    upper_bounds = np.asarray((args.a1_upper, args.a2_upper), dtype=np.float64)
    _validate_configuration(
        initial_guess,
        lower_bounds,
        upper_bounds,
        args.diff_step,
        args.noise_std,
        args.fisher_diff_step,
    )

    model = NagumoForwardModel(
        params.observation_points,
        verbose=args.forward_verbose,
    )
    true_theta = np.asarray((params.a1, params.a2), dtype=np.float64)
    preflight_report: dict[str, float] = {}
    if not args.skip_preflight or args.preflight_only:
        preflight_report = run_preflight_checks(
            model,
            clean_observed,
            times,
            initial_guess,
            upper_bounds,
            true_theta=true_theta,
        )
    if args.preflight_only:
        return None

    from scipy.optimize import least_squares

    cache = ExactForwardCache(lambda a1, a2: model.solve(a1, a2, times))
    evaluator = ResidualEvaluator(observed, args.noise_std, cache)
    result = least_squares(
        evaluator,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        jac="2-point",
        diff_step=args.diff_step,
        ftol=args.ftol,
        xtol=args.xtol,
        gtol=args.gtol,
        max_nfev=args.max_nfev,
        verbose=args.optimizer_verbose,
    )
    predicted, _ = cache.predict(result.x)
    weighted_data_residual = build_data_residual(
        predicted, observed, args.noise_std
    )
    data_residual = weighted_data_residual * args.noise_std
    fisher_result = compute_fisher_information(
        cache,
        result.x,
        args.noise_std,
        lower_bounds,
        upper_bounds,
        args.fisher_diff_step,
    )
    history = evaluator.history_array()
    output_dir = Path(args.output_dir).resolve()
    result_path = save_results(
        output_dir,
        result,
        initial_guess=initial_guess,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        predicted=predicted,
        observed=observed,
        times=times,
        history=history,
        forward_solve_count=cache.solve_count,
        preflight_report=preflight_report,
        fisher_result=fisher_result,
        true_theta=true_theta,
    )
    plot_results(
        output_dir,
        times,
        observed,
        predicted,
        history,
        true_theta,
        fisher_result.confidence_intervals_95,
    )
    plot_fisher_confidence_ellipse(
        output_dir,
        result.x,
        true_theta,
        fisher_result.fisher_matrix,
    )

    relative_error_a1 = abs(result.x[0] - true_theta[0]) / abs(true_theta[0])
    relative_error_a2 = abs(result.x[1] - true_theta[1]) / abs(true_theta[1])
    print("\nInverse-problem summary")
    print(f"Optimization converged: {result.success} ({result.message})")
    print(f"Recovered a1: {result.x[0]:.12g}")
    print(f"Recovered a2: {result.x[1]:.12g}")
    print(f"True parameters (validation only): {true_theta}")
    print(f"Relative error a1: {relative_error_a1:.12e}")
    print(f"Relative error a2: {relative_error_a2:.12e}")
    print(f"Initial guess: {initial_guess}")
    print(f"Observation noise standard deviation: {args.noise_std:.12g}")
    print(f"Observation noise seed: {args.noise_seed}")
    print(f"Fisher relative difference step: {args.fisher_diff_step:.12g}")
    print(f"Bounds: lower={lower_bounds}, upper={upper_bounds}")
    print(f"Number of forward solves: {cache.solve_count}")
    print(f"Final data residual norm: {np.linalg.norm(data_residual):.12e}")
    print(
        "Final weighted data residual norm: "
        f"{np.linalg.norm(weighted_data_residual):.12e}"
    )
    print(f"Final objective: {result.cost:.12e}")
    print("Fisher information matrix:")
    print(fisher_result.fisher_matrix)
    print(f"Fisher numerical rank: {fisher_result.rank}")
    print(f"Fisher condition number: {fisher_result.condition_number:.12e}")
    if fisher_result.rank < 2:
        print(
            "WARNING: Fisher information is rank deficient; parameter standard "
            "errors and 95% confidence intervals are unavailable."
        )
    else:
        for index, name in enumerate(("a1", "a2")):
            lower, upper = fisher_result.confidence_intervals_95[index]
            print(
                f"{name} standard error: {fisher_result.standard_errors[index]:.12e}"
            )
            print(f"{name} 95% confidence interval: [{lower:.12g}, {upper:.12g}]")
    print(f"Saved result: {result_path}")
    print(f"Saved plots: {output_dir}")
    return result


def _validate_configuration(
    initial_guess: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    diff_step: float,
    noise_std: float,
    fisher_diff_step: float,
) -> None:
    arrays = (initial_guess, lower_bounds, upper_bounds)
    if any(values.shape != (2,) or not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("All parameter vectors must contain exactly two finite values")
    if np.any(lower_bounds <= 0.0):
        raise ValueError("Lower diffusion bounds must be strictly positive")
    if np.any(upper_bounds <= lower_bounds):
        raise ValueError("Each upper bound must be greater than its lower bound")
    if np.any(initial_guess < lower_bounds) or np.any(initial_guess > upper_bounds):
        raise ValueError("Initial guess must lie within the supplied bounds")
    if not np.isfinite(diff_step) or diff_step <= 0.0:
        raise ValueError("diff_step must be finite and positive")
    _validate_noise_std(noise_std)
    if not np.isfinite(fisher_diff_step) or fisher_diff_step <= 0.0:
        raise ValueError("fisher_diff_step must be finite and positive")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover constant anisotropic diffusion coefficients in the Nagumo model "
            "without parameter regularization."
        )
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a1-init", type=float, default=1.0)
    parser.add_argument("--a2-init", type=float, default=1.0)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.005,
        help="standard deviation of additive Gaussian observation noise",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="random seed used to generate observation noise",
    )
    parser.add_argument("--a1-lower", type=float, default=1.0e-4)
    parser.add_argument("--a2-lower", type=float, default=1.0e-4)
    parser.add_argument("--a1-upper", type=float, default=2.0)
    parser.add_argument("--a2-upper", type=float, default=2.0)
    parser.add_argument("--diff-step", type=float, default=1.0e-3)
    parser.add_argument(
        "--fisher-diff-step",
        type=float,
        default=1.0e-3,
        help="relative central-difference step for Fisher sensitivities",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--xtol", type=float, default=1.0e-8)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--optimizer-verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--forward-verbose", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_inverse_problem(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
