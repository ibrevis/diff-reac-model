"""Recover the anisotropic Nagumo diffusion coefficients from sensor data."""

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inverse_output"
HISTORY_COLUMNS = (
    "evaluation",
    "a1",
    "a2",
    "data_norm",
    "regularization_norm",
    "total_residual_norm",
    "objective",
    "forward_solves",
    "cache_hit",
)


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


def build_regularized_residual(
    predicted: np.ndarray,
    observed: np.ndarray,
    theta: Sequence[float],
    theta_ref: Sequence[float],
    lambda_reg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, data, and zero-order Tikhonov residual vectors."""
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    theta_ref = np.asarray(theta_ref, dtype=np.float64)

    if predicted.shape != observed.shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} does not match observations "
            f"{observed.shape}"
        )
    if predicted.ndim != 2:
        raise ValueError("Predicted and observed arrays must be two-dimensional")
    if theta.shape != (2,) or theta_ref.shape != (2,):
        raise ValueError("theta and theta_ref must each contain exactly two values")
    if not np.all(np.isfinite(predicted)):
        raise FloatingPointError("Predicted observations contain non-finite values")
    if not np.all(np.isfinite(observed)):
        raise FloatingPointError("Observed values contain non-finite values")
    if not np.isfinite(lambda_reg) or lambda_reg < 0.0:
        raise ValueError(f"lambda_reg must be finite and non-negative; got {lambda_reg}")

    data_residual = (predicted - observed).ravel(order="C")
    regularization_residual = np.sqrt(lambda_reg) * (theta - theta_ref)
    residual = np.concatenate((data_residual, regularization_residual))
    return residual, data_residual, regularization_residual


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


@dataclass
class ResidualEvaluator:
    observed: np.ndarray
    theta_ref: np.ndarray
    lambda_reg: float
    cache: ExactForwardCache
    print_diagnostics: bool = True

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=np.float64)
        self.theta_ref = np.asarray(self.theta_ref, dtype=np.float64)
        self.history: list[list[float]] = []

    def __call__(self, theta: Sequence[float]) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        predicted, cache_hit = self.cache.predict(values)
        residual, data_residual, regularization_residual = build_regularized_residual(
            predicted,
            self.observed,
            values,
            self.theta_ref,
            self.lambda_reg,
        )
        data_norm = float(np.linalg.norm(data_residual))
        reg_norm = float(np.linalg.norm(regularization_residual))
        total_norm = float(np.linalg.norm(residual))
        objective = 0.5 * total_norm**2
        evaluation = len(self.history) + 1
        self.history.append(
            [
                float(evaluation),
                float(values[0]),
                float(values[1]),
                data_norm,
                reg_norm,
                total_norm,
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
                f"    ||r_data|| = {data_norm:.12e}\n"
                f"    ||r_reg||  = {reg_norm:.12e}\n"
                f"    ||r||       = {total_norm:.12e}\n"
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
    axes[0].semilogy(evaluations, np.maximum(history[:, 6], np.finfo(float).tiny))
    axes[0].set(ylabel="Objective", title="Optimization history")
    axes[0].grid(alpha=0.25)
    axes[1].plot(evaluations, history[:, 1], marker=".", label="a1")
    axes[1].plot(evaluations, history[:, 2], marker=".", label="a2")
    axes[1].set(xlabel="Residual evaluation", ylabel="Coefficient")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimization_history.png", dpi=180)
    plt.close(fig)


def save_results(
    output_dir: Path,
    result,
    *,
    initial_guess: np.ndarray,
    theta_ref: np.ndarray,
    lambda_reg: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    predicted: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    history: np.ndarray,
    forward_solve_count: int,
    preflight_report: dict[str, float],
    true_theta: np.ndarray | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, data_residual, regularization_residual = build_regularized_residual(
        predicted, observed, result.x, theta_ref, lambda_reg
    )
    result_path = output_dir / "inverse_result.npz"
    payload: dict[str, np.ndarray | float | int | bool | str] = {
        "a1": float(result.x[0]),
        "a2": float(result.x[1]),
        "initial_guess": initial_guess,
        "theta_ref": theta_ref,
        "lambda_reg": float(lambda_reg),
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
        "final_regularization_norm": float(np.linalg.norm(regularization_residual)),
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

    observed = load_observation_data(args.observations)
    times = np.asarray(params.observation_times, dtype=np.float64)
    initial_guess = np.asarray((args.a1_init, args.a2_init), dtype=np.float64)
    theta_ref = np.asarray((args.a1_ref, args.a2_ref), dtype=np.float64)
    lower_bounds = np.asarray((args.a1_lower, args.a2_lower), dtype=np.float64)
    upper_bounds = np.asarray((args.a1_upper, args.a2_upper), dtype=np.float64)
    _validate_configuration(
        initial_guess, theta_ref, args.lambda_reg, lower_bounds, upper_bounds, args.diff_step
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
            observed,
            times,
            initial_guess,
            upper_bounds,
            true_theta=true_theta,
        )
    if args.preflight_only:
        return None

    from scipy.optimize import least_squares

    cache = ExactForwardCache(lambda a1, a2: model.solve(a1, a2, times))
    evaluator = ResidualEvaluator(observed, theta_ref, args.lambda_reg, cache)
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
    residual, data_residual, regularization_residual = build_regularized_residual(
        predicted, observed, result.x, theta_ref, args.lambda_reg
    )
    history = evaluator.history_array()
    output_dir = Path(args.output_dir).resolve()
    result_path = save_results(
        output_dir,
        result,
        initial_guess=initial_guess,
        theta_ref=theta_ref,
        lambda_reg=args.lambda_reg,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        predicted=predicted,
        observed=observed,
        times=times,
        history=history,
        forward_solve_count=cache.solve_count,
        preflight_report=preflight_report,
        true_theta=true_theta,
    )
    plot_results(output_dir, times, observed, predicted, history)

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
    print(f"Reference parameters: {theta_ref}")
    print(f"Regularization lambda: {args.lambda_reg:.12g}")
    print(f"Bounds: lower={lower_bounds}, upper={upper_bounds}")
    print(f"Number of forward solves: {cache.solve_count}")
    print(f"Final data residual norm: {np.linalg.norm(data_residual):.12e}")
    print(f"Final regularization norm: {np.linalg.norm(regularization_residual):.12e}")
    print(f"Final total residual norm: {np.linalg.norm(residual):.12e}")
    print(f"Final objective: {result.cost:.12e}")
    print(f"Saved result: {result_path}")
    print(f"Saved plots: {output_dir}")
    return result


def _validate_configuration(
    initial_guess: np.ndarray,
    theta_ref: np.ndarray,
    lambda_reg: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    diff_step: float,
) -> None:
    arrays = (initial_guess, theta_ref, lower_bounds, upper_bounds)
    if any(values.shape != (2,) or not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("All parameter vectors must contain exactly two finite values")
    if np.any(lower_bounds <= 0.0):
        raise ValueError("Lower diffusion bounds must be strictly positive")
    if np.any(upper_bounds <= lower_bounds):
        raise ValueError("Each upper bound must be greater than its lower bound")
    if np.any(initial_guess < lower_bounds) or np.any(initial_guess > upper_bounds):
        raise ValueError("Initial guess must lie within the supplied bounds")
    if np.any(theta_ref <= 0.0):
        raise ValueError("Reference diffusion coefficients must be positive")
    if not np.isfinite(lambda_reg) or lambda_reg < 0.0:
        raise ValueError("lambda_reg must be finite and non-negative")
    if not np.isfinite(diff_step) or diff_step <= 0.0:
        raise ValueError("diff_step must be finite and positive")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover constant anisotropic diffusion coefficients in the Nagumo model."
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a1-init", type=float, default=1.0)
    parser.add_argument("--a2-init", type=float, default=1.0)
    parser.add_argument("--a1-ref", type=float, default=1.0)
    parser.add_argument("--a2-ref", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0e-4)
    parser.add_argument("--a1-lower", type=float, default=1.0e-4)
    parser.add_argument("--a2-lower", type=float, default=1.0e-4)
    parser.add_argument("--a1-upper", type=float, default=2.0)
    parser.add_argument("--a2-upper", type=float, default=2.0)
    parser.add_argument("--diff-step", type=float, default=1.0e-3)
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
