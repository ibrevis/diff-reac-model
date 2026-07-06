#!/usr/bin/env python3
"""Compare PINN and FEM solution archives on their common masked grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_KEYS = ("x", "y", "mask", "times", "u")


def load_solution(path: Path) -> dict[str, np.ndarray]:
    """Load and validate one solution archive."""
    if not path.is_file():
        raise ValueError(f"solution archive does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in EXPECTED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
        solution = {key: np.asarray(archive[key]).copy() for key in EXPECTED_KEYS}

    x, y = solution["x"], solution["y"]
    mask, times, u = solution["mask"], solution["times"], solution["u"]
    if x.ndim != 1 or y.ndim != 1 or times.ndim != 1:
        raise ValueError(f"{path}: x, y, and times must be one-dimensional")
    if mask.dtype != np.bool_ or mask.shape != (len(y), len(x)):
        raise ValueError(f"{path}: mask must be boolean with shape (len(y), len(x))")
    expected_u_shape = (len(times), len(y), len(x))
    if u.shape != expected_u_shape:
        raise ValueError(f"{path}: u has shape {u.shape}, expected {expected_u_shape}")
    if not all(np.all(np.isfinite(array)) for array in (x, y, times)):
        raise ValueError(f"{path}: coordinates and times must be finite")
    if not np.all(np.isfinite(u[:, mask])):
        raise ValueError(f"{path}: u contains a non-finite value inside the disk")
    if not np.all(np.isnan(u[:, ~mask])):
        raise ValueError(f"{path}: u must contain NaN at every point outside the disk")
    return solution


def require_matching_grids(
    pinn: dict[str, np.ndarray], fem: dict[str, np.ndarray]
) -> None:
    """Require both archives to describe exactly the same samples."""
    for key in ("x", "y", "mask", "times"):
        if not np.array_equal(pinn[key], fem[key]):
            raise ValueError(f"PINN and FEM archives have different {key!r} arrays")
    if pinn["u"].shape != fem["u"].shape:
        raise ValueError(
            f"PINN and FEM fields have different shapes: "
            f"{pinn['u'].shape} and {fem['u'].shape}"
        )


def compute_metrics(
    pinn: dict[str, np.ndarray], fem: dict[str, np.ndarray]
) -> list[dict[str, float]]:
    """Compute discrete errors over the masked disk at each time."""
    mask = fem["mask"]
    metrics: list[dict[str, float]] = []
    for time, pinn_field, fem_field in zip(
        fem["times"], pinn["u"], fem["u"], strict=True
    ):
        difference = pinn_field[mask] - fem_field[mask]
        difference_norm = float(np.linalg.norm(difference))
        fem_norm = float(np.linalg.norm(fem_field[mask]))
        relative_l2 = difference_norm / fem_norm if fem_norm > 0.0 else (
            0.0 if difference_norm == 0.0 else float("inf")
        )
        metrics.append(
            {
                "time": float(time),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "relative_l2": relative_l2,
                "max_abs_error": float(np.max(np.abs(difference))),
            }
        )
    return metrics


def write_metrics(path: Path, metrics: list[dict[str, float]]) -> None:
    """Write metrics to CSV and print the same table."""
    fieldnames = ("time", "rmse", "relative_l2", "max_abs_error")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"{'time':>8} {'RMSE':>14} {'relative L2':>14} {'max abs error':>14}")
    for row in metrics:
        print(
            f"{row['time']:8.3f} {row['rmse']:14.6e} "
            f"{row['relative_l2']:14.6e} {row['max_abs_error']:14.6e}"
        )


def add_disk_outline(axis: plt.Axes) -> None:
    axis.add_patch(
        plt.Circle((0.0, 0.0), 1.0, fill=False, color="black", linewidth=0.7)
    )
    axis.set(aspect="equal", xlabel="x", ylabel="y")


def plot_comparison(
    path: Path, pinn: dict[str, np.ndarray], fem: dict[str, np.ndarray]
) -> None:
    """Plot PINN, FEM, and absolute-error fields at every snapshot."""
    x_grid, y_grid = np.meshgrid(fem["x"], fem["y"], indexing="xy")
    mask = fem["mask"]
    rows = len(fem["times"])
    figure, axes = plt.subplots(rows, 3, figsize=(14.0, 4.0 * rows), squeeze=False)

    for row, (time, pinn_field, fem_field) in enumerate(
        zip(fem["times"], pinn["u"], fem["u"], strict=True)
    ):
        combined = np.concatenate((pinn_field[mask], fem_field[mask]))
        value_min, value_max = float(np.min(combined)), float(np.max(combined))
        if value_min == value_max:
            padding = max(abs(value_min), 1.0) * 1.0e-12
            value_min -= padding
            value_max += padding

        error = np.full(mask.shape, np.nan, dtype=np.float64)
        error[mask] = np.abs(pinn_field[mask] - fem_field[mask])

        pinn_image = axes[row, 0].pcolormesh(
            x_grid, y_grid, pinn_field, shading="auto", cmap="viridis",
            vmin=value_min, vmax=value_max,
        )
        fem_image = axes[row, 1].pcolormesh(
            x_grid, y_grid, fem_field, shading="auto", cmap="viridis",
            vmin=value_min, vmax=value_max,
        )
        error_image = axes[row, 2].pcolormesh(
            x_grid, y_grid, error, shading="auto", cmap="magma", vmin=0.0
        )

        axes[row, 0].set_title(f"PINN: t = {time:g}")
        axes[row, 1].set_title(f"FEM: t = {time:g}")
        axes[row, 2].set_title(f"Absolute error: t = {time:g}")
        for axis in axes[row]:
            add_disk_outline(axis)
        figure.colorbar(pinn_image, ax=axes[row, 0], label="u")
        figure.colorbar(fem_image, ax=axes[row, 1], label="u")
        figure.colorbar(error_image, ax=axes[row, 2], label=r"$|u_{PINN}-u_{FEM}|$")

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PINN and FEM solution archives.")
    parser.add_argument(
        "--pinn", type=Path,
        default=Path("pinn_monodomain_results/pinn_solution.npz"),
    )
    parser.add_argument("--fem", type=Path, default=Path("fem_solution.npz"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("comparison_results")
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        pinn = load_solution(args.pinn)
        fem = load_solution(args.fem)
        require_matching_grids(pinn, fem)
        metrics = compute_metrics(pinn, fem)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_metrics(args.output_dir / "comparison_metrics.csv", metrics)
        plot_comparison(
            args.output_dir / "comparison_snapshots.png", pinn, fem
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"comparison outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
