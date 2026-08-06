"""Train a parametric PINN for the forced monodomain problem.

The network approximates ``u(x, y, t, a1, a2)`` on the perforated unit disk.
It combines the parametric model in ``PINN-code-parametric.py`` with the
sampling, validation, and optimization strategy from
``PINN-code-new-simple_tuned.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Training and checkpointing do not depend on plotting.
    plt = None  # type: ignore[assignment]

from params import (
    A_stim,
    T,
    a1 as nominal_a1,
    a2 as nominal_a2,
    hole_r,
    hole_x,
    hole_y,
    r_value,
    s_stim,
    snapshot_times,
    source_amplitude,
    source_period,
    source_phase,
    source_spatial_width,
    source_time_width,
    source_x,
    source_y,
    x0,
    y0,
)


A_MIN = 0.1
A_MAX = 0.9


@dataclass(frozen=True)
class ModelSpec:
    hidden_layers: int = 5
    hidden_width: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12_000
    interior_batch: int = 8_192
    boundary_batch: int = 512
    validation_interior: int = 4_096
    validation_boundary: int = 512
    boundary_weight: float = 1.0
    grad_clip: float = 5.0
    clip_warmup_epochs: int = 500
    validation_every: int = 100
    lr_patience_checks: int = 10
    early_stop_checks: int = 25
    validation_min_delta: float = 1.0e-4
    log_every: int = 100
    seed: int = 42


class MonodomainPINN(nn.Module):
    """MLP with a hard transform enforcing the initial condition."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 5
        for _ in range(spec.hidden_layers):
            linear = nn.Linear(input_width, spec.hidden_width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            input_width = spec.hidden_width

        output = nn.Linear(input_width, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        x = points[:, 0:1]
        y = points[:, 1:2]
        t = points[:, 2:3]
        a1_values = points[:, 3:4]
        a2_values = points[:, 4:5]

        normalized = torch.cat(
            (
                x,
                y,
                2.0 * t / T - 1.0,
                2.0 * (a1_values - A_MIN) / (A_MAX - A_MIN) - 1.0,
                2.0 * (a2_values - A_MIN) / (A_MAX - A_MIN) - 1.0,
            ),
            dim=1,
        )
        raw = self.network(normalized)

        reaction_decay = torch.exp(-r_value * t)
        return reaction_decay * initial_condition(x, y) + (1.0 - reaction_decay) * raw


def initial_condition(x: Tensor, y: Tensor) -> Tensor:
    radius_sq = (x - x0) ** 2 + (y - y0) ** 2
    return A_stim * torch.exp(-radius_sq / (2.0 * s_stim**2))


def source_spatial_profile(x: Tensor, y: Tensor) -> Tensor:
    radius_sq = (x - source_x) ** 2 + (y - source_y) ** 2
    return source_amplitude * torch.exp(
        -radius_sq / (2.0 * source_spatial_width**2)
    )


def source_time_profile(t: Tensor) -> Tensor:
    phase_angle = math.pi * (t - source_phase) / source_period
    scaled_width = math.pi * source_time_width / source_period
    return torch.exp(-torch.sin(phase_angle).square() / (2.0 * scaled_width**2))


def source_term(x: Tensor, y: Tensor, t: Tensor) -> Tensor:
    return source_spatial_profile(x, y) * source_time_profile(t)


def _inside_domain(xy: Tensor) -> Tensor:
    inside_outer = xy[:, 0].square() + xy[:, 1].square() <= 1.0
    outside_hole = (
        (xy[:, 0] - hole_x).square() + (xy[:, 1] - hole_y).square()
        >= hole_r**2
    )
    return inside_outer & outside_hole


def sample_uniform_xy(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sample uniformly by area in the perforated unit disk."""
    accepted: list[Tensor] = []
    remaining = count
    while remaining:
        candidate_count = max(2 * remaining, 16)
        radius = torch.sqrt(
            torch.rand((candidate_count, 1), device=device, dtype=dtype)
        )
        angle = 2.0 * math.pi * torch.rand(
            (candidate_count, 1), device=device, dtype=dtype
        )
        candidates = torch.cat(
            (radius * torch.cos(angle), radius * torch.sin(angle)), dim=1
        )
        batch = candidates[_inside_domain(candidates)][:remaining]
        accepted.append(batch)
        remaining -= len(batch)
    return torch.cat(accepted, dim=0)


def sample_source_xy(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sample around the source and reject points outside the domain."""
    accepted: list[Tensor] = []
    remaining = count
    center = torch.tensor([[source_x, source_y]], device=device, dtype=dtype)
    while remaining:
        candidate_count = max(2 * remaining, 16)
        candidates = center + source_spatial_width * torch.randn(
            (candidate_count, 2), device=device, dtype=dtype
        )
        batch = candidates[_inside_domain(candidates)][:remaining]
        accepted.append(batch)
        remaining -= len(batch)
    return torch.cat(accepted, dim=0)


def sample_uniform_times(
    count: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    return T * torch.rand((count, 1), device=device, dtype=dtype)


def pulse_centers(device: torch.device, dtype: torch.dtype) -> Tensor:
    first = math.ceil((0.0 - source_phase) / source_period)
    last = math.floor((T - source_phase) / source_period)
    indices = torch.arange(first, last + 1, device=device, dtype=dtype)
    return source_phase + source_period * indices


def sample_pulse_times(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sample truncated Gaussians around every source pulse in [0, T]."""
    centers = pulse_centers(device, dtype)
    accepted: list[Tensor] = []
    remaining = count
    while remaining:
        candidate_count = max(2 * remaining, 16)
        choices = torch.randint(len(centers), (candidate_count,), device=device)
        candidates = centers[choices, None] + source_time_width * torch.randn(
            (candidate_count, 1), device=device, dtype=dtype
        )
        in_interval = (candidates[:, 0] >= 0.0) & (candidates[:, 0] <= T)
        batch = candidates[in_interval][:remaining]
        accepted.append(batch)
        remaining -= len(batch)
    return torch.cat(accepted, dim=0)


def sample_stratified_diffusivities(
    count: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Latin-hypercube sample independent a1 and a2 marginals."""
    if count <= 0:
        raise ValueError("count must be positive")
    strata = torch.arange(count, device=device, dtype=dtype)[:, None]
    offsets = torch.rand((count, 2), device=device, dtype=dtype)
    unit_samples = (strata + offsets) / count
    unit_samples[:, 0] = unit_samples[torch.randperm(count, device=device), 0]
    unit_samples[:, 1] = unit_samples[torch.randperm(count, device=device), 1]
    return A_MIN + (A_MAX - A_MIN) * unit_samples


def _append_diffusivities(
    points_xyt: Tensor, device: torch.device, dtype: torch.dtype
) -> Tensor:
    diffusivities = sample_stratified_diffusivities(len(points_xyt), device, dtype)
    return torch.cat((points_xyt, diffusivities), dim=1)


def sample_interior_components(
    count: int, device: torch.device, dtype: torch.dtype
) -> dict[str, Tensor]:
    """Return the tuned 50/25/25 uniform, source, and pulse mixture."""
    if count < 4:
        raise ValueError("interior count must be at least 4")
    uniform_count = count // 2
    source_count = count // 4
    pulse_count = count - uniform_count - source_count

    uniform = torch.cat(
        (
            sample_uniform_xy(uniform_count, device, dtype),
            sample_uniform_times(uniform_count, device, dtype),
        ),
        dim=1,
    )
    source = torch.cat(
        (
            sample_source_xy(source_count, device, dtype),
            sample_uniform_times(source_count, device, dtype),
        ),
        dim=1,
    )
    pulse = torch.cat(
        (
            sample_source_xy(pulse_count, device, dtype),
            sample_pulse_times(pulse_count, device, dtype),
        ),
        dim=1,
    )
    return {
        "uniform": _append_diffusivities(uniform, device, dtype),
        "source": _append_diffusivities(source, device, dtype),
        "pulse": _append_diffusivities(pulse, device, dtype),
    }


def sample_interior(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.cat(tuple(sample_interior_components(count, device, dtype).values()))


def sample_boundary(
    count: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Sample both circles by perimeter and return points plus outward normals."""
    if count < 2:
        raise ValueError("boundary count must be at least 2")
    inner_count = int(round(count * hole_r / (1.0 + hole_r)))
    inner_count = min(max(inner_count, 1), count - 1)
    outer_count = count - inner_count

    outer_angle = 2.0 * math.pi * torch.rand(
        (outer_count, 1), device=device, dtype=dtype
    )
    outer_direction = torch.cat(
        (torch.cos(outer_angle), torch.sin(outer_angle)), dim=1
    )
    outer_xy = outer_direction
    outer_normals = outer_direction

    inner_angle = 2.0 * math.pi * torch.rand(
        (inner_count, 1), device=device, dtype=dtype
    )
    inner_direction = torch.cat(
        (torch.cos(inner_angle), torch.sin(inner_angle)), dim=1
    )
    hole_center = torch.tensor([[hole_x, hole_y]], device=device, dtype=dtype)
    inner_xy = hole_center + hole_r * inner_direction
    inner_normals = -inner_direction

    xy = torch.cat((outer_xy, inner_xy), dim=0)
    normals = torch.cat((outer_normals, inner_normals), dim=0)
    points_xyt = torch.cat(
        (xy, sample_uniform_times(count, device, dtype)), dim=1
    )
    points = _append_diffusivities(points_xyt, device, dtype)
    permutation = torch.randperm(count, device=device)
    return points[permutation], normals[permutation]


def gradient(value: Tensor, points: Tensor, create_graph: bool = True) -> Tensor:
    return torch.autograd.grad(
        value,
        points,
        grad_outputs=torch.ones_like(value),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def pde_residual(model: MonodomainPINN, points: Tensor) -> Tensor:
    points = points.detach().requires_grad_(True)
    u = model(points)
    grad_u = gradient(u, points)
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]
    a1_values = points[:, 3:4]
    a2_values = points[:, 4:5]

    div_flux = (
        gradient(a1_values * u_x, points)[:, 0:1]
        + gradient(a2_values * u_y, points)[:, 1:2]
    )
    forcing = source_term(points[:, 0:1], points[:, 1:2], points[:, 2:3])
    return u_t - div_flux + r_value * u - forcing


def boundary_flux(
    model: MonodomainPINN, points: Tensor, normals: Tensor
) -> Tensor:
    points = points.detach().requires_grad_(True)
    grad_u = gradient(model(points), points)
    return (
        points[:, 3:4] * grad_u[:, 0:1] * normals[:, 0:1]
        + points[:, 4:5] * grad_u[:, 1:2] * normals[:, 1:2]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.float32 if device.type == "mps" else torch.float64
    dtype = torch.float32 if requested == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; use --dtype float32")
    return dtype


def _validate_latin_hypercube(samples: Tensor) -> None:
    count = len(samples)
    assert samples.shape == (count, 2)
    assert torch.all(samples >= A_MIN) and torch.all(samples <= A_MAX)
    unit_samples = (samples - A_MIN) / (A_MAX - A_MIN)
    bins = torch.clamp((unit_samples * count).floor().long(), max=count - 1)
    expected = torch.ones(count, device=samples.device, dtype=torch.long)
    for column in range(2):
        assert torch.equal(torch.bincount(bins[:, column], minlength=count), expected)


def validate_setup(
    model: MonodomainPINN,
    device: torch.device,
    dtype: torch.dtype,
    count: int = 120,
) -> float:
    """Validate geometry, sampling strata, source, residuals, and autodiff."""
    components = sample_interior_components(count, device, dtype)
    interior = torch.cat(tuple(components.values()))
    boundary, normals = sample_boundary(count, device, dtype)
    assert {name: len(points) for name, points in components.items()} == {
        "uniform": count // 2,
        "source": count // 4,
        "pulse": count - count // 2 - count // 4,
    }
    assert interior.shape == (count, 5)
    assert boundary.shape == (count, 5) and normals.shape == (count, 2)
    _validate_latin_hypercube(
        sample_stratified_diffusivities(64, device, dtype)
    )
    for points in (*components.values(), boundary):
        assert torch.all(points[:, 3:5] >= A_MIN)
        assert torch.all(points[:, 3:5] <= A_MAX)

    tolerance = 100.0 * torch.finfo(dtype).eps
    outer_radius_sq = interior[:, 0].square() + interior[:, 1].square()
    hole_radius_sq = (
        (interior[:, 0] - hole_x).square()
        + (interior[:, 1] - hole_y).square()
    )
    assert torch.all(outer_radius_sq <= 1.0 + tolerance)
    assert torch.all(hole_radius_sq >= hole_r**2 - tolerance)

    boundary_outer_radius = torch.linalg.vector_norm(boundary[:, :2], dim=1)
    hole_center = torch.tensor([hole_x, hole_y], device=device, dtype=dtype)
    boundary_hole_radius = torch.linalg.vector_norm(
        boundary[:, :2] - hole_center, dim=1
    )
    on_outer = torch.isclose(
        boundary_outer_radius,
        torch.ones_like(boundary_outer_radius),
        atol=tolerance,
        rtol=tolerance,
    )
    on_hole = torch.isclose(
        boundary_hole_radius,
        torch.full_like(boundary_hole_radius, hole_r),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.all(on_outer | on_hole) and torch.any(on_outer) and torch.any(on_hole)
    assert torch.allclose(
        torch.linalg.vector_norm(normals, dim=1),
        torch.ones(count, device=device, dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    expected_normals = torch.empty_like(normals)
    expected_normals[on_outer] = boundary[on_outer, :2]
    expected_normals[on_hole] = (hole_center - boundary[on_hole, :2]) / hole_r
    assert torch.allclose(
        normals, expected_normals, atol=tolerance, rtol=tolerance
    )

    centers = pulse_centers(device, dtype)
    pulse_samples = sample_pulse_times(6_000, device, dtype)
    nearest_centers = (pulse_samples - centers[None, :]).abs().argmin(dim=1)
    assert torch.all(torch.bincount(nearest_centers, minlength=len(centers)) > 0)

    source_peak = source_term(
        torch.full((1, 1), source_x, device=device, dtype=dtype),
        torch.full((1, 1), source_y, device=device, dtype=dtype),
        torch.full((1, 1), source_phase, device=device, dtype=dtype),
    )
    assert torch.allclose(
        source_peak,
        torch.full_like(source_peak, source_amplitude),
        atol=tolerance,
        rtol=tolerance,
    )

    residual = pde_residual(model, interior)
    flux = boundary_flux(model, boundary, normals)
    assert residual.shape == (count, 1) and torch.isfinite(residual).all()
    assert flux.shape == (count, 1) and torch.isfinite(flux).all()
    smoke_loss = residual.square().mean() + flux.square().mean()
    gradients = torch.autograd.grad(smoke_loss, tuple(model.parameters()))
    assert all(torch.isfinite(value).all() for value in gradients)

    initial_points = sample_interior(count, device, dtype)
    initial_points[:, 2] = 0.0
    with torch.inference_mode():
        initial_error = (
            model(initial_points)
            - initial_condition(initial_points[:, 0:1], initial_points[:, 1:2])
        ).abs().max().item()
    assert initial_error <= 10.0 * torch.finfo(dtype).eps
    return initial_error


def evaluate_residual_groups(
    model: MonodomainPINN,
    components: dict[str, Tensor],
    boundary_points: Tensor,
    normals: Tensor,
    boundary_weight: float,
) -> dict[str, float]:
    model.eval()
    values = {
        f"{name}_pde_loss": float(
            pde_residual(model, points).square().mean().detach().cpu()
        )
        for name, points in components.items()
    }
    values["boundary_loss"] = float(
        boundary_flux(model, boundary_points, normals)
        .square()
        .mean()
        .detach()
        .cpu()
    )
    counts = {name: len(points) for name, points in components.items()}
    values["pde_loss"] = sum(
        counts[name] * values[f"{name}_pde_loss"] for name in components
    ) / sum(counts.values())
    values["total_loss"] = (
        values["pde_loss"] + boundary_weight * values["boundary_loss"]
    )
    return values


def train_model(
    model: MonodomainPINN,
    config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, float]], dict[str, Any], dict[str, Tensor]]:
    learning_rate_levels = (1.0e-3, 3.0e-4, 1.0e-4)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate_levels[0], foreach=False
    )
    learning_rate_stage = 0

    validation_components = sample_interior_components(
        config.validation_interior, device, dtype
    )
    validation_boundary_points, validation_normals = sample_boundary(
        config.validation_boundary, device, dtype
    )

    history: list[dict[str, float]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_losses: dict[str, float] = {}
    best_state: dict[str, Tensor] | None = None
    checks_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        training_components = sample_interior_components(
            config.interior_batch, device, dtype
        )
        interior = torch.cat(tuple(training_components.values()))
        boundary_points, boundary_normals = sample_boundary(
            config.boundary_batch, device, dtype
        )
        residual = pde_residual(model, interior)
        component_counts = [len(points) for points in training_components.values()]
        residual_groups = torch.split(residual, component_counts)
        group_losses = {
            f"{name}_pde_loss": group_residual.square().mean()
            for name, group_residual in zip(training_components, residual_groups)
        }
        pde_loss = residual.square().mean()
        boundary_loss = boundary_flux(
            model, boundary_points, boundary_normals
        ).square().mean()
        total_loss = pde_loss + config.boundary_weight * boundary_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        max_norm = config.grad_clip if epoch <= config.clip_warmup_epochs else math.inf
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient at epoch {epoch}")
        optimizer.step()

        row = {
            "epoch": float(epoch),
            "total_loss": float(total_loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            **{
                name: float(value.detach().cpu())
                for name, value in group_losses.items()
            },
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

        should_validate = (
            epoch == 1
            or epoch % config.validation_every == 0
            or epoch == config.epochs
        )
        stop_now = False
        if should_validate:
            validation = evaluate_residual_groups(
                model,
                validation_components,
                validation_boundary_points,
                validation_normals,
                config.boundary_weight,
            )
            row.update({f"validation_{key}": value for key, value in validation.items()})
            improved = validation["total_loss"] < best_validation_loss * (
                1.0 - config.validation_min_delta
            )
            if improved:
                best_validation_loss = validation["total_loss"]
                best_epoch = epoch
                best_losses = validation.copy()
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            if (
                checks_without_improvement >= config.lr_patience_checks
                and learning_rate_stage < len(learning_rate_levels) - 1
            ):
                learning_rate_stage += 1
                new_lr = learning_rate_levels[learning_rate_stage]
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                checks_without_improvement = 0
                print(f"Validation plateau: reducing learning rate to {new_lr:.1e}")
            elif (
                learning_rate_stage == len(learning_rate_levels) - 1
                and checks_without_improvement >= config.early_stop_checks
            ):
                stopped_early = True
                stop_now = True

        history.append(row)
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            validation_text = ""
            if should_validate:
                validation_text = (
                    f"  val={validation['total_loss']:.4e} "
                    f"[uniform={validation['uniform_pde_loss']:.2e}, "
                    f"source={validation['source_pde_loss']:.2e}, "
                    f"pulse={validation['pulse_pde_loss']:.2e}, "
                    f"boundary={validation['boundary_loss']:.2e}]"
                )
            print(
                f"epoch {epoch:6d}/{config.epochs}  train={row['total_loss']:.4e} "
                f"[uniform={row['uniform_pde_loss']:.2e}, "
                f"source={row['source_pde_loss']:.2e}, "
                f"pulse={row['pulse_pde_loss']:.2e}, "
                f"boundary={row['boundary_loss']:.2e}]  "
                f"grad={row['gradient_norm']:.2e} lr={row['learning_rate']:.2e}"
                f"{validation_text}"
            )
        if stop_now:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation epoch was {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without a finite validation checkpoint")

    model.load_state_dict(best_state)
    restored_losses = evaluate_residual_groups(
        model,
        validation_components,
        validation_boundary_points,
        validation_normals,
        config.boundary_weight,
    )
    checkpoint_tolerance = max(1.0e-10, 100.0 * torch.finfo(dtype).eps)
    if not math.isclose(
        restored_losses["total_loss"],
        best_validation_loss,
        rel_tol=checkpoint_tolerance,
        abs_tol=checkpoint_tolerance,
    ):
        raise RuntimeError("Restored checkpoint does not reproduce its validation loss")

    summary: dict[str, Any] = {
        "epochs_completed": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation_loss,
        "best_validation_losses": best_losses,
        "restored_validation_losses": restored_losses,
        "learning_rate_stage": learning_rate_stage,
    }
    return history, summary, best_state


def save_history(history: list[dict[str, float]], output_file: Path) -> None:
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], output_file: Path) -> None:
    if plt is None:
        print("Skipping loss plot because matplotlib is not installed")
        return
    epochs = [row["epoch"] for row in history]
    validation_rows = [row for row in history if "validation_total_loss" in row]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for loss_name in ("total_loss", "pde_loss", "boundary_loss"):
        axes[0].plot(
            epochs,
            [row[loss_name] for row in history],
            label=f"train {loss_name}",
            linewidth=1.0,
            alpha=0.75,
        )
    axes[0].plot(
        [row["epoch"] for row in validation_rows],
        [row["validation_total_loss"] for row in validation_rows],
        label="fixed validation total",
        linewidth=2.0,
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Loss",
        yscale="log",
        title="Training and validation losses",
    )
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for group in ("uniform", "source", "pulse"):
        axes[1].plot(
            epochs,
            [row[f"{group}_pde_loss"] for row in history],
            label=f"train {group}",
            linewidth=0.8,
            alpha=0.45,
        )
        axes[1].plot(
            [row["epoch"] for row in validation_rows],
            [row[f"validation_{group}_pde_loss"] for row in validation_rows],
            label=f"validation {group}",
            linewidth=1.5,
        )
    axes[1].plot(
        [row["epoch"] for row in validation_rows],
        [row["validation_boundary_loss"] for row in validation_rows],
        label="validation boundary",
        linewidth=1.5,
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Loss",
        yscale="log",
        title="Loss by sampling stratum",
    )
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(prediction - reference) / max(denominator, 1.0e-14))


def evaluate_nominal_case(
    model: MonodomainPINN,
    device: torch.device,
    dtype: torch.dtype,
    comparison_dir: Path,
    output_dir: Path,
) -> list[dict[str, float]] | None:
    points_file = comparison_dir / "comparison_points.txt"
    if not points_file.exists():
        print(f"Skipping nominal snapshots because {points_file} was not found")
        return None

    points_xy = np.atleast_2d(np.loadtxt(points_file))
    snapshot_values = np.empty(
        (len(snapshot_times), len(points_xy)), dtype=np.float64
    )
    model.eval()
    with torch.inference_mode():
        for index, snapshot_time in enumerate(snapshot_times):
            points = np.column_stack(
                (
                    points_xy,
                    np.full(len(points_xy), snapshot_time),
                    np.full(len(points_xy), nominal_a1),
                    np.full(len(points_xy), nominal_a2),
                )
            )
            inputs = torch.as_tensor(points, device=device, dtype=dtype)
            snapshot_values[index] = model(inputs).squeeze(-1).cpu().numpy()

    snapshot_file = output_dir / "pinn_parametric_tuned_nominal_snapshots.npz"
    np.savez_compressed(
        snapshot_file,
        points=points_xy,
        times=snapshot_times,
        u=snapshot_values,
        a1=np.asarray(nominal_a1),
        a2=np.asarray(nominal_a2),
    )
    print(f"Saved nominal PINN snapshots to {snapshot_file}")

    fem_file = comparison_dir / "fem_comparison_snapshots.npz"
    if not fem_file.exists():
        print(f"Skipping FEM error diagnostics because {fem_file} was not found")
        return None

    with np.load(fem_file) as fem_data:
        fem_points = fem_data["points"]
        fem_times = fem_data["times"]
        fem_values = fem_data["u"]
    if not (
        np.allclose(points_xy, fem_points)
        and np.allclose(snapshot_times, fem_times)
        and fem_values.shape == snapshot_values.shape
    ):
        raise ValueError("FEM and PINN snapshot coordinates/times do not match")

    source_radius = np.linalg.norm(
        points_xy - np.array([source_x, source_y]), axis=1
    )
    source_mask = source_radius <= 2.0 * source_spatial_width
    fem_errors: list[dict[str, float]] = []
    for index, snapshot_time in enumerate(snapshot_times):
        row = {
            "time": float(snapshot_time),
            "relative_l2": relative_l2(snapshot_values[index], fem_values[index]),
            "source_relative_l2": relative_l2(
                snapshot_values[index, source_mask], fem_values[index, source_mask]
            ),
        }
        fem_errors.append(row)
        print(
            f"t={row['time']:.2f}: relative L2={row['relative_l2']:.3e}, "
            f"source-region relative L2={row['source_relative_l2']:.3e}"
        )
    return fem_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float64"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12_000)
    parser.add_argument("--interior-batch", type=int, default=8_192)
    parser.add_argument("--boundary-batch", type=int, default=512)
    parser.add_argument("--validation-interior", type=int, default=4_096)
    parser.add_argument("--validation-boundary", type=int, default=512)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("parametric_training_tuned")
    )
    parser.add_argument(
        "--comparison-dir", type=Path, default=Path("comparison_results")
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Do not generate nominal snapshots or FEM diagnostics.",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.interior_batch < 4 or args.validation_interior < 4:
        parser.error("interior batch sizes must be at least 4")
    if args.boundary_batch < 2 or args.validation_boundary < 2:
        parser.error("boundary batch sizes must be at least 2")
    if args.validation_every <= 0 or args.log_every <= 0:
        parser.error("validation and logging intervals must be positive")
    return args


def _json_config(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.set_default_dtype(torch.float32)
    set_seed(args.seed)

    config = TrainingConfig(
        epochs=args.epochs,
        interior_batch=args.interior_batch,
        boundary_batch=args.boundary_batch,
        validation_interior=args.validation_interior,
        validation_boundary=args.validation_boundary,
        validation_every=args.validation_every,
        log_every=args.log_every,
        seed=args.seed,
    )
    spec = ModelSpec()
    model = MonodomainPINN(spec).to(device=device, dtype=dtype)
    print(f"Using device={device}; training dtype={dtype}")
    initial_error = validate_setup(model, device, dtype)
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"maximum initial-condition error: {initial_error:.3e}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    history, training_summary, best_state = train_model(
        model, config, device, dtype
    )
    elapsed_minutes = (time.time() - start_time) / 60.0
    training_summary["elapsed_minutes"] = elapsed_minutes
    print(
        f"Best fixed-validation loss "
        f"{training_summary['best_validation_total']:.4e} at epoch "
        f"{training_summary['best_epoch']}"
    )
    print(f"Total training time: {elapsed_minutes:.1f} minutes")

    checkpoint = {
        "format_version": 1,
        "model_state_dict": best_state,
        "model_spec": asdict(spec),
        "parameter_bounds": {
            "a1": [A_MIN, A_MAX],
            "a2": [A_MIN, A_MAX],
        },
        "nominal_parameters": {"a1": nominal_a1, "a2": nominal_a2},
        "training_config": _json_config(config),
        "dtype": str(dtype),
        "best_epoch": training_summary["best_epoch"],
        "best_validation_losses": training_summary["best_validation_losses"],
    }
    checkpoint_file = args.output_dir / "pinn_parametric_tuned_best.pt"
    torch.save(checkpoint, checkpoint_file)
    save_history(history, args.output_dir / "training_history.csv")
    plot_history(history, args.output_dir / "pinn_parametric_tuned_losses.png")
    print(f"Saved best checkpoint to {checkpoint_file}")

    fem_errors = None
    if not args.skip_comparison:
        fem_errors = evaluate_nominal_case(
            model, device, dtype, args.comparison_dir, args.output_dir
        )

    summary = {
        "model_spec": asdict(spec),
        "training_config": _json_config(config),
        "device": str(device),
        "dtype": str(dtype),
        "parameter_bounds": {
            "a1": [A_MIN, A_MAX],
            "a2": [A_MIN, A_MAX],
        },
        "nominal_parameters": {"a1": nominal_a1, "a2": nominal_a2},
        "initial_condition_max_error": initial_error,
        **training_summary,
        "nominal_fem_errors": fem_errors,
    }
    summary_file = args.output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved training summary to {summary_file}")


if __name__ == "__main__":
    main()
