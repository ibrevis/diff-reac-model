#!/usr/bin/env python3
"""Physics-informed neural network for the passive monodomain problem.

This solves the same initial-boundary-value problem as ``FEM-code-new.py``:

    u_t - div(a(x, y) grad(u)) + r u = 0       in x^2 + y^2 < 1,
    (a grad(u)) . n = 0                        on x^2 + y^2 = 1,
    u(x, y, 0) = exp(-((x-0.4)^2 + y^2)/(2*0.12^2)).

No FEM data are used during training.  The initial condition is imposed exactly
by the network output transform; the PDE and Neumann condition form the loss.

Example
-------
    conda activate ml-torch
    python PINN-code.py

For a quick end-to-end check:
    python PINN-code.py --epochs 2 --interior-batch 32 --boundary-batch 16 \
        --hidden-layers 2 --hidden-width 16 --grid-size 31 --output-dir smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


# These physical parameters intentionally mirror FEM-code-new.py.
T_FINAL = 5.0
A_STIM = 1.0
S_STIM = 0.12
X0, Y0 = 0.4, 0.0
A_HEALTHY, A_SCAR = 0.1, 0.01
XA, YA, WA = -0.3, 0.0, 0.2
R_VALUE = 1.0


@dataclass(frozen=True)
class ModelSpec:
    hidden_layers: int
    hidden_width: int


class MonodomainPINN(nn.Module):
    """MLP approximation with an output transform enforcing u(x, y, 0)=u0."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 3
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
        normalized = torch.cat((x, y, 2.0 * t / T_FINAL - 1.0), dim=1)
        raw = self.network(normalized)

        # This reaction-aware transform is exactly u0 at t=0 while leaving the
        # network free to represent the solution for every t>0.
        reaction_decay = torch.exp(-R_VALUE * t)
        return reaction_decay * initial_condition(x, y) + (1.0 - reaction_decay) * raw


def initial_condition(x: Tensor, y: Tensor) -> Tensor:
    radius_sq = (x - X0) ** 2 + (y - Y0) ** 2
    return A_STIM * torch.exp(-radius_sq / (2.0 * S_STIM**2))


def diffusivity(x: Tensor, y: Tensor) -> Tensor:
    scar_radius_sq = (x - XA) ** 2 + (y - YA) ** 2
    scar = torch.exp(-scar_radius_sq / (2.0 * WA**2))
    return A_HEALTHY - (A_HEALTHY - A_SCAR) * scar


def sample_times(count: int, device: torch.device, dtype: torch.dtype,
                 early_time_fraction: float) -> Tensor:
    uniform_draw = torch.rand((count, 1), device=device, dtype=dtype)
    early_draw = torch.rand((count, 1), device=device, dtype=dtype)
    choose_early = torch.rand((count, 1), device=device) < early_time_fraction
    # Squaring a uniform draw concentrates samples near the rapid initial transient.
    return T_FINAL * torch.where(choose_early, early_draw.square(), uniform_draw)


def sample_interior(count: int, device: torch.device, dtype: torch.dtype,
                    early_time_fraction: float) -> Tensor:
    radius = torch.sqrt(torch.rand((count, 1), device=device, dtype=dtype))
    angle = 2.0 * math.pi * torch.rand((count, 1), device=device, dtype=dtype)
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    t = sample_times(count, device, dtype, early_time_fraction)
    return torch.cat((x, y, t), dim=1)


def sample_boundary(count: int, device: torch.device, dtype: torch.dtype,
                    early_time_fraction: float) -> Tensor:
    angle = 2.0 * math.pi * torch.rand((count, 1), device=device, dtype=dtype)
    x = torch.cos(angle)
    y = torch.sin(angle)
    t = sample_times(count, device, dtype, early_time_fraction)
    return torch.cat((x, y, t), dim=1)


def gradient(value: Tensor, points: Tensor, create_graph: bool = True) -> Tensor:
    result = torch.autograd.grad(
        value,
        points,
        grad_outputs=torch.ones_like(value),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return result


def pde_residual(model: MonodomainPINN, points: Tensor) -> Tensor:
    points = points.detach().requires_grad_(True)
    u = model(points)
    grad_u = gradient(u, points)
    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    a = diffusivity(points[:, 0:1], points[:, 1:2])
    flux_x, flux_y = a * u_x, a * u_y
    div_flux = gradient(flux_x, points)[:, 0:1] + gradient(flux_y, points)[:, 1:2]
    return u_t - div_flux + R_VALUE * u


def boundary_flux(model: MonodomainPINN, points: Tensor) -> Tensor:
    points = points.detach().requires_grad_(True)
    u = model(points)
    grad_u = gradient(u, points)
    a = diffusivity(points[:, 0:1], points[:, 1:2])
    # On the unit circle the outward unit normal is n=(x,y).
    return a * (grad_u[:, 0:1] * points[:, 0:1] +
                grad_u[:, 1:2] * points[:, 1:2])


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_payload(model: MonodomainPINN, spec: ModelSpec, args: argparse.Namespace,
                       epoch: int, losses: dict[str, float]) -> dict:
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["snapshot_times"] = list(config["snapshot_times"])
    return {
        "model_state": model.state_dict(),
        "model_spec": asdict(spec),
        "epoch": epoch,
        "losses": losses,
        "config": config,
        "physical_parameters": {
            "T": T_FINAL, "A_stim": A_STIM, "s_stim": S_STIM,
            "x0": X0, "y0": Y0, "a_healthy": A_HEALTHY,
            "a_scar": A_SCAR, "xa": XA, "ya": YA, "wa": WA,
            "r": R_VALUE,
        },
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(epochs, [row["total_loss"] for row in history], label="total")
    axis.semilogy(epochs, [row["pde_loss"] for row in history], label="PDE")
    axis.semilogy(epochs, [row["boundary_loss"] for row in history], label="boundary")
    axis.set(xlabel="Epoch", ylabel="Mean squared residual", title="PINN training history")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def predict_in_batches(model: MonodomainPINN, points: Tensor, batch_size: int = 65536) -> Tensor:
    return torch.cat([model(points[start:start + batch_size])
                      for start in range(0, len(points), batch_size)], dim=0)


def export_solution(model: MonodomainPINN, args: argparse.Namespace,
                    device: torch.device, dtype: torch.dtype) -> None:
    coordinates = np.linspace(-1.0, 1.0, args.grid_size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    mask = x_grid**2 + y_grid**2 <= 1.0 + 1.0e-12
    xy = np.column_stack((x_grid[mask], y_grid[mask]))

    fields: list[np.ndarray] = []
    for time in args.snapshot_times:
        points_np = np.column_stack((xy, np.full(len(xy), time)))
        points = torch.as_tensor(points_np, device=device, dtype=dtype)
        values = predict_in_batches(model, points).squeeze(1).cpu().numpy()
        field = np.full(x_grid.shape, np.nan, dtype=np.float64)
        field[mask] = values
        fields.append(field)
    u = np.stack(fields)

    np.savez_compressed(
        args.output_dir / "pinn_solution.npz",
        x=coordinates,
        y=coordinates,
        mask=mask,
        times=np.asarray(args.snapshot_times, dtype=np.float64),
        u=u,
    )

    columns = min(3, len(args.snapshot_times))
    rows = math.ceil(len(args.snapshot_times) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.2 * rows), squeeze=False)
    for axis, time, field in zip(axes.flat, args.snapshot_times, fields):
        image = axis.pcolormesh(x_grid, y_grid, field, shading="auto", cmap="viridis")
        axis.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="black", linewidth=0.8))
        axis.set(aspect="equal", xlabel="x", ylabel="y", title=f"PINN: t = {time:g}")
        fig.colorbar(image, ax=axis, label="u")
    for axis in axes.flat[len(fields):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "solution_snapshots.png", dpi=180)
    plt.close(fig)


def validate_setup(model: MonodomainPINN, device: torch.device, dtype: torch.dtype,
                   early_time_fraction: float) -> float:
    interior = sample_interior(1024, device, dtype, early_time_fraction)
    boundary = sample_boundary(1024, device, dtype, early_time_fraction)
    if torch.any(interior[:, :2].square().sum(dim=1) > 1.0 + 1.0e-5):
        raise RuntimeError("Interior sampler generated a point outside the unit disk")
    boundary_radius = boundary[:, :2].square().sum(dim=1)
    tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-12
    if not torch.allclose(boundary_radius, torch.ones_like(boundary_radius), atol=tolerance):
        raise RuntimeError("Boundary sampler generated a point off the unit circle")

    initial_points = interior.detach().clone()
    initial_points[:, 2] = 0.0
    with torch.no_grad():
        predicted = model(initial_points)
        expected = initial_condition(initial_points[:, 0:1], initial_points[:, 1:2])
        initial_error = float(torch.max(torch.abs(predicted - expected)).cpu())
    if initial_error > tolerance:
        raise RuntimeError(f"Initial-condition transform error is {initial_error:.3e}")
    return initial_error


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; use --dtype float32")
    torch.set_default_dtype(dtype)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = ModelSpec(args.hidden_layers, args.hidden_width)
    model = MonodomainPINN(spec).to(device=device, dtype=dtype)
    initial_error = validate_setup(model, device, dtype, args.early_time_fraction)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )

    print(f"device={device}, dtype={args.dtype}, parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"maximum initial-condition error: {initial_error:.3e}")
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 0
    best_losses: dict[str, float] = {}
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        interior = sample_interior(
            args.interior_batch, device, dtype, args.early_time_fraction
        )
        boundary = sample_boundary(
            args.boundary_batch, device, dtype, args.early_time_fraction
        )
        residual = pde_residual(model, interior)
        flux = boundary_flux(model, boundary)
        pde_loss = residual.square().mean()
        boundary_loss = flux.square().mean()
        total_loss = pde_loss + args.boundary_weight * boundary_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient at epoch {epoch}")
        optimizer.step()
        scheduler.step()

        losses = {
            "total_loss": float(total_loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
        }
        row = {
            "epoch": epoch,
            **losses,
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        history.append(row)
        if losses["total_loss"] < best_loss:
            best_loss = losses["total_loss"]
            best_epoch = epoch
            best_losses = losses
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:6d}/{args.epochs}  total={losses['total_loss']:.4e}  "
                f"pde={losses['pde_loss']:.4e}  boundary={losses['boundary_loss']:.4e}  "
                f"lr={row['learning_rate']:.2e}"
            )

    final_losses = {key: history[-1][key]
                    for key in ("total_loss", "pde_loss", "boundary_loss")}
    save_checkpoint(
        args.output_dir / "model_final.pt",
        checkpoint_payload(model, spec, args, args.epochs, final_losses),
    )
    if best_state is None:
        raise RuntimeError("Training finished without a valid best model")
    model.load_state_dict(best_state)
    save_checkpoint(
        args.output_dir / "model_best.pt",
        checkpoint_payload(model, spec, args, best_epoch, best_losses),
    )
    write_history(args.output_dir / "training_history.csv", history)
    plot_history(args.output_dir / "training_loss.png", history)
    model.eval()
    export_solution(model, args, device, dtype)

    summary = {
        "device": str(device),
        "dtype": args.dtype,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_condition_max_error": initial_error,
        "best_epoch": best_epoch,
        "best_losses": best_losses,
        "npz_shapes": {
            "x": [args.grid_size],
            "y": [args.grid_size],
            "mask": [args.grid_size, args.grid_size],
            "times": [len(args.snapshot_times)],
            "u": [len(args.snapshot_times), args.grid_size, args.grid_size],
        },
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best epoch={best_epoch}, best total loss={best_loss:.4e}")
    print(f"outputs written to {args.output_dir.resolve()}")


def parse_snapshot_times(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("snapshot times must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one snapshot time is required")
    if any(value < 0.0 or value > T_FINAL for value in values):
        raise argparse.ArgumentTypeError(f"snapshot times must lie in [0, {T_FINAL}]")
    return values


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a pure PyTorch PINN for the unit-disk monodomain problem."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pinn_monodomain_results"))
    parser.add_argument("--epochs", type=positive_int, default=12_000)
    parser.add_argument("--interior-batch", type=positive_int, default=2_048)
    parser.add_argument("--boundary-batch", type=positive_int, default=512)
    parser.add_argument("--hidden-layers", type=positive_int, default=5)
    parser.add_argument("--hidden-width", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-time-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=positive_int, default=100)
    parser.add_argument("--grid-size", type=positive_int, default=201)
    parser.add_argument(
        "--snapshot-times", type=parse_snapshot_times,
        default=parse_snapshot_times("0,1,2,3,4,5"),
        help="comma-separated output times in [0,5] (default: 0,1,2,3,4,5)",
    )
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.grid_size < 3:
        parser.error("--grid-size must be at least 3")
    if args.learning_rate <= 0.0 or args.min_learning_rate < 0.0:
        parser.error("learning rates must be positive (minimum may be zero)")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if args.boundary_weight < 0.0:
        parser.error("--boundary-weight cannot be negative")
    if args.grad_clip <= 0.0:
        parser.error("--grad-clip must be positive")
    if not 0.0 <= args.early_time_fraction <= 1.0:
        parser.error("--early-time-fraction must lie in [0,1]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args, parser)
    train(args)


if __name__ == "__main__":
    main()
