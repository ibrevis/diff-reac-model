# %%
import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


# %%
# Physical parameters inherited from PINN-code-simple.ipynb.
T_FINAL = 5.0
A_STIM = 1.0
S_STIM = 0.12
X0, Y0 = 0.4, 0.0
A_MIN, A_MAX = 0.1, 0.9
R_VALUE = 1.0


# %%
@dataclass(frozen=True)
class ModelSpec:
    hidden_layers: int
    hidden_width: int


class MonodomainPINN(nn.Module):
    """MLP approximation with an output transform enforcing u(x, y, 0)=u0."""

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
        a1 = points[:, 3:4]
        a2 = points[:, 4:5]
        # Normalize time and both parameters to [-1, 1]; x and y already lie there.
        t_normalized = 2.0 * t / T_FINAL - 1.0
        a1_normalized = 2.0 * (a1 - A_MIN) / (A_MAX - A_MIN) - 1.0
        a2_normalized = 2.0 * (a2 - A_MIN) / (A_MAX - A_MIN) - 1.0
        normalized = torch.cat(
            (x, y, t_normalized, a1_normalized, a2_normalized), dim=1
        )
        raw = self.network(normalized)

        # This reaction-aware transform is exactly u0 at t=0 while leaving the
        # network free to represent the solution for every t>0.
        reaction_decay = torch.exp(-R_VALUE * t)
        return reaction_decay * initial_condition(x, y) + (1.0 - reaction_decay) * raw


# %% [markdown]
# ## Parametric anisotropic diffusion
#
# The network represents u(x,y,t,a1,a2) for independent, spatially constant
# a1,a2 in [0.1,0.9]. It is trained on
#
#     u_t - a1*u_xx - a2*u_yy + R*u = 0
#
# in the unit disk, with zero normal flux and the same Gaussian initial
# condition as the non-parametric notebook.


# %%
def initial_condition(x: Tensor, y: Tensor) -> Tensor:
    radius_sq = (x - X0) ** 2 + (y - Y0) ** 2
    return A_STIM * torch.exp(-radius_sq / (2.0 * S_STIM**2))


def sample_diffusivities(
    count: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    # Independent coefficients for every collocation point.
    return A_MIN + (A_MAX - A_MIN) * torch.rand(
        (count, 2), device=device, dtype=dtype
    )


def sample_times(
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    early_time_fraction: float,
) -> Tensor:
    uniform_draw = torch.rand((count, 1), device=device, dtype=dtype)
    early_draw = torch.rand((count, 1), device=device, dtype=dtype)
    choose_early = torch.rand((count, 1), device=device) < early_time_fraction
    # Squaring a uniform draw concentrates samples near the rapid initial transient.
    return T_FINAL * torch.where(choose_early, early_draw.square(), uniform_draw)


def sample_interior(
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    early_time_fraction: float,
) -> Tensor:
    radius = torch.sqrt(torch.rand((count, 1), device=device, dtype=dtype))
    angle = 2.0 * math.pi * torch.rand((count, 1), device=device, dtype=dtype)
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    t = sample_times(count, device, dtype, early_time_fraction)
    diffusivities = sample_diffusivities(count, device, dtype)
    return torch.cat((x, y, t, diffusivities), dim=1)


def sample_boundary(
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    early_time_fraction: float,
) -> Tensor:
    angle = 2.0 * math.pi * torch.rand((count, 1), device=device, dtype=dtype)
    x = torch.cos(angle)
    y = torch.sin(angle)
    t = sample_times(count, device, dtype, early_time_fraction)
    diffusivities = sample_diffusivities(count, device, dtype)
    return torch.cat((x, y, t, diffusivities), dim=1)


# %%
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
    a1, a2 = points[:, 3:4], points[:, 4:5]

    flux_x, flux_y = a1 * u_x, a2 * u_y
    div_flux = gradient(flux_x, points)[:, 0:1] + gradient(flux_y, points)[:, 1:2]
    return u_t - div_flux + R_VALUE * u


def boundary_flux(model: MonodomainPINN, points: Tensor) -> Tensor:
    points = points.detach().requires_grad_(True)
    u = model(points)
    grad_u = gradient(u, points)
    a1, a2 = points[:, 3:4], points[:, 4:5]
    # On the unit circle the outward unit normal is n=(x,y).
    return (
        a1 * grad_u[:, 0:1] * points[:, 0:1]
        + a2 * grad_u[:, 1:2] * points[:, 1:2]
    )


# %%
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# %%
dtype = torch.float32  # torch.float64
if device.type == "mps" and dtype == torch.float64:
    raise ValueError("MPS does not support float64; use --dtype float32")
torch.set_default_dtype(dtype)


# %%
set_seed(42)


# %%
hidden_layers = 5
hidden_width = 128
learning_rate = 1.0e-3
epochs = 120000
min_learning_rate = 1.0e-5

interior_batch = 8192  # 4096  # 2048
boundary_batch = 512
early_time_fraction = 0.5

boundary_weight = 10.0
grad_clip = 1.0
log_every = 100


# %%
spec = ModelSpec(hidden_layers, hidden_width)
model = MonodomainPINN(spec).to(device=device, dtype=dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=min_learning_rate
)


# %%
def validate_setup(
    model: MonodomainPINN,
    device: torch.device,
    dtype: torch.dtype,
    early_time_fraction: float,
    count: int = 32,
) -> float:
    interior = sample_interior(count, device, dtype, early_time_fraction)
    boundary = sample_boundary(count, device, dtype, early_time_fraction)

    for name, points in (("interior", interior), ("boundary", boundary)):
        assert points.shape == (count, 5), f"Unexpected {name} shape: {points.shape}"
        coefficients = points[:, 3:5]
        assert torch.all(coefficients >= A_MIN)
        assert torch.all(coefficients <= A_MAX)

    residual = pde_residual(model, interior)
    flux = boundary_flux(model, boundary)
    assert residual.shape == (count, 1) and torch.isfinite(residual).all()
    assert flux.shape == (count, 1) and torch.isfinite(flux).all()

    initial_points = sample_interior(count, device, dtype, early_time_fraction)
    initial_points[:, 2] = 0.0
    with torch.inference_mode():
        initial_error = (
            model(initial_points)
            - initial_condition(initial_points[:, 0:1], initial_points[:, 1:2])
        ).abs().max().item()
    assert initial_error <= 10.0 * torch.finfo(dtype).eps
    return initial_error


initial_error = validate_setup(model, device, dtype, early_time_fraction)
print(
    f"device={device}, dtype={dtype}, "
    f"parameters={sum(p.numel() for p in model.parameters()):,}"
)
print(f"maximum initial-condition error: {initial_error:.3e}")


# %%
history: list[dict[str, float]] = []
best_loss = math.inf
best_epoch = 0
best_losses: dict[str, float] = {}
best_state: dict[str, Tensor] | None = None

for epoch in range(1, epochs + 1):
    model.train()
    interior = sample_interior(
        interior_batch, device, dtype, early_time_fraction
    )
    boundary = sample_boundary(
        boundary_batch, device, dtype, early_time_fraction
    )
    residual = pde_residual(model, interior)
    flux = boundary_flux(model, boundary)
    pde_loss = residual.square().mean()
    boundary_loss = flux.square().mean()
    total_loss = pde_loss + boundary_weight * boundary_loss

    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite loss at epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
        print(
            f"epoch {epoch:6d}/{epochs}  total={losses['total_loss']:.4e}  "
            f"pde={losses['pde_loss']:.4e}  "
            f"boundary={losses['boundary_loss']:.4e}  "
            f"lr={row['learning_rate']:.2e}"
        )

dir_path = Path("parametric_training")
os.makedirs(dir_path, exist_ok=True)
torch.save(best_state, dir_path / "pinn_parametric_best.pt")


# %%
epochs_history = [row["epoch"] for row in history]

plt.figure(figsize=(9, 5))

for loss_name in ("total_loss", "pde_loss", "boundary_loss"):
    plt.plot(
        epochs_history,
        [row[loss_name] for row in history],
        label=loss_name,
        linewidth=1.5,
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")  # Useful because the losses differ greatly in magnitude
plt.title("PINN training losses")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(dir_path / "pinn_parametric_losses.png", dpi=300)


# %%
final_losses = {
    key: history[-1][key]
    for key in ("total_loss", "pde_loss", "boundary_loss")
}
