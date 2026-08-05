from mpi4py import MPI
import numpy as np

from dolfinx import fem, geometry, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import ufl
from petsc4py import PETSc

from FEM_functions import plot_solution
from mesh_generator import generate_disk_with_hole_mesh

from matplotlib import pyplot as plt

from pathlib import Path

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------
from params import (
    A_stim,
    T,
    a1,
    a2,
    dt,
    h,
    hole_r,
    hole_x,
    hole_y,
    num_steps,
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
    theta,
    x0,
    y0,
)

print(f"Target mesh size h = {h:.4f}  =>  expected number of vertices ~ {1/h**2:.0f}")
print(f"Time step dt = {dt:.4f}  =>  expected number of time steps ~ {T/dt:.0f}")

results_dir = Path("comparison_results")
results_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 2. Mesh: unit disk with a circular hole via gmsh
# ----------------------------------------------------------------------
mesh_comm  = MPI.COMM_WORLD
model_rank = 0
domain = generate_disk_with_hole_mesh(
    mesh_comm,
    model_rank,
    mesh_size=h,
    hole_center=(hole_x, hole_y),
    hole_radius=hole_r,
)

tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)

cells = domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)
points = domain.geometry.x[:, :2]

fig, ax = plt.subplots(figsize=(6, 6))
ax.triplot(points[:, 0], points[:, 1], cells, color="black", linewidth=0.25)

ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Mesh: unit disk with circular hole")
plt.savefig("mesh.png", dpi=200)
# plt.show()

# ----------------------------------------------------------------------
# 3. Function space and coefficients
# ----------------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 2))
x = ufl.SpatialCoordinate(domain)

# spatially varying diffusivity a(x, y)
# a_coeff = a_healthy - (a_healthy - a_scar) * ufl.exp(
#     -((x[0] - xa) ** 2 + (x[1] - ya) ** 2) / (2.0 * wa ** 2)
# )

a1_coeff = fem.Constant(domain, default_scalar_type(a1))
a2_coeff = fem.Constant(domain, default_scalar_type(a2))

D_coeff = ufl.as_matrix((
    (a1_coeff, 0.0),
    (0.0, a2_coeff),
))

# reaction coefficient (uniform here). To make it heterogeneous, replace with, e.g.:
#   r_coeff = 0.5 + 2.0 * ufl.exp(-((x[0]-xr)**2 + (x[1]-yr)**2)/(2*wr**2))
r_coeff = fem.Constant(domain, default_scalar_type(r_value))

dt_c    = fem.Constant(domain, default_scalar_type(dt))
theta_c = fem.Constant(domain, default_scalar_type(theta))

# External source F(x, t) = f(x) g(t). The temporal factors are Constants so
# the compiled linear form can be reused after updating them each time step.
source_spatial = source_amplitude * ufl.exp(
    -(
        (x[0] - source_x) ** 2 + (x[1] - source_y) ** 2
    ) / (2.0 * source_spatial_width ** 2)
)

def source_time_profile(t):
    """Smooth periodic Gaussian-like heartbeat with unit peak value."""
    phase_angle = np.pi * (t - source_phase) / source_period
    scaled_width = np.pi * source_time_width / source_period
    return np.exp(-np.sin(phase_angle) ** 2 / (2.0 * scaled_width ** 2))

source_time_n = fem.Constant(
    domain, default_scalar_type(source_time_profile(0.0))
)
source_time_np1 = fem.Constant(
    domain, default_scalar_type(source_time_profile(dt))
)

a1_plot = fem.Function(V)
a2_plot = fem.Function(V)
points = V.element.interpolation_points
points = points() if callable(points) else points
a1_plot.interpolate(fem.Expression(a1_coeff, points))
a2_plot.interpolate(fem.Expression(a2_coeff, points))
plot_solution(a1_plot,V, "x-direction diffusivity",display=False)
plot_solution(a2_plot,V, "y-direction reaction",display=False)

# ----------------------------------------------------------------------
# 4. Initial condition
# ----------------------------------------------------------------------
def u0_expr(x):
    return A_stim * np.exp(-((x[0] - x0) ** 2 + (x[1] - y0) ** 2) / (2.0 * s_stim ** 2))

u_n = fem.Function(V, name="u")     # solution at previous step
u_n.interpolate(u0_expr)

uh = fem.Function(V, name="u")      # solution at current step
uh.interpolate(u0_expr)

plot_solution(u_n,V, "Initial condition $u_0$",save=True,filename="u_initial.png",display=False)

# ----------------------------------------------------------------------
# 5. Source profiles: temporal heartbeat g(t) and spatial Gaussian f(x, y)
# ----------------------------------------------------------------------
time_samples = np.linspace(0.0, T, 2001)
heartbeat_values = source_time_profile(time_samples)

fig, ax_time = plt.subplots(figsize=(6, 5))
ax_time.plot(time_samples, heartbeat_values, color="tab:red")
ax_time.set_xlabel("Time $t$")
ax_time.set_ylabel("$g(t)$")
ax_time.set_title("Periodic heartbeat $g(t)$")
ax_time.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("source_time_distr.png", dpi=200)
plt.close(fig)

source_plot = fem.Function(V, name="f")
source_interpolation_points = V.element.interpolation_points
source_interpolation_points = (
    source_interpolation_points()
    if callable(source_interpolation_points)
    else source_interpolation_points
)
source_plot.interpolate(
    fem.Expression(source_spatial, source_interpolation_points)
)
plot_solution(
    source_plot,
    V,
    "Spatial source $f(x,y)$",
    colorbar_label="$f(x,y)$",
    save=True,
    filename="source_spatial_distr.png"
)

# ----------------------------------------------------------------------
# 6. Variational forms (theta-scheme)
# ----------------------------------------------------------------------
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

def B(w):   # weak spatial operator  (a grad w, grad v) + (r w, v)
    # return a_coeff * ufl.dot(ufl.grad(w), ufl.grad(v)) + r_coeff * w * v
    diffusion = ufl.inner(
        ufl.dot(D_coeff, ufl.grad(w)),
        ufl.grad(v),
    )
    reaction = r_coeff * w * v
    return diffusion + reaction

a_form = (u * v + dt_c * theta_c * B(u)) * ufl.dx
source_theta = (
    theta_c * source_time_np1 + (1.0 - theta_c) * source_time_n
) * source_spatial
L_form = (
    u_n * v
    - dt_c * (1.0 - theta_c) * B(u_n)
    + dt_c * source_theta * v
) * ufl.dx

bilinear_form = fem.form(a_form)
linear_form   = fem.form(L_form)

# Bilinear form is SPD (mass term + non-negative reaction), so the pure-Neumann
# system is non-singular: no Dirichlet BCs, no nullspace handling required.
A = assemble_matrix(bilinear_form)   # bcs default to [] (Neumann is natural)
A.assemble()
b = create_vector(V)

# ----------------------------------------------------------------------
# 7. Linear solver  (A is time-independent -> factor once, reuse)
# ----------------------------------------------------------------------
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# # ----------------------------------------------------------------------
# # 8. Output
# # ----------------------------------------------------------------------
# xdmf = XDMFFile(domain.comm, "monodomain_disk_with_hole.xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.write_function(uh, 0.0)

points_xy = np.loadtxt(results_dir / "comparison_points.txt")

# DOLFINx evaluates points as (x, y, z)
eval_points = np.zeros((len(points_xy), 3), dtype=domain.geometry.x.dtype)
eval_points[:, :2] = points_xy

# Find the mesh cell containing each point—only done once
tree = geometry.bb_tree(
    domain,
    domain.topology.dim,
    padding=1.0e-10,
)
candidate_cells = geometry.compute_collisions_points(tree, eval_points)
colliding_cells = geometry.compute_colliding_cells(
    domain, candidate_cells, eval_points
)

point_cells = np.full(len(points_xy), -1, dtype=np.int32)
for i in range(len(points_xy)):
    cells = colliding_cells.links(i)
    if len(cells) > 0:
        point_cells[i] = cells[0]

inside = point_cells >= 0
print(f"{inside.sum()} of {len(inside)} points lie inside the FEM mesh")

def evaluate_at_comparison_points(u):
    """Evaluate a scalar FEM function at all comparison points."""
    values = np.full(len(points_xy), np.nan)

    values[inside] = np.asarray(
        u.eval(eval_points[inside], point_cells[inside])
    ).reshape(-1).real

    return values


# Convert requested times to integer step numbers
snapshot_steps = np.rint(snapshot_times / dt).astype(int)

if not np.allclose(snapshot_steps * dt, snapshot_times):
    raise ValueError("Some snapshot_times do not coincide with time steps")

snapshot_index_by_step = {
    step: i for i, step in enumerate(snapshot_steps)
}

# Shape: (number of times, number of spatial points)
snapshot_values = np.full(
    (len(snapshot_times), len(points_xy)),
    np.nan,
)

# Save initial condition at t = 0
if 0 in snapshot_index_by_step:
    snapshot_values[snapshot_index_by_step[0]] = (
        evaluate_at_comparison_points(u_n)
    )

# ----------------------------------------------------------------------
# 9. Time-stepping
# ----------------------------------------------------------------------
t = 0.0
for n in range(num_steps):
    t_n = t
    t += dt
    step = n + 1

    source_time_n.value = default_scalar_type(source_time_profile(t_n))
    source_time_np1.value = default_scalar_type(source_time_profile(t))

    # assemble right-hand side
    with b.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b, linear_form)
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )

    # solve  A uh = b
    solver.solve(b, uh.x.petsc_vec)      # DOLFINx <= 0.7: use  uh.vector
    uh.x.scatter_forward()

    # Save requested snapshot
    if step in snapshot_index_by_step:
        snapshot_index = snapshot_index_by_step[step]
        snapshot_values[snapshot_index] = (
            evaluate_at_comparison_points(uh)
        )
        print(f"Stored snapshot at t = {t:.2f}")

    # advance:  u^n <- u^{n+1}
    u_n.x.array[:] = uh.x.array

    # if domain.comm.size == 1 and (n + 1) in snapshot_index_by_step:
    #     snapshot_index = snapshot_index_by_step[n + 1]
    #     snapshot_fields[snapshot_index] = sample_function(
    #         uh, sample_mask, sample_points, sample_cells
    #     )

    # diagnostics + output every 10 steps
#     if (n + 1) % 10 == 0:
#         umax = domain.comm.allreduce(np.max(np.abs(uh.x.array)), op=MPI.MAX)
#         if domain.comm.rank == 0:
#             print(f"step {n + 1:4d}   t = {t:5.2f}   max|u| = {umax:.4e}")
#         xdmf.write_function(uh, t)

# xdmf.close()

output_file = results_dir / "fem_comparison_snapshots.npz"

np.savez_compressed(
    output_file,
    points=points_xy,              # shape: (n_points, 2)
    times=snapshot_times,          # shape: (n_times,)
    u=snapshot_values,             # shape: (n_times, n_points)
)

print(f"Saved results to {output_file}")

# ----------------------------------------------------------------------
# 10. Save final-state plot
# ----------------------------------------------------------------------
plot_solution(uh,V, f"Final solution $u(x, t={T})$",save=True,filename="u_final.png",display=False)
