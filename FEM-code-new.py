"""
Linearized monodomain (passive cable) reaction-diffusion problem on the unit disk.

    u_t - div(a(x) grad u) + r(x) u = 0     in  D = { x^2 + y^2 <= 1 },  t in (0, T]
    (a grad u) . n = 0                       on  dD   (homogeneous Neumann: insulated tissue)
    u(x, 0) = u_0(x)                         (localized depolarization)

Time integration : theta-scheme  (theta = 1.0 -> backward Euler,  0.5 -> Crank-Nicolson)
Space             : continuous P1 Lagrange finite elements
Weak form (fully discrete, find u^{n+1} in H^1(D) for all v in H^1(D)):

    (u^{n+1}, v) + dt*theta * B(u^{n+1}, v) = (u^n, v) - dt*(1-theta) * B(u^n, v)
    with   B(w, v) = (a grad w, grad v) + (r w, v)

Tested with FEniCSx / DOLFINx 0.11  (+ gmsh, petsc4py, mpi4py).
Run serial:      python FEM-code-new.py
Run parallel:    mpirun -n 4 python FEM-code-new.py  (PNG plotting is skipped)
Output:          monodomain_disk.msh, monodomain_disk.xdmf/.h5 (open in ParaView),
                 mesh.png, u_initial.png, u_final.png, and fem_solution.npz
"""

from mpi4py import MPI
import numpy as np
import gmsh

from dolfinx import fem, geometry, default_scalar_type
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import ufl
from petsc4py import PETSc

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------
T         = 5.0          # final time (~ 5 membrane time constants, since r ~ 1)
num_steps = 250          # number of time steps
dt        = T / num_steps
theta     = 1.0          # 1.0 = backward Euler, 0.5 = Crank-Nicolson
h         = 0.05         # target mesh size (space constant lambda ~ sqrt(a/r) ~ 0.32)

# Initial depolarization: Gaussian bump  u0 = A * exp(-|x - x0|^2 / (2 s^2))
A_stim, s_stim = 1.0, 0.12
x0, y0         = 0.4, 0.0        # off-centre so the wave meets the scar and the boundary

# Diffusivity a(x): healthy tissue with a low-conductivity "scar" patch.
# Stays >= a_scar > 0 everywhere  => uniform ellipticity (well-posedness).
a_healthy, a_scar = 0.1, 0.01
xa, ya, wa        = -0.3, 0.0, 0.2

# Reaction r(x): uniform leak / repolarization rate (>= 0 for the maximum principle).
r_value = 1.0

# Regular-grid snapshots matching the defaults in PINN-code.py.
grid_size     = 201
snapshot_times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

# ----------------------------------------------------------------------
# 2. Mesh: unit disk via gmsh
# ----------------------------------------------------------------------
mesh_comm  = MPI.COMM_WORLD
model_rank = 0
gmsh.initialize()
if mesh_comm.rank == model_rank:
    gmsh.model.add("disk")
    disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)   # centre (0,0,0), radii (1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [disk], tag=1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(2)
    gmsh.write("monodomain_disk.msh")

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
# DOLFINx >= 0.9 returns a MeshData object; <= 0.8 returns a (mesh, cell_tags, facet_tags) tuple
domain = mesh_data.mesh if hasattr(mesh_data, "mesh") else mesh_data[0]
gmsh.finalize()

# ----------------------------------------------------------------------
# 3. Function space and coefficients
# ----------------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)

# spatially varying diffusivity a(x, y)
a_coeff = a_healthy - (a_healthy - a_scar) * ufl.exp(
    -((x[0] - xa) ** 2 + (x[1] - ya) ** 2) / (2.0 * wa ** 2)
)

# reaction coefficient (uniform here). To make it heterogeneous, replace with, e.g.:
#   r_coeff = 0.5 + 2.0 * ufl.exp(-((x[0]-xr)**2 + (x[1]-yr)**2)/(2*wr**2))
r_coeff = fem.Constant(domain, default_scalar_type(r_value))

dt_c    = fem.Constant(domain, default_scalar_type(dt))
theta_c = fem.Constant(domain, default_scalar_type(theta))

# ----------------------------------------------------------------------
# 4. Initial condition
# ----------------------------------------------------------------------
def u0_expr(x):
    return A_stim * np.exp(-((x[0] - x0) ** 2 + (x[1] - y0) ** 2) / (2.0 * s_stim ** 2))

u_n = fem.Function(V, name="u")     # solution at previous step
u_n.interpolate(u0_expr)

uh = fem.Function(V, name="u")      # solution at current step
uh.interpolate(u0_expr)


def prepare_sampling_grid():
    """Build the PINN grid and locate a FEM cell for every masked point."""
    coordinates = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    mask = x_grid**2 + y_grid**2 <= 1.0 + 1.0e-12

    xy = np.column_stack((x_grid[mask], y_grid[mask]))
    points = np.zeros((len(xy), 3), dtype=domain.geometry.x.dtype)
    points[:, :2] = xy

    tdim = domain.topology.dim
    tree = geometry.bb_tree(domain, tdim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidates, points)

    cells = np.full(len(points), -1, dtype=np.int32)
    for point_index in range(len(points)):
        links = colliding_cells.links(point_index)
        if len(links) > 0:
            cells[point_index] = links[0]

    # Gmsh represents the circular boundary by straight edges. A few points
    # on the mathematical unit circle can therefore lie just outside the
    # polygonal mesh. Evaluate those points using the nearest P1 cell.
    missing = np.flatnonzero(cells < 0)
    if len(missing) > 0:
        index_map = domain.topology.index_map(tdim)
        cell_entities = np.arange(
            index_map.size_local + index_map.num_ghosts, dtype=np.int32
        )
        midpoint_tree = geometry.create_midpoint_tree(domain, tdim, cell_entities)
        closest_cells = geometry.compute_closest_entity(
            tree, midpoint_tree, domain, points[missing]
        )
        if np.any(closest_cells < 0):
            raise RuntimeError("Could not locate a FEM cell for every sampling point")
        cells[missing] = closest_cells

    return coordinates, mask, points, cells


def sample_function(function, mask, points, cells):
    """Evaluate a scalar FEM function and restore the masked-grid layout."""
    values = np.asarray(function.eval(points, cells)).reshape(-1).real
    if not np.all(np.isfinite(values)):
        raise RuntimeError("FEM sampling produced a non-finite interior value")
    field = np.full(mask.shape, np.nan, dtype=np.float64)
    field[mask] = values
    return field


if domain.comm.size == 1:
    sample_coordinates, sample_mask, sample_points, sample_cells = prepare_sampling_grid()
    snapshot_steps_float = snapshot_times / dt
    snapshot_steps = np.rint(snapshot_steps_float).astype(np.int64)
    if not np.allclose(snapshot_steps_float, snapshot_steps, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Every FEM snapshot time must align with a time step")
    if snapshot_steps[0] != 0 or snapshot_steps[-1] > num_steps:
        raise RuntimeError("FEM snapshot times must lie within the simulated interval")
    snapshot_index_by_step = {
        int(step): index for index, step in enumerate(snapshot_steps)
    }
    snapshot_fields = [None] * len(snapshot_times)
    snapshot_fields[0] = sample_function(u_n, sample_mask, sample_points, sample_cells)
else:
    snapshot_fields = None
    if domain.comm.rank == 0:
        print("[FEM NPZ export skipped: run in serial to generate fem_solution.npz]")


def save_plot(filename, title, values=None):
    """Save an off-screen top-down plot of the mesh or a nodal field."""
    try:
        import pyvista
        from dolfinx.plot import vtk_mesh

        cells, types, geometry = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, geometry)

        plotter = pyvista.Plotter(off_screen=True, window_size=(1200, 1000))
        plotter.set_background("white")

        if values is None:
            plotter.add_mesh(
                grid,
                color="white",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
            )
        else:
            grid.point_data["u"] = np.asarray(values).real
            grid.set_active_scalars("u")
            plotter.add_mesh(
                grid,
                scalars="u",
                cmap="viridis",
                clim=(0.0, A_stim),
                show_edges=False,
                scalar_bar_args={"title": "u"},
            )

        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.add_text(title, font_size=12, color="black")
        plotter.screenshot(filename)
        plotter.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to save {filename} with PyVista") from exc


if domain.comm.size == 1:
    save_plot("mesh.png", "Finite-element mesh")
    save_plot("u_initial.png", "Initial condition: u(x, 0)", u_n.x.array.copy())
elif domain.comm.rank == 0:
    print("[PNG plotting skipped: run in serial to generate mesh.png, u_initial.png, and u_final.png]")

# ----------------------------------------------------------------------
# 5. Variational forms (theta-scheme)
# ----------------------------------------------------------------------
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

def B(w):   # weak spatial operator  (a grad w, grad v) + (r w, v)
    return a_coeff * ufl.dot(ufl.grad(w), ufl.grad(v)) + r_coeff * w * v

a_form = (u * v + dt_c * theta_c * B(u)) * ufl.dx
L_form = (u_n * v - dt_c * (1.0 - theta_c) * B(u_n)) * ufl.dx

bilinear_form = fem.form(a_form)
linear_form   = fem.form(L_form)

# Bilinear form is SPD (mass term + non-negative reaction), so the pure-Neumann
# system is non-singular: no Dirichlet BCs, no nullspace handling required.
A = assemble_matrix(bilinear_form)   # bcs default to [] (Neumann is natural)
A.assemble()
b = create_vector(V)

# ----------------------------------------------------------------------
# 6. Linear solver  (A is time-independent -> factor once, reuse)
# ----------------------------------------------------------------------
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# ----------------------------------------------------------------------
# 7. Output
# ----------------------------------------------------------------------
xdmf = XDMFFile(domain.comm, "monodomain_disk.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(uh, 0.0)

# ----------------------------------------------------------------------
# 8. Time-stepping
# ----------------------------------------------------------------------
t = 0.0
for n in range(num_steps):
    t += dt

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

    # advance:  u^n <- u^{n+1}
    u_n.x.array[:] = uh.x.array

    if domain.comm.size == 1 and (n + 1) in snapshot_index_by_step:
        snapshot_index = snapshot_index_by_step[n + 1]
        snapshot_fields[snapshot_index] = sample_function(
            uh, sample_mask, sample_points, sample_cells
        )

    # diagnostics + output every 10 steps
    if (n + 1) % 10 == 0:
        umax = domain.comm.allreduce(np.max(np.abs(uh.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"step {n + 1:4d}   t = {t:5.2f}   max|u| = {umax:.4e}")
        xdmf.write_function(uh, t)

xdmf.close()

if domain.comm.size == 1:
    if any(field is None for field in snapshot_fields):
        raise RuntimeError("One or more requested FEM snapshots were not sampled")
    np.savez_compressed(
        "fem_solution.npz",
        x=sample_coordinates,
        y=sample_coordinates,
        mask=sample_mask,
        times=snapshot_times,
        u=np.stack(snapshot_fields),
    )

# ----------------------------------------------------------------------
# 9. Save final-state plot
# ----------------------------------------------------------------------
if domain.comm.size == 1:
    save_plot("u_final.png", f"Final solution: u(x, t={T})", uh.x.array.copy())
