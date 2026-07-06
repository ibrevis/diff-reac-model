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

Tested with FEniCSx / DOLFINx 0.8-0.9  (+ gmsh, petsc4py, mpi4py).
Run serial:      python3 monodomain_disk.py
Run parallel:    mpirun -n 4 python3 monodomain_disk.py
Output:          monodomain_disk.xdmf/.h5  (open in ParaView)  and optional u_final.png
"""

from mpi4py import MPI
import numpy as np
import gmsh

from dolfinx import fem, default_scalar_type
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

    # diagnostics + output every 10 steps
    if (n + 1) % 10 == 0:
        umax = domain.comm.allreduce(np.max(np.abs(uh.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"step {n + 1:4d}   t = {t:5.2f}   max|u| = {umax:.4e}")
        xdmf.write_function(uh, t)

xdmf.close()

# ----------------------------------------------------------------------
# 9. (optional) quick plot of the final state with pyvista
# ----------------------------------------------------------------------
try:
    import pyvista
    from dolfinx.plot import vtk_mesh          # DOLFINx <= 0.6: create_vtk_mesh
    cells, types, geom = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, geom)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=False, cmap="viridis")
    plotter.view_xy()
    plotter.add_text(f"u(x, t={T})", font_size=12)
    if pyvista.OFF_SCREEN:
        plotter.screenshot("u_final.png")
    else:
        plotter.show()
except Exception as exc:
    if MPI.COMM_WORLD.rank == 0:
        print(f"[pyvista plot skipped: {exc}]")
