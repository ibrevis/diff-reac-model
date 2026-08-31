"""Parameters for the Nagumo finite-element example."""

# Time integration
T = 5.0
num_steps = 1000 #500
dt = T / num_steps

# Mesh: unit disk with an off-centre circular hole
h = 0.025
mesh_filename = "nagumo_disk_with_hole.msh"
hole_x, hole_y, hole_r = -0.25, 0.0, 0.3

# Anisotropic diffusion tensor D = diag(a1, a2)
a1, a2 = 0.8, 0.6

# Nagumo reaction r(u) = u (u - alpha) (u - 1)
alpha = 0.2

# Initial Gaussian u0 = A0 exp(-|x - x0|^2 / (2 sigma0^2))
A0 = 1.0
sigma0 = 0.12
x0, y0 = 0.4, 0.0
