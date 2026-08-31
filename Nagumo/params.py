"""Parameters for the Nagumo finite-element example."""

# Time integration
T = 5.0
num_steps = 1000
dt = T / num_steps

# Mesh: unit disk with an off-centre circular hole
h = 0.025
mesh_filename = "nagumo_disk_with_hole.msh"
hole_x, hole_y, hole_r = -0.25, 0.0, 0.3

# Anisotropic diffusion tensor D = diag(a1, a2). These values generated the
# synthetic coarse observations and are used only for forward examples and
# post-inversion validation.
a1, a2 = 0.8, 0.6

# Nagumo reaction r(u) = u (u - alpha) (u - 1)
alpha = 0.2

# Initial Gaussian u0 = A0 exp(-|x - x0|^2 / (2 sigma0^2))
A0 = 1.0
sigma0 = 0.12
x0, y0 = 0.4, 0.0

# Observation metadata used by observations_coarse.npy. Row i contains the
# solution after time step i + 1, and columns follow this point ordering.
observation_points = (
    (0.25, 0.25),
    (0.00, 0.50),
    (-0.25, 0.50),
    (-0.50, 0.25),
    (-0.75, 0.00),
)
observation_point_labels = tuple(f"({x:.2f}, {y:.2f})" for x, y in observation_points)
observation_times = tuple(dt * step for step in range(1, num_steps + 1))
