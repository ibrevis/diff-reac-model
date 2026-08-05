"""Shared parameters for the FEM monodomain examples."""

import numpy as np

# Time integration
T = 5.0          # final time (~ 5 membrane time constants, since r ~ 1)
num_steps = 500  # number of time steps
dt = T / num_steps
theta = 1.0      # 1.0 = backward Euler, 0.5 = Crank-Nicolson

# Mesh
h = 0.025
mesh_filename = "monodomain_disk_with_hole.msh"

# Initial depolarization: Gaussian bump u0 = A * exp(-|x - x0|^2 / (2 s^2))
A_stim, s_stim = 1.0, 0.12
x0, y0 = 0.4, 0.0

# Diffusivity a(x): healthy tissue with a low-conductivity "scar" patch.
a_healthy, a_scar = 0.1, 0.01
xa, ya, wa = -0.3, 0.0, 0.2

# Optional anisotropic diffusivity values used by the notebook variant.
a1, a2 = 0.8, 0.6

# Reaction r(x): uniform leak / repolarization rate.
r_value = 1.0

# Regular-grid snapshots matching the defaults in PINN-code.py.
grid_size = 201
snapshot_times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

# Circular hole inside the unit disk.
hole_x, hole_y, hole_r = -0.25, 0.0, 0.3
