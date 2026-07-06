from dolfinx import fem, geometry, default_scalar_type
# from dolfinx.io import gmsh as gmshio, XDMFFile
# from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import ufl

import matplotlib.pyplot as plt

def plot_solution(u,V, title):
    coords = V.tabulate_dof_coordinates()[:, :2]
    values = u.x.array.real

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.tricontourf(
        coords[:, 0], coords[:, 1], values,
        levels=30,
        cmap="viridis"
    )

    fig.colorbar(contour, ax=ax, label="u")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.show()