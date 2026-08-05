import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from dolfinx.plot import vtk_mesh


def _triangles_from_vtk_cells(cells):
    """Extract linear triangles from DOLFINx/VTK cell connectivity."""
    triangles = []
    i = 0
    while i < len(cells):
        n_vertices = cells[i]
        cell = cells[i + 1 : i + 1 + n_vertices]

        if n_vertices == 3:
            triangles.append(cell)
        elif n_vertices == 6:
            v0, v1, v2, e01, e12, e20 = cell
            triangles.extend(
                (
                    [v0, e01, e20],
                    [e01, v1, e12],
                    [e20, e12, v2],
                    [e01, e12, e20],
                )
            )
        else:
            raise ValueError(
                f"plot_solution only supports triangular P1/P2 cells, got {n_vertices} nodes"
            )

        i += n_vertices + 1

    return np.asarray(triangles, dtype=np.int32)

# def plot_solution(u, V, title, save=False, filename=None, display=True): 
def plot_solution(
    u,
    V,
    title,
    save=False,
    filename=None,
    display=True,
    colorbar_label="u",
):   
    """Plot a finite-element solution, with optional saving and display."""
    if save and filename is None:
        raise ValueError("filename must be provided when save=True")

    coords = V.tabulate_dof_coordinates()[:, :2]
    values = u.x.array.real
    cells, _, _ = vtk_mesh(V)
    triangles = _triangles_from_vtk_cells(cells)
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.tricontourf(
        triangulation,
        values,
        levels=30,
        cmap="viridis"
    )

    try:
        fig.colorbar(contour, ax=ax, label=colorbar_label)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        if save:
            fig.savefig(filename, bbox_inches="tight")
        if display:
            plt.show()
    finally:
        plt.close(fig)
