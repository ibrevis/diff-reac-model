"""Mesh generation utilities for the disk-with-hole FEM domain."""

import gmsh
from mpi4py import MPI

from dolfinx.io import gmsh as gmshio

from params import h, hole_r, hole_x, hole_y, mesh_filename


def generate_disk_with_hole_mesh(
    mesh_comm=MPI.COMM_WORLD,
    model_rank=0,
    gdim=2,
    mesh_size=h,
    hole_center=(hole_x, hole_y),
    hole_radius=hole_r,
    filename=mesh_filename,
):
    """Generate a unit disk mesh with one circular hole and return a DOLFINx mesh."""
    gmsh.initialize()
    try:
        if mesh_comm.rank == model_rank:
            gmsh.model.add("disk_with_hole")
            outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
            hole = gmsh.model.occ.addDisk(
                hole_center[0],
                hole_center[1],
                0.0,
                hole_radius,
                hole_radius,
            )
            domain_surfaces, _ = gmsh.model.occ.cut(
                [(2, outer)],
                [(2, hole)],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            surface_tags = [tag for dim, tag in domain_surfaces if dim == 2]
            gmsh.model.addPhysicalGroup(2, surface_tags, tag=1)
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
            gmsh.model.mesh.generate(gdim)
            gmsh.write(filename)

        mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        return mesh_data.mesh if hasattr(mesh_data, "mesh") else mesh_data[0]
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    generate_disk_with_hole_mesh()
