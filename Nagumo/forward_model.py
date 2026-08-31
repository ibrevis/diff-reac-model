"""Reusable FEniCSx forward model for the anisotropic Nagumo equation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector

try:
    from . import params
except ImportError:  # Support ``python Nagumo/forward_model.py`` style imports.
    import params  # type: ignore[no-redef]


def _mesh_generator():
    """Import the repository's existing mesh generator from either run location."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from mesh_generator import generate_disk_with_hole_mesh

    return generate_disk_with_hole_mesh


class NagumoForwardModel:
    """Deterministic parameter-to-observation map for the Nagumo FEM model.

    The implementation deliberately supports one MPI rank. Point ownership is
    therefore unambiguous and the returned observation array is local NumPy data.
    """

    def __init__(
        self,
        observation_points: Sequence[Sequence[float]] = params.observation_points,
        *,
        comm=MPI.COMM_WORLD,
        verbose: bool = False,
    ) -> None:
        if comm.size != 1:
            raise RuntimeError(
                "NagumoForwardModel currently requires exactly one MPI rank; "
                f"received {comm.size}. Run without mpirun or with 'mpirun -n 1'."
            )

        self.comm = comm
        self.verbose = verbose
        self._points_xy = self._validate_points(observation_points)
        self.points = np.zeros((len(self._points_xy), 3), dtype=np.float64)
        self.points[:, :2] = self._points_xy

        mesh_path = Path(__file__).resolve().parent / params.mesh_filename
        generate_mesh = _mesh_generator()
        self.domain = generate_mesh(
            comm,
            0,
            mesh_size=params.h,
            hole_center=(params.hole_x, params.hole_y),
            hole_radius=params.hole_r,
            filename=str(mesh_path),
        )
        if self.domain.geometry.dim != 2:
            raise RuntimeError(
                f"Expected a two-dimensional mesh, got gdim={self.domain.geometry.dim}"
            )

        self.points = self.points.astype(self.domain.geometry.x.dtype, copy=False)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.u_n = fem.Function(self.V, name="u_previous")
        self.uh = fem.Function(self.V, name="u")
        self.explicit_state = fem.Function(self.V, name="explicit_state")
        self.u_n.interpolate(self._initial_condition)
        self.u_n.x.scatter_forward()
        self._initial_values = self.u_n.x.array.copy()
        self._reset_state()

        self._num_owned_dofs = (
            self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        )
        self._build_point_cell_map()
        self._build_variational_problem()
        self.last_initial_reset_verified = False

    @staticmethod
    def _validate_points(points: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] not in (2, 3):
            raise ValueError(
                "observation_points must have shape (n_sensors, 2) or "
                f"(n_sensors, 3); got {values.shape}"
            )
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("observation_points must be non-empty and finite")
        if values.shape[1] == 3 and not np.allclose(values[:, 2], 0.0):
            raise ValueError("Observation points must lie in the z=0 plane")
        return values[:, :2].copy()

    @staticmethod
    def _initial_condition(x: np.ndarray) -> np.ndarray:
        distance_squared = (x[0] - params.x0) ** 2 + (x[1] - params.y0) ** 2
        return params.A0 * np.exp(-distance_squared / (2.0 * params.sigma0**2))

    def _reset_state(self) -> None:
        self.u_n.x.array[:] = self._initial_values
        self.u_n.x.scatter_forward()
        self.uh.x.array[:] = self._initial_values
        self.uh.x.scatter_forward()
        self.explicit_state.x.array[:] = 0.0
        self.explicit_state.x.scatter_forward()

    def _build_point_cell_map(self) -> None:
        tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        candidates = geometry.compute_collisions_points(tree, self.points)
        collisions = geometry.compute_colliding_cells(
            self.domain, candidates, self.points
        )
        self.point_cells = np.full(len(self.points), -1, dtype=np.int32)
        for index in range(len(self.points)):
            cells = collisions.links(index)
            if len(cells):
                self.point_cells[index] = cells[0]

        missing = np.flatnonzero(self.point_cells < 0)
        if len(missing):
            coordinates = ", ".join(
                f"({self._points_xy[i, 0]:.16g}, {self._points_xy[i, 1]:.16g})"
                for i in missing
            )
            raise ValueError(
                "The following observation point(s) could not be assigned to a "
                f"mesh cell: {coordinates}"
            )

    def _build_variational_problem(self) -> None:
        trial = ufl.TrialFunction(self.V)
        test = ufl.TestFunction(self.V)

        self.a1_constant = fem.Constant(self.domain, PETSc.ScalarType(params.a1))
        self.a2_constant = fem.Constant(self.domain, PETSc.ScalarType(params.a2))
        diffusion = (
            self.a1_constant * ufl.Dx(trial, 0) * ufl.Dx(test, 0)
            + self.a2_constant * ufl.Dx(trial, 1) * ufl.Dx(test, 1)
        )
        self.diffusion_form = fem.form(params.dt * diffusion * ufl.dx)

        mass_weights_form = fem.form(test * ufl.dx)
        self.m_lumped = create_vector(self.V)
        with self.m_lumped.localForm() as local_mass:
            local_mass.set(0.0)
        assemble_vector(self.m_lumped, mass_weights_form)
        self.m_lumped.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        local_min = np.min(self.m_lumped.array[: self._num_owned_dofs])
        global_min = self.comm.allreduce(local_min, op=MPI.MIN)
        if global_min <= 0.0:
            raise RuntimeError(f"Non-positive lumped mass detected: {global_min}")

        self.A = assemble_matrix(self.diffusion_form)
        self.A.assemble()
        self._add_lumped_mass_to_operator()
        self.b = create_vector(self.V)

        self.solver = PETSc.KSP().create(self.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.A)

    def _add_lumped_mass_to_operator(self) -> None:
        self.A.setDiagonal(self.m_lumped, PETSc.InsertMode.ADD_VALUES)
        self.A.assemble()

    def _assemble_operator(self, a1: float, a2: float) -> None:
        self.a1_constant.value = PETSc.ScalarType(a1)
        self.a2_constant.value = PETSc.ScalarType(a2)
        self.A.zeroEntries()
        assemble_matrix(self.A, self.diffusion_form)
        self.A.assemble()
        self._add_lumped_mass_to_operator()
        self.solver.setOperators(self.A)

    def _sensor_values(self, function: fem.Function) -> np.ndarray:
        values = np.asarray(function.eval(self.points, self.point_cells)).reshape(-1)
        if values.shape != (len(self.points),):
            raise RuntimeError(
                "Unexpected point-evaluation shape: "
                f"{values.shape}, expected {(len(self.points),)}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Non-finite finite-element sensor values detected")
        return values.copy()

    @staticmethod
    def _prepare_times(observation_times: Sequence[float]) -> np.ndarray:
        times = np.asarray(observation_times, dtype=np.float64).copy()
        if times.ndim != 1 or len(times) == 0:
            raise ValueError("observation_times must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(times)):
            raise ValueError("observation_times contains non-finite values")
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, params.T)
        if np.any(times < -tolerance) or np.any(times > params.T + tolerance):
            raise ValueError(f"observation_times must lie in [0, {params.T}]")

        times = np.clip(times, 0.0, params.T)
        nearest_steps = np.rint(times / params.dt)
        nearest_times = nearest_steps * params.dt
        close = np.abs(times - nearest_times) <= tolerance
        times[close] = nearest_times[close]
        return times

    def solve(
        self,
        a1: float,
        a2: float,
        observation_times: Sequence[float] = params.observation_times,
    ) -> np.ndarray:
        """Solve from t=0 and return ``(n_times, n_sensors)`` observations."""
        coefficients = np.asarray((a1, a2), dtype=np.float64)
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients <= 0.0):
            raise ValueError(
                f"Diffusion coefficients must be finite and positive; got {coefficients}"
            )
        times = self._prepare_times(observation_times)

        self._reset_state()
        self.last_initial_reset_verified = bool(
            np.array_equal(self.u_n.x.array, self._initial_values)
        )
        if not self.last_initial_reset_verified:
            raise RuntimeError("Failed to reset the forward solution to its initial state")
        self._assemble_operator(float(a1), float(a2))

        grid_values = np.empty((params.num_steps + 1, len(self.points)))
        grid_values[0] = self._sensor_values(self.u_n)

        for step in range(1, params.num_steps + 1):
            u_values = self.u_n.x.array
            reaction = u_values * (u_values - params.alpha) * (u_values - 1.0)
            self.explicit_state.x.array[:] = u_values - params.dt * reaction
            self.explicit_state.x.scatter_forward()
            self.b.pointwiseMult(self.m_lumped, self.explicit_state.x.petsc_vec)

            self.solver.solve(self.b, self.uh.x.petsc_vec)
            reason = int(self.solver.getConvergedReason())
            if reason <= 0:
                raise RuntimeError(
                    f"PETSc solve failed at step {step} for a1={a1}, a2={a2}; "
                    f"convergence reason={reason}"
                )
            self.uh.x.scatter_forward()
            local_values = self.uh.x.array[: self._num_owned_dofs]
            if not np.all(np.isfinite(local_values)):
                raise FloatingPointError(
                    f"Non-finite solution at step {step} for a1={a1}, a2={a2}"
                )

            grid_values[step] = self._sensor_values(self.uh)
            self.u_n.x.array[:] = self.uh.x.array
            self.u_n.x.scatter_forward()

            if self.verbose and (step == 1 or step % 50 == 0):
                print(
                    f"step {step:4d}/{params.num_steps} "
                    f"t={step * params.dt:.4f} "
                    f"min(u)={np.min(local_values):.6e} "
                    f"max(u)={np.max(local_values):.6e}"
                )

        grid_times = np.arange(params.num_steps + 1, dtype=np.float64) * params.dt
        predicted = np.column_stack(
            [np.interp(times, grid_times, grid_values[:, i]) for i in range(len(self.points))]
        )
        expected_shape = (len(times), len(self.points))
        if predicted.shape != expected_shape:
            raise RuntimeError(
                f"Forward output has shape {predicted.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(predicted)):
            raise FloatingPointError("Forward observations contain non-finite values")
        return predicted


def solve_forward(
    a1: float,
    a2: float,
    observation_points: Sequence[Sequence[float]],
    observation_times: Sequence[float],
) -> np.ndarray:
    """Convenience wrapper around :class:`NagumoForwardModel`."""
    model = NagumoForwardModel(observation_points)
    return model.solve(a1, a2, observation_times)


__all__ = ["NagumoForwardModel", "solve_forward"]
