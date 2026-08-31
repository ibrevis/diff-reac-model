# Nagumo inverse problem

`inverse_problem.py` recovers the two constant coefficients in
`D = diag(a1, a2)` from `observations_coarse.npy`. The data layout is time-major:
1000 post-step times by 5 physical sensors. Sensor coordinates and times are
defined in `params.py`.

The implementation uses the same P1, lumped-mass IMEX Euler FEniCSx model as the
forward notebook. It requires one MPI rank and an environment containing
DOLFINx/PETSc, NumPy, SciPy, Gmsh, and Matplotlib.

Run all forward checks without optimization:

```bash
python Nagumo/inverse_problem.py --preflight-only
```

Run the inverse problem with the documented defaults:

```bash
python Nagumo/inverse_problem.py \
  --observations Nagumo/observations_coarse.npy \
  --a1-init 1.0 --a2-init 1.0 \
  --a1-ref 1.0 --a2-ref 1.0 \
  --lambda-reg 1e-4 \
  --a1-lower 1e-4 --a2-lower 1e-4 \
  --a1-upper 2.0 --a2-upper 2.0
```

The minimized objective is

```text
0.5 * (||vec_C(y_pred - y_obs)||_2^2
       + lambda_reg * ||theta - theta_ref||_2^2).
```

No noise standard deviation is recorded in the repository, so the data residual
is unweighted. The known synthetic values in `params.py` are used only by the
preflight and final reporting, never by the optimizer.

Results and noninteractive diagnostic figures are written to
`Nagumo/inverse_output/` by default. Use `--help` for all bounds, tolerances,
finite-difference, output, verbosity, and preflight options.

Dependency-light tests can be run with:

```bash
python -m unittest -v Nagumo/test_inverse_problem.py
```

Set `NAGUMO_RUN_FEM_TESTS=1` to include the slower FEniCSx repeatability test.
