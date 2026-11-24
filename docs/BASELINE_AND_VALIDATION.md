# KGM-M0 baseline and validation record

## Repository baseline

- Repository: `kuffman/KuffGravMag`
- Baseline branch: `main`
- Baseline revision: `aad19b48bd105e2656aaeb18c82dd9b7e697e02a`
- Development branch: `fix/gravity-analytic-kernel`
- The pre-existing local `main` working tree contained user edits. The development branch was created from the committed baseline in an isolated copy; the original checkout and its edits were left untouched.
- The baseline repository had no automated test suite or test-runner configuration. Its Windows workflow built the desktop executable but did not run scientific tests.

## Reproduced gravity discrepancy

The reproduction uses a rectangular body with vertices `(-100, 200)`, `(100, 200)`, `(100, 400)`, and `(-100, 400)` metres, a density contrast of `500 kg/m³`, and an observation at `(0, 0)` metres. Coordinates use positive depth. The legacy `y_min = y_max = 0` settings bypass the old finite-strike multiplier and expose the 2D kernel directly.

At revision `aad19b48bd105e2656aaeb18c82dd9b7e697e02a`, the gravity kernel returned **−0.2918924291 mGal**. Direct integration of the infinite-strike area kernel gives **+0.8870240147 mGal**, positive downward. This matches the discrepancy reported in the earlier review.

The analytical reference and units are derived in [SCIENTIFIC_METHODS.md](SCIENTIFIC_METHODS.md). The tests calculate the reference from the closed form rather than hard-coding the rounded display value.

## Environment

- Python: `3.11.18` (`.python-version`)
- uv used for the development environment: `0.13.9`
- Locked core test packages: NumPy `2.4.6`, pytest `8.4.2`
- `pyproject.toml` defines app dependency groups; `uv.lock` pins their resolved versions and transitive dependencies.

Recreate the core environment and run the suite from the repository root:

```sh
uv python install 3.11.18
uv sync --locked
uv run --locked python -m pytest -q
```

Use `uv sync --locked --all-groups` to install the web, desktop, build, and development groups together.

## Checks

| Check | Baseline / result |
|---|---|
| Exact benchmark before the kernel change | Failed as expected: −0.2918924291 mGal instead of +0.8870240147 mGal |
| Regression suite before the kernel change | 14 failed, 2 passed; the failures exposed the benchmark, invalid-input handling, edge observations, and explicit infinite-strike bounds |
| Regression suite after the kernel change | 18 passed |
| `compileall` for application, engine, and tests | Passed |
| `git diff --check` | Passed |

The suite's closed-form rectangle tolerance is `1e-6` relative. The rectangle, winding, symmetry, superposition, depth/elevation, and geometry regressions were written before the kernel edit. The nonrectangular triangle area-quadrature check and explicit infinite-bound equivalence check were added afterward as additional cross-checks. The triangle uses 64-point-per-axis Gaussian quadrature with `2e-6` relative tolerance. Symmetry, superposition, translation, and winding comparisons use `1e-12` relative/absolute tolerances where appropriate.

Only the core test group was installed for these checks. The desktop and web interfaces were not launched, and the Windows packaging workflow was not run locally.
