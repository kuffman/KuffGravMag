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
| Regression suite after the gravity-kernel change | 19 passed |
| `compileall` for application, engine, and tests | Passed |
| `git diff --check` | Passed |
| Locked CI workflow | Added for pull requests and branch pushes using Python 3.11.18 and uv 0.13.9 |

The suite's closed-form rectangle tolerance is `1e-6` relative. The rectangle, winding, symmetry, superposition, depth/elevation, and geometry regressions were written before the kernel edit. The nonrectangular triangle area-quadrature check and explicit infinite-bound equivalence check were added afterward as additional cross-checks. The triangle uses 64-point-per-axis Gaussian quadrature with `2e-6` relative tolerance. Symmetry, superposition, translation, and winding comparisons use `1e-12` relative/absolute tolerances where appropriate.

All locked dependency groups were installed and the lock check passed. The regression suite was run in that environment. The desktop and web interfaces were not launched, and the Windows packaging workflow was not run locally. The desktop, web, and PDF outputs now state whether the gravity result uses the benchmarked infinite-strike reference or the unvalidated finite-strike midpoint approximation.

## KGM-M2 magnetic validation

The 16-case pre-fix magnetic regression suite had 12 failures and 4 passes. It reproduced the reported polarity and winding defect: for the rectangle `(-100, 200)`, `(100, 200)`, `(100, 400)`, `(-100, 400)` with horizontal remanent magnetization `1 A/m`, the old kernel returned `+87.4675783496 nT`; reversing the winding changed its sign. The analytical positive-convention result is `−87.4675783496 nT` for either winding. The old total-field projection factor was `0.138675` at inclination `60°` and `D−alpha = 60°`, versus the correct profile-horizontal factor `0.25`. In the direct oblique projection regression, the old `dT` was `30.5353 nT`, while projecting its own horizontal and vertical components gives `38.6010 nT`.

After KGM-M2, the rectangle gives `dX = −87.4675783496 nT` for horizontal magnetization and `dZ = +87.4675783496 nT` for vertical magnetization. An independent 72-point-per-axis Gaussian area integral for a nonrectangular triangle gives `(−24.7481385006315, −19.1608788508598) nT`; the edge kernel differs by less than `4e-14 nT` in the measured components. The suite covers winding, field-angle cases, total-field projection, symmetry, superposition, susceptibility scaling, zero magnetization, and geometry errors. The complete locked suite passes: **35 tests** (19 gravity and 16 magnetic).

The full derivation and tolerances are in [SCIENTIFIC_METHODS.md](SCIENTIFIC_METHODS.md) and [MAGNETIC_METHOD.md](MAGNETIC_METHOD.md). These tests validate the synthetic infinite-strike calculation only; finite-strike magnetics still use the old approximate midpoint multiplier, and the interface does not yet expose field declination, profile azimuth, or remanent declination controls.
