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

All locked dependency groups were installed and the lock check passed. At the M0/M1 stage, the desktop and web interfaces had not yet been launched locally; the Windows packaging workflow was not run locally. KGM-M3 later adds UI and report smoke checks described below.

## KGM-M2 magnetic validation

The 17-case pre-fix magnetic regression suite had 13 failures and 4 passes. It reproduced the reported polarity and winding defect: for the rectangle `(-100, 200)`, `(100, 200)`, `(100, 400)`, `(-100, 400)` with horizontal remanent magnetization `1 A/m`, the old kernel returned `+87.4675783496 nT`; reversing the winding changed its sign. The analytical positive-convention result is `−87.4675783496 nT` for either winding. The old total-field projection factor was `0.138675` at inclination `60°` and `D−alpha = 60°`, versus the correct profile-horizontal factor `0.25`. In the direct oblique projection regression, the old `dT` was `30.5353 nT`, while projecting its own horizontal and vertical components gives `38.6010 nT`.

After KGM-M2, the rectangle gives `dX = −87.4675783496 nT` for horizontal magnetization and `dZ = +87.4675783496 nT` for vertical magnetization. An independent 72-point-per-axis Gaussian area integral for a nonrectangular triangle gives `(−24.7481385006315, −19.1608788508598) nT`; the edge kernel differs by less than `4e-14 nT` in the measured components. The suite covers winding, field-angle cases, total-field projection, symmetry, edge principal values, superposition, susceptibility scaling, zero magnetization, and geometry errors. The complete locked suite passes: **36 tests** (19 gravity and 17 magnetic).

The full infinite-strike derivation and tolerances are in [SCIENTIFIC_METHODS.md](SCIENTIFIC_METHODS.md) and [MAGNETIC_METHOD.md](MAGNETIC_METHOD.md). KGM-M3 adds exact finite-prism face integrals, angle controls, explicit strike modes, JSON projects, and full-vector magnetic validation as summarized below.

## KGM-M3 finite-strike validation

The regression suite was written before the computational engine changes. Its first run against the M2 branch had **9 failures**, beginning with the missing explicit `strike_mode` and full-component APIs. Against the prior midpoint implementation, an independent volume integral reproduced an asymmetric-box gravity result of `0.6990738981 mGal` where the midpoint correction returned `0.60848391 mGal` (low by `0.09058999 mGal`, **12.96%**). For `M=(0.8,-0.35,0.42) A/m`, the independent finite-prism magnetic reference at the profile origin was `dB=(-54.96475481, 12.92947538, 36.49399496) nT`; the old calculation omitted `dY` and returned `dX=-54.77665896 nT`, `dZ=39.33275916 nT`.

The corrected face-integral engine matches the closed-form rectangular-prism gravity corner sum and independent 32-point-per-axis 3D volume integration. The magnetic `dX`, `dY`, `dZ`, and projected `dT` match independent direct dipole volume integration for equatorial, low-inclination, and oblique profiles. Additional regressions cover symmetric/asymmetric off-profile strike bounds, winding invariance, symmetry, superposition, susceptibility scaling, zero magnetization, physical zero width versus the historical zero-bound sentinel, and convergence to infinite strike at `±1,000,000 m`. The equations, assumptions, and numerical tolerances are in [FINITE_STRIKE_METHOD.md](FINITE_STRIKE_METHOD.md).

KGM-M3 also adds explicit Infinite Strike/Finite Strike controls, field and remanent orientation controls, JSON project load/save with legacy mode resolution, and PDF metadata for the field, body properties, strike bounds, and polygon vertices. The inversion engine was not modified.

Final local verification used the locked Python 3.11.18 environment. `uv lock --check`, `uv sync --locked --all-groups`, and the CI-style `uv sync --locked` all passed. The full scientific suite passed in both dependency configurations: **50 passed** after each sync. `compileall` and `git diff --check` passed.

The web application completed both a Streamlit `AppTest` run with no exceptions and a live browser smoke check. The finite-strike status and controls rendered, and the PDF action generated a 246,433-byte report. Decoded report text contains the finite-prism validation statement, field settings, body density/susceptibility/remanence, strike mode and bounds, and polygon vertices. The desktop GUI smoke check could not complete on this Mac: `uv run --locked python app.py` exits during Qt startup because the PyQt6 Cocoa platform plugin cannot be loaded. Explicit plugin-path and offscreen startup attempts did not resolve it. The Windows packaging workflow was not run locally.

PR #1 and PR #2 remain open and unmerged. Their latest reviewed heads (`b3b6e87d015245a6ea782835853d83104d89626c` and `243a2917b2a0bb9f708616ec8fdb2276360a9cff`) each have a completed successful Scientific regression tests GitHub Actions run. KGM-M3 is submitted separately against the KGM-M2 branch; its pull-request CI result is recorded in the PR. The original checkout remains at baseline revision `aad19b48bd105e2656aaeb18c82dd9b7e697e02a` with its pre-existing local edits intact.
