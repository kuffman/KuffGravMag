# KuffGravMag KGM-M5 scientific validation report

Generated deterministically by `uv run --locked --group validation python -m validation.report --write docs/VALIDATION_REPORT.md`.
The report includes no wall-clock timestamp; the seed, trial count, dependency pin, and lockfile digest identify the run.

## Provenance

- Python requirement: `>=3.11,<3.12`
- NumPy: `2.4.6`; SciPy: `1.17.1`; Pandas: `2.3.3`; Harmonica: `0.7.0`
- Harmonica is an optional validation dependency pinned to `0.7.0`; it is not a runtime application dependency.
- `uv.lock` SHA-256: `adaaa1bce8ad70af834dd19d80f01a788186223ca09cb514636c4a518b189019`
- Inversion noise ensemble: seed `20261124`, `32` paired trials per level.

## M5.1 analytical and independent references

- Centered infinite-strike rectangle analytical gravity: `0.887024015 mGal`.
- Maximum finite-prism gravity residual against the independent Nagy corner sum: `4.552e-15 mGal`.
- Finite-prism gravity residual against independent 48-point tensor volume quadrature: `2.109e-15 mGal`.
- Maximum finite-prism magnetic component residual against independent 48-point tensor volume quadrature: `1.791e-12 nT`.
- Concave infinite-strike polygon residual against 64-point-per-axis area quadrature: gravity `5.551e-16 mGal`; maximum magnetic component `1.137e-13 nT`.

## M5.2 Harmonica comparison

- Maximum downward gravity residual: `1.554e-15 mGal`.
- Maximum `dX`, `dY`, or `dZ` residual: `7.440e-08 nT`.
- Comparison uses Harmonica 0.7.0 prism models. KuffGravMag local coordinates map to easting/northing/upward by a 90° eastward profile azimuth, reversed upward/downward signs, and transformed strike bounds and magnetization. Both packages report magnetic components in nT; gravity is compared in mGal with the downward component selected.

## M5.3 numerical stress results

- Maximum gravity residual across 1 mm, 1 m, and 1 km geometry scales: `5.230e-12 mGal`.
- Near-face magnetic components remain finite at a 1 µm separation: `true`.
- At a 20 km observation offset, finite-prism gravity differs from its point-mass asymptote by `1.874e-04` relative; the magnetic field differs from its dipole asymptote by `3.747e-04` relative.
- The test suite also exercises winding reversal, malformed polygons, zero-width bodies, explicit singular magnetic vertices, and locked regression cases.

## M5.4 inversion reliability

Fixed-geometry density is recovered from synthetic profiles with paired standardized noise at three stated standard deviations. Uncertainty is the inversion engine's local linear 95% approximation, not geological uncertainty.

| Noise σ (mGal) | RMSE density (kg/m³) | Mean local standard error (kg/m³) | 95% interval coverage | Identifiable fits |
|---:|---:|---:|---:|---:|
| 0.01 | 0.914 | 1.088 | 0.938 | 1.000 |
| 0.03 | 2.742 | 3.264 | 0.938 | 1.000 |
| 0.10 | 9.139 | 10.879 | 0.938 | 1.000 |

A shallow `500 kg/m³` body with top depth `200 m` and a deeper `748.1 kg/m³` body with top depth `350 m` match one station to relative residual `0.000e+00`. Across `101` profile stations their relative RMS difference is `0.229`. This is a concrete density-depth tradeoff, not a unique geological interpretation.
A separate fit uses `2` observations for `3` density/geometry parameters. Its optimizer converges (`true`), but the data Jacobian rank is `1/3`; locally identifiable is `false`, and standard errors are available is `false`.

## Tolerances and limitations

- Analytical rectangle/prism gravity checks use relative tolerances from `2e-11` to `5e-10`, with absolute tolerances scaled to the modeled geometry. Harmonica gravity comparison uses `rtol=2e-10`, `atol=2e-11 mGal`; magnetic component comparison uses `rtol=3e-8`, `atol=3e-8 nT`.
- The local profile-to-global transform is axis aligned only at 90° profile-azimuth increments; the Harmonica regression intentionally uses 90° and does not claim validation at arbitrary azimuths.
- Rectangular-prism references validate the finite 3D geometry supported by the engine. Spheres and infinite cylinders are not production body geometries and are not used as acceptance oracles. A 2D rectangular section is the supported dyke-style idealization.
- Synthetic tests validate equations, conventions, numerical behavior, and the inversion workflow on controlled inputs. They do not validate field-data accuracy, remanence uncertainty, self-demagnetization, arbitrary 3D geology, or uniqueness of potential-field inversion.
