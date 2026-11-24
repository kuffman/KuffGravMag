# KGM-M5 scientific validation

KGM-M5 adds reusable reference calculations, independent software comparison,
stress tests, deterministic inversion studies, and a reproducible report. The
work validates the numerical behavior of the supported profile and prism
geometries. It does not establish accuracy on field data.

## M5.1 independent reference library

`validation/analytical.py` contains standalone oracles that do not call the
production forward engine:

- A closed-form centered rectangle reference for positive-down gravity in the
  infinite-strike limit.
- Closed-form centered rectangle references for the infinite-strike magnetic
  `dX` and positive-down `dZ` components per unit magnetization.
- A Nagy corner-sum reference for the downward gravity component of a
  rectangular prism.
- Tensor Gauss-Legendre volume integration of gravity and the 3D magnetic dipole
  field as an independent numerical cross-check for prism results.
- Ear-clipped triangle quadrature for gravity and magnetic components of simple
  convex or concave infinite-strike polygons, checked against a concave profile
  body and reversed winding.
- Point-mass and magnetic-dipole asymptotes for far-field checks.

A 2D dyke-style body is represented by the supported cross-section polygon and
strike mode. Spheres and infinite cylinders are excluded as acceptance oracles:
the application models 2D polygons and their uniform finite-strike extrusions,
not sphere or cylinder source primitives.

## M5.2 Harmonica convention mapping

The optional `validation` dependency group pins Harmonica 0.7.0 and bounds
pandas to `>=2.1,<3` because this Harmonica release imports a pandas internal
API removed in pandas 3. The locked comparison exercises its rectangular-prism
gravity and magnetic functions; it does not add Harmonica to the application
runtime. Harmonica documents its
Cartesian frame as easting, northing, and upward, returns gravity in mGal, and
returns magnetic components in nT. Its `g_z` selector is the downward gravity
component. See the [Harmonica prism gravity API](https://www.fatiando.org/harmonica/latest/api/generated/harmonica.prism_gravity.html),
the [Harmonica prism magnetic API](https://www.fatiando.org/harmonica/latest/api/generated/harmonica.prism_magnetic.html),
and the [Harmonica releases](https://github.com/fatiando/harmonica/releases).

The automated comparison uses a 90° profile azimuth so the prism remains
axis-aligned in Harmonica coordinates. In the local frame `(x profile, y
strike, z down)`, the mapping is `E=x`, `N=-y`, and `U=-z`. Strike bounds,
magnetization, and output vectors are transformed with the same basis; the
vertical magnetic component is negated when converting from upward to
positive-down. `dT` is then projected using the declared local field direction.
The adapter rejects non-axis-aligned profile azimuths instead of silently
rotating an axis-aligned prism approximation.

## M5.3 numerical reliability

The regression suite compares finite-prism gravity against the Nagy oracle at
1 mm, 1 m, and 1 km geometry scales; checks points 1 µm above the top surface;
and compares a 20 km-offset result with point-mass and magnetic-dipole
asymptotes. It retains the earlier winding, zero-width, invalid-geometry,
vertex-singularity, symmetry, and independent-volume-integral checks.

The reference and production tolerances are documented alongside each
regression. The principal acceptance tolerances are:

- Nagy prism gravity: `rtol=2e-11`, `atol=2e-11 mGal` for the nominal geometry.
- Scale stress gravity: `rtol=5e-10`, with an absolute tolerance scaled by
  model length.
- Harmonica gravity: `rtol=2e-10`, `atol=2e-11 mGal`.
- Harmonica magnetic components: `rtol=3e-8`, `atol=3e-8 nT`.
- Independent finite-prism volume magnetic reference: `rtol=3e-8`,
  `atol=3e-8 nT` in the existing 32-point baseline tests; M5 adds a 48-point
  quadrature check for the reusable library.
- Far-field point-source/dipole checks: relative residual below `5e-3` at the
  selected 20 km offset; this tests asymptotic behavior and is not a field-data
  accuracy claim.

Observations at finite-prism magnetic edges or vertices remain singular and
raise `ValueError`. Values just outside the body are checked separately from
the singular boundary itself.

## M5.4 inversion noise and non-uniqueness

`validation/inversion_studies.py` uses a fixed seed (`20261124`) and paired
standard-normal noise draws at `0.01`, `0.03`, and `0.1 mGal`, with 32 trials
per level in the committed report. It records density bias and RMSE, local
standard-error averages, empirical 95% interval coverage, and the fraction of
identifiable fits. Tests use a shorter 16-trial run and confirm deterministic
output and increasing error/uncertainty with the paired noise levels.

A second synthetic construction chooses a deeper prism density to match the
shallow model exactly at one station. The full profile then differs, showing
the density-depth tradeoff in a concrete way. The inversion engine's local
linear errors and covariance are diagnostics for the selected model and data;
they do not include geological alternatives, remanence error, demagnetization,
or forward-model discrepancy.

A separate two-observation fit estimates density and two geometry parameters.
The report records optimizer convergence, numerical Jacobian rank, local
identifiability, and whether standard errors are available. Convergence alone
does not make an underdetermined model identifiable.

## M5.5 reproduction and CI

From the repository root:

```sh
uv sync --locked --group validation
uv run --locked --group validation python -m pytest -q
uv run --locked --group validation python -m validation.report --write docs/VALIDATION_REPORT.md
uv run --locked --group validation python -m validation.report --check docs/VALIDATION_REPORT.md
```

The report records the optional library versions, the `uv.lock` SHA-256, fixed
random seed, trial count, numerical comparisons, and limitations. It omits
timestamps so regeneration can be checked byte-for-byte. The
`independent-validation` GitHub Actions job installs the locked optional group,
runs the full suite with Harmonica comparisons enabled, and checks the report.

The reproducible report is [VALIDATION_REPORT.md](VALIDATION_REPORT.md). Its
results are synthetic mathematics and software cross-checks, not a certification
of arbitrary geology or field measurements.
