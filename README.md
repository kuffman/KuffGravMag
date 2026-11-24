# KuffGravMag

KuffGravMag is a desktop and web workbench for gravity and magnetic profile modelling and processing. Infinite Strike kernels use validated 2D calculations; Finite Strike mode evaluates the actual extruded 3D polygonal prism with closed-form face integrals. The bounded inversion workflow and its synthetic reliability checks are documented separately; inversion solutions remain potentially non-unique.

## Reproducible Python setup

The project uses [uv](https://docs.astral.sh/uv/) and pins Python 3.11.18 in `.python-version`. `uv.lock` records exact dependency versions for supported platforms.

Install uv, then run from the repository root:

```sh
uv python install 3.11.18
uv sync --locked
```

The default environment includes NumPy and pytest for the scientific engine tests. Install the front-end dependencies when needed:

```sh
# Streamlit web interface
uv sync --locked --group web
uv run --locked streamlit run app_web.py

# PyQt desktop interface
uv sync --locked --group desktop
uv run --locked python app.py

# Windows packaging tools, in addition to the desktop group
uv sync --locked --group desktop --group build

# Independent scientific reference comparisons and report generation
uv sync --locked --group validation
uv run --locked --group validation python -m validation.report --check docs/VALIDATION_REPORT.md
```

To install every group in one environment, use `uv sync --locked --all-groups`. The legacy `requirements.txt` remains for older workflows; the uv lockfile is the reproducible dependency source.

## Tests

```sh
uv run --locked python -m pytest -q
```

The scientific suite covers infinite-strike gravity and magnetic references, finite-strike rectangular-prism formulas, independent 3D volume integration, Harmonica comparisons, numerical stress, and synthetic inversion noise, tradeoffs, and uncertainty. See the [KGM-M5 validation method](docs/SCIENTIFIC_VALIDATION.md), [reproducible validation report](docs/VALIDATION_REPORT.md), [baseline record](docs/BASELINE_AND_VALIDATION.md), [gravity method note](docs/SCIENTIFIC_METHODS.md), [magnetic method note](docs/MAGNETIC_METHOD.md), [finite-prism method note](docs/FINITE_STRIKE_METHOD.md), and [inversion method note](docs/INVERSION_METHOD.md).

## Survey processing (KGM-M6)

The desktop File menu and web sidebar accept a versioned profile CSV with explicit distance, elevation, units, coordinate reference, azimuth, and elevation datum. Processing includes free-air and optional infinite Bouguer-slab reduction, base-station diurnal correction, polynomial regional/residual separation, linear interpolation, profile spectral transforms, and reproducible ZIP export. Grid transforms and conditioned Euler depth estimation are available through the scientific engine API. Input requirements, equations, sign conventions, edge effects, and omissions are in the [KGM-M6 processing guide](docs/KGM-M6_PROCESSING.md).

## Scope and limitations

The desktop, web, and PDF outputs distinguish benchmarked Infinite Strike 2D kernels from validated Finite Strike prism integrals. The interfaces expose field intensity, inclination, declination, profile azimuth, remanent orientation, strike mode, and strike bounds. Project JSON schema version 2 preserves model settings, observations, optional raw survey data, and processing-step manifests; earlier project versions remain loadable. PDF reports include field settings, body properties, strike mode, bounds, and vertices. Finite magnetics assumes uniform magnetization and no self-demagnetization. Inversion uncertainty is a local approximation, and different body properties or depths can explain similar potential-field data.

Terrain corrections, IGRF/main-field removal, and stabilized low-latitude transformations are not implemented. The profile correction controls do not claim a complete free-air or Bouguer anomaly. Euler UI support and noisy multi-source window behavior are deferred; the conditioned core solver is available through Python and is tested against an analytic point-source field.

GitHub Actions runs the locked scientific suite, optional Harmonica comparison, and reproducible report check on pull requests and branch pushes. The Windows packaging workflow also builds the application.

## M6 roadmap

KGM-M6's staged status and deferred work are in [docs/KGM-M6_PLAN.md](docs/KGM-M6_PLAN.md). The validated profile-processing core is under review; additional low-latitude magnetic methods and interactive Euler interpretation require further independent validation before implementation.
