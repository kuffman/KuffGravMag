# KuffGravMag Studio

KuffGravMag Studio is a desktop and web workbench for gravity and magnetic survey processing, forward modelling, and inversion. Infinite Strike kernels use validated 2D calculations; Finite Strike mode evaluates the actual extruded 3D polygonal prism with closed-form face integrals. The bounded inversion workflow and its synthetic reliability checks are documented separately; inversion solutions remain potentially non-unique.

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

## Studio workspace (KGM-M7)

The desktop window adds a project explorer and guided actions for observed profiles, survey processing, grid interpretation, body editing, inversion, CSV export, and reporting. A selected polynomial residual can flow from survey processing into the model and inversion views, with missing stations filtered and the elevation sign recorded. The analysis tabs keep the draggable 2D cross-section and profile plots beside an interactive 3D view of finite-strike prisms. Project writes are atomic; a debounced recovery snapshot can restore models, observations, survey metadata, and processing history after an interrupted session. PDF reports include a canonical SHA-256 fingerprint of those project inputs.

The explorer records a source filename and checksum for an inspected regular grid. Grid arrays are not embedded in project JSON yet, and a reopened project requires the original grid file to be selected again. Infinite-strike bodies stay in the 2D section because drawing them as finite prisms would misstate their extent.

## Scope and limitations

The desktop, web, and PDF outputs distinguish benchmarked Infinite Strike 2D kernels from validated Finite Strike prism integrals. The interfaces expose field intensity, inclination, declination, profile azimuth, remanent orientation, strike mode, and strike bounds. Project JSON schema version 2 preserves model settings, observations, optional raw survey data, and processing-step manifests; earlier project versions remain loadable. PDF reports include field settings, body properties, strike mode, bounds, vertices, processing records, and the canonical input fingerprint. Finite magnetics assumes uniform magnetization and no self-demagnetization. Inversion uncertainty is a local approximation, and different body properties or depths can explain similar potential-field data.

The Academic & Research Laboratory adds interactive synthetic gravity and magnetic sphere, infinite-cylinder, and infinite-strike dyke experiments, seeded noise, sphere density-depth fits and uncertainty diagnostics, and reproducible CSV export. The Ghana-relevant coordinate anchor is explicitly synthetic and does not represent field data or a geological interpretation; equations, assumptions, and deferred scope are documented in the [KGM-M8 laboratory guide](docs/KGM-M8_ACADEMIC_RESEARCH_LAB.md).

Terrain correction, IGRF-14 evaluation, stabilized magnetic transforms, Euler window sweeps, spectral depth estimates, and regular-grid interpretation are available through the validated processing engine and grid tools. The legacy profile panel still lacks station map coordinates and its magnetic input is already an anomaly, so it does not apply terrain or main-field corrections to that channel. CRS conversion, basemaps, portable project bundles for spatial grids, 3D vertex editing, and desktop/web workflow parity remain deferred.

GitHub Actions runs the locked scientific suite, optional Harmonica comparison, and reproducible report check on pull requests and branch pushes. The Windows packaging workflow also builds the application.

## M6 roadmap

KGM-M6's staged status and deferred work are in [docs/KGM-M6_PLAN.md](docs/KGM-M6_PLAN.md). The validated profile-processing core is under review; additional low-latitude magnetic methods and interactive Euler interpretation require further independent validation before implementation.
