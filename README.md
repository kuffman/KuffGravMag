# KuffGravMag

KuffGravMag is a desktop and web workbench for gravity and magnetic profile modelling. Infinite Strike kernels use validated 2D calculations; Finite Strike mode evaluates the actual extruded 3D polygonal prism with closed-form face integrals. The inversion engine remains outside the validated scope.

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
```

To install every group in one environment, use `uv sync --locked --all-groups`. The legacy `requirements.txt` remains for older workflows; the uv lockfile is the reproducible dependency source.

## Tests

```sh
uv run --locked python -m pytest -q
```

The scientific suite covers infinite-strike gravity and magnetic references, finite-strike rectangular-prism formulas, independent 3D volume integration, all magnetic components, field-angle projection, winding, symmetry, superposition, susceptibility scaling, strike-mode compatibility, and project round-tripping. See [the validation record](docs/BASELINE_AND_VALIDATION.md), [the gravity method note](docs/SCIENTIFIC_METHODS.md), [the magnetic method note](docs/MAGNETIC_METHOD.md), and [the finite-prism method note](docs/FINITE_STRIKE_METHOD.md).

## Scope and limitations

The desktop, web, and PDF outputs distinguish benchmarked Infinite Strike 2D kernels from validated Finite Strike prism integrals. The interfaces expose field intensity, inclination, declination, profile azimuth, remanent orientation, strike mode, and strike bounds. JSON projects preserve the model and observations; PDF reports include the field settings, body properties, strike mode, bounds, and vertices. Finite magnetics assumes uniform magnetization and no self-demagnetization. Inversion behavior remains follow-up work.

GitHub Actions runs the locked scientific regression suite on pushes and pull requests. The existing Windows packaging workflow remains separate.
