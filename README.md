# KuffGravMag

KuffGravMag is a desktop and web workbench for 2D gravity and magnetic profile modelling. The infinite-strike gravity and magnetic kernels have analytical and independent numerical regression coverage. Finite-strike gravity and magnetics remain approximate, and the inversion engine remains outside the validated scope.

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

The gravity suite covers the positive-down analytical rectangle, an independent triangle area integral, winding invariance, zero and negative density, symmetry, superposition, depth and observation elevation, boundary observations, repeated vertices, and malformed inputs. The magnetic suite covers analytical rectangle components, an independent triangle dipole-area integral, field-angle projection, winding, symmetry, superposition, susceptibility scaling, zero magnetization, and malformed geometry. See [the validation record](docs/BASELINE_AND_VALIDATION.md), [the gravity method note](docs/SCIENTIFIC_METHODS.md), and [the magnetic method note](docs/MAGNETIC_METHOD.md).

## Scope and limitations

The desktop, web, and PDF outputs identify gravity as either 2D infinite-strike (checked against analytical and independent numerical benchmarks) or approximate finite-strike. Finite-strike gravity still uses the previous midpoint multiplier, which is not a validated end correction. The magnetic kernel uses a winding-independent boundary-charge integral for the infinite-strike 2D case; its equations, conventions, numerical benchmarks, and interface limitations are documented in [docs/MAGNETIC_METHOD.md](docs/MAGNETIC_METHOD.md). Finite-strike magnetics remain approximate, and inversion behavior remains follow-up work.

GitHub Actions runs the locked scientific regression suite on pushes and pull requests. The existing Windows packaging workflow remains separate.
