# KuffGravMag

KuffGravMag is a desktop and web workbench for 2D gravity and magnetic profile modelling. The gravity kernel has analytical and numerical regression coverage; the magnetic and inversion calculations remain unvalidated.

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

The suite covers the positive-down analytical rectangle, an independent triangle area integral, winding invariance, zero and negative density, symmetry, superposition, depth and observation elevation, boundary observations, repeated vertices, and malformed inputs. See [the validation record](docs/BASELINE_AND_VALIDATION.md) and [the scientific method note](docs/SCIENTIFIC_METHODS.md).

## Scope and limitations

This change corrects the 2D infinite-strike gravity edge integral and its polygon handling. Finite-strike gravity still uses the previous midpoint multiplier, which is not a validated end correction. Magnetic polarity, field projection, and inversion behavior are unchanged and remain follow-up work.
