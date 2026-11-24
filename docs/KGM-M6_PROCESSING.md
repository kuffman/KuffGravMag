# KGM-M6 — Survey processing and interpretation

KGM-M6 adds a reproducible profile-processing core to the forward and inversion
workbench. Raw station data remain in each export, and every derived product
records units, parameters, assumptions, source hashes, and output hashes.

## Survey profile CSV, schema version 1

The strict importer accepts UTF-8 CSV with these columns:

| Column | Requirement | Meaning |
|---|---|---|
| `distance_m` | Required, finite, unique | Distance along the profile in metres |
| `elevation_m` | Required, finite | Station elevation in metres, positive upward |
| `gravity_mgal` | Optional | Measured gravity channel in mGal |
| `magnetic_anomaly_nt` | Optional | Measured total-field anomaly in nT |
| `station_time_utc` | Optional | ISO-8601 timestamp with a UTC offset |
| `station_id` | Optional | Station label retained with the source rows |

At least one measured channel must be present and contain a finite value.
Blank channel cells remain missing in their original rows; they are not dropped
or filled with zero. The importer requires metadata for `coordinate_reference`,
`profile_azimuth_deg` in degrees, and `elevation_datum`. The coordinate reference
may be a named projected CRS or an explicitly described local profile frame.
Elevation metadata must identify the datum used by the survey. The importer
records the source filename and SHA-256 hash. The legacy positional CSV import
remains available for older project workflows.

The versioned base-station CSV has exactly `time_utc,total_field_nt` columns.
Times must be strictly increasing UTC-offset timestamps; readings must be
finite. Magnetic diurnal correction is only available when every profile
station time falls within the base-station time span.

## Corrections and separation

For a station height `h` and a declared reduction datum `h0`, the free-air
reduced channel is

```text
g_fa = g_observed + 0.3086 * (h - h0)  mGal
```

The coefficient is the conventional approximate free-air gradient documented
by the [USGS free-air gradient report](https://pubs.usgs.gov/of/1985/0530/report.pdf).
When a positive slab density `rho` in kg/m³ is explicitly supplied, the
infinite-slab reduction is

```text
g_bouguer_slab = g_fa - 0.00004191 * rho * (h - h0)  mGal
```

The coefficient is the USGS Bouguer factor converted from density in g/cm³ to
kg/m³; see the [USGS Bouguer correction factor table](https://pubs.usgs.gov/of/1980/2012/report.pdf).
Both outputs retain the raw gravity channel. These are reductions only. The
application does not calculate normal gravity, a complete free-air or Bouguer
anomaly, terrain, tides, meter drift, latitude, or Earth-curvature corrections.

Magnetic diurnal correction uses a base-station time series and a user-selected
reference time:

```text
T_corrected[i] = T_observed[i] - (B_base(time[i]) - B_base(reference_time))
```

The base value is linearly interpolated without extrapolation. This removes
only the recorded base-station temporal change. No IGRF or other main-field
model is evaluated or subtracted. The processing bundle includes the base
series used by this correction.

Polynomial regional separation fits the user-selected degree (0–5) by
least-squares on profile distance. It returns the fitted regional and residual
side by side. A polynomial fit is a chosen mathematical separation and does not
identify a unique geological background.

## Interpolation and spectral transforms

Linear profile interpolation requires at least two finite stations and a
positive user-selected interval. It uses only the finite input range and does
not extrapolate. Scattered XYZ gridding uses linear interpolation inside the
input convex hull and leaves outside cells missing. A spectral grid must be
complete; missing cells cannot be silently filled before a transform.

The profile Fourier operations use a 1D wavenumber and therefore assume a 2D
potential field invariant along strike. Profile samples must be regular and
finite. The grid transform uses a complete, regular 2D grid and its radial
wavenumber. For either form, upward continuation uses `exp(-|k|h)`, horizontal
derivatives use `i*k`, and the positive-down vertical derivative uses `|k|`.
Analytical-signal amplitude combines the horizontal and vertical derivatives;
tilt is `atan2(vertical derivative down, horizontal-gradient magnitude)` in
degrees. Results are sensitive to sampling, Nyquist limits, and edge handling.

Reflect padding is the default and the result marks the padded-width boundary
band for review. The optional no-padding mode assumes periodic repetition at
the profile ends or all grid edges and can wrap edge effects across the result.
The UI requires explicit interpolation before profile Fourier processing.

## Euler depth estimate

`engine.euler.euler_deconvolution_grid` implements the fixed-structural-index
3D Euler equation over a user-selected regular grid window. Coordinates use
metres and positive-up z; the supplied vertical derivative must use that same
positive-up convention. The linear solve estimates source location and base
level. It reports depth below the observation plane, rank, scaled condition
number, and residual RMS. It rejects underdetermined, rank-deficient,
ill-conditioned, and nonpositive-depth solutions. The structural index is user
supplied; it is not inferred. Euler deconvolution's dependence on structural
index and window choice is described in the [peer-reviewed Euler equation reference](https://doi.org/10.1186/s40517-019-0130-y).

The core Euler solver is independently checked against a synthetic point-source
field with a known homogeneous degree. An interactive Euler grid workflow is
deferred until its window selection and noisy multi-source behavior have a
validated UX. Regular-grid spectral outputs and Euler estimates remain
available through the Python API.

## Projects, UI, and exports

Project JSON schema version 2 stores the raw survey profile, coordinate and
elevation metadata, source provenance, and processing-step manifests. The
loader continues to accept earlier project versions. The desktop menu and web
sidebar expose survey import, free-air and optional slab reduction, magnetic
diurnal correction, polynomial regional separation, regular interpolation,
profile spectral transforms, previews, and processing-bundle download.

The reproducible ZIP contains `profile.csv`, a `manifest.json`, any
regular-profile outputs, and supporting base-station CSVs. It preserves the raw
channels and records output units, correction signs, transform assumptions,
processing parameters, and SHA-256 hashes.

## Validated scope and deferred work

Synthetic tests compare corrections with independently evaluated scalar
formulas, regional fitting with a known orthogonal polynomial-plus-residual
signal, interpolation with linear analytic fields, scattered gridding with a
planar field, Fourier transforms with periodic analytic modes, and Euler
deconvolution with a known point source. These validate numerical behavior for
the declared synthetic inputs; they do not establish accuracy for field data.

Terrain corrections are deferred until a terrain-data format, datum, density,
and independent terrain benchmark are defined. IGRF/main-field corrections are
deferred until a supported reference model, date/location metadata, and
independent field-vector tests are selected. Stable reduction-to-pole or other
low-latitude transforms are deferred; the application preserves the declared
field orientation but does not claim a validated low-inclination stabilization.
These advanced operations are intentionally absent from current controls.
