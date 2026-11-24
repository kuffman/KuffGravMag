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
field with a known homogeneous degree. The grid interpretation workflow lets
the user select an odd square window in cells, stride, one or more structural
indices (0–5), maximum scaled condition number, and optional maximum RMS
residual. Failed and filtered windows remain in the result set with their
status. Structural index zero omits the base-level parameter because the
zero-index equation cannot identify it. Estimates from overlapping windows
are not independent observations and do not provide geological confidence
intervals.

The same grid workflow fits a band-limited radial power spectrum using
`log(P) = intercept - 2 * depth * k`. It reports the selected wavenumber band,
the least-squares slope standard error, and R². The standard error describes
only the selected linear fit; it excludes sampling, model, and geological
uncertainty. The method estimates a source-population depth trend and does not
resolve the depth of an individual body.

## DEM residual-terrain correction

`engine.terrain.TerrainDEM` accepts a regular grid of cell-centred elevations
on projected metre coordinates or a declared local east/north metre frame.
The DEM datum and survey-station datum must match exactly; elevations are
positive upward. The caller explicitly supplies station easting, northing,
elevation, a station coordinate reference that exactly matches the DEM, a
horizontal reference plane, and constant density in kg/m³. Each
finite DEM cell is represented by a rectangular prism between its cell
elevation and the reference plane. Cells above the plane use positive density
contrast; cells below use a negative density contrast. Missing DEM cells are
skipped and counted. The returned additive term is positive-down vertical
attraction in mGal and can be wrapped in a `ProcessingProduct` for a manifest.

This is a finite-footprint DEM residual-mass calculation. It does not infer
terrain outside the grid, apply Earth curvature, transform coordinates, or
replace a complete spherical terrain workflow. The model should not be
described as a complete Bouguer anomaly by itself. Tests compare the prism sum
with a separate closed-form rectangular-prism calculation and the pinned
Harmonica 0.7.0 `prism_gravity` implementation.

## IGRF-14 main-field correction

The pinned IGRF-14 coefficient file is bundled with the engine, so results are
offline and reproducible. Its SHA-256 is recorded in the processing manifest
and checked at load time. `igrf14_main_field` requires geodetic WGS-84 latitude,
east longitude, ellipsoid altitude in metres, and a UTC date or decimal year.
The supported range is 1900.0–2030.0; requests outside it fail instead of
extrapolating. The result records the model, decimal year, location, altitude,
and north/east/down components in nT, where Z/down is positive. Declination is
east of true north and inclination is positive down. The coefficient release
is [IAGA IGRF-14](https://doi.org/10.5281/zenodo.14012302), and the official
[BGS calculator](https://geomag.bgs.ac.uk/data_service/models_compass/igrf_calc.html)
defines the WGS-84 altitude and component conventions used by the reference
tests.

`apply_igrf_main_field_correction` subtracts IGRF total intensity F from
aligned raw total-field observations and retains X, Y, Z, F, the input field,
station coordinates, dates, and corrected values. It must not be applied to
the existing `magnetic_anomaly_nt` survey channel because that channel is
already labelled as an anomaly. IGRF represents the large-scale internal main
field; it does not remove local crustal anomalies, diurnal changes, or external
field variation.

## Stabilized magnetic direction transform

The grid interpretation workflow offers a regularized frequency-domain
direction transform. Field, total source magnetization, and target directions
are entered explicitly; inclination is positive down and declination is east
of north. The default target is vertical, corresponding to reduction to the
pole. The directional kernel follows the planar potential-field form
`Theta = m_down + i*(m_e*k_e + m_n*k_n)/|k|`; the observed product is inverted
using Tikhonov regularization and multiplied by the target product. The
required positive dimensionless lambda and maximum allowed gain are saved in
the UI result; the engine reports maximum gain, RMS gain, and the fraction of
frequencies above unit gain. The implementation follows the documented
[Harmonica RTP kernel](https://www.fatiando.org/harmonica/v0.7.0/api/generated/harmonica.filters.reduction_to_pole_kernel.html)
with a ridge-regularized inverse.

Tests compare the unregularized-limit result with Harmonica at moderate
inclination and check that low-latitude regularization caps gain. This does
not establish a unique magnetization direction or make a noisy low-latitude
transform geologically stable. Remanence, nonuniform magnetization, missing
cells, sampling, and edge effects remain assumptions for interpretation.

## Projects, UI, and exports

Project JSON schema version 2 stores the raw survey profile, coordinate and
elevation metadata, source provenance, and processing-step manifests. The
loader continues to accept earlier project versions. The desktop menu and web
sidebar expose survey import, free-air and optional slab reduction, magnetic
diurnal correction, polynomial regional separation, regular interpolation,
profile spectral transforms, previews, and processing-bundle download. The
desktop and web grid panels accept complete regular XYZ grids for grid
derivatives, magnetic direction transforms, Euler window sweeps, and
band-limited spectral depth estimates. Irregular XYZ transects continue to use
the existing map-slicing workflow.

The reproducible ZIP contains `profile.csv`, a `manifest.json`, any
regular-profile outputs, and supporting base-station CSVs. It preserves the raw
channels and records output units, correction signs, transform assumptions,
processing parameters, and SHA-256 hashes.

## Validated scope and deferred work

Synthetic tests compare corrections with independently evaluated scalar
formulas, terrain prisms with a separate closed form and Harmonica, IGRF-14
vectors with official BGS calculator values at four locations, regional fitting
with a known orthogonal polynomial-plus-residual signal, interpolation with
linear analytic fields, gridding with a planar field, Fourier transforms with
periodic analytic modes, the magnetic transform with Harmonica, spectral depth
with a synthetic exponential spectrum, and Euler deconvolution/window filters
with analytic homogeneous fields. These checks validate the declared inputs
and formulas; they do not establish accuracy for field data.

Remaining limits: terrain outside the loaded DEM, spatially varying density,
geoid/ellipsoid conversion, SRTM/GeoTIFF reading, geodetic-to-projected CRS
conversion, per-station geodetic position derivation, and direct IGRF correction
controls in the legacy profile-import panel are deferred. The legacy profile
magnetic field is an anomaly, so it is intentionally not relabelled as raw
total field to enable subtraction. The grid spectral workflow requires complete
regular XYZ input; automated missing-cell infill, GIS reprojection, and
multi-source Euler uncertainty estimates are also deferred.
