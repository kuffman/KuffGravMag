# KGM-M6 plan — geophysical data processing

KGM-M5 was merged to `main` in merge commit `53b8ddc`. KGM-M6 implementation
begins from that integrated baseline and is delivered in reviewable stages.

## M6 baseline — validated profile-processing core

Implemented in PR #7 on `codex/kgm-m6-processing-core`:

- Versioned profile and base-station schemas with explicit units, coordinate
  metadata, missing-value behavior, and source hashes.
- Free-air and infinite Bouguer-slab reductions, with raw channels retained;
  magnetic diurnal correction from a supplied base series.
- Polynomial regional/residual separation, linear profile interpolation, and
  scattered XYZ gridding with a convex-hull mask.
- 1D profile and 2D grid Fourier transforms, derivatives, analytic signal, tilt,
  and edge-treatment metadata.
- A fixed-structural-index Euler source/depth solver with rank and conditioning
  diagnostics, validated against a synthetic analytic point source.
- Desktop/web profile-processing controls, project JSON schema version 2, and
  reproducible processing-bundle export.

## M6.1 — terrain, IGRF, and grid interpretation

Implemented on `codex/kgm-m61-validated-interpretation`, based on the merged
M6 baseline:

- Cell-centred regular DEM residual-terrain attraction using explicit metric
  geometry, matched elevation datum, reference plane, station elevations, and
  density; exported as a provenance-bearing processing product.
- Offline IGRF-14 vector evaluation using the bundled coefficient file and
  explicit WGS-84 location, ellipsoid altitude, date, validity range, and N/E/down
  component convention. Raw total-field correction subtracts model F and keeps
  the field vector and location metadata.
- Tikhonov-regularized magnetic direction transforms with explicit field,
  magnetization, and target directions; maximum/RMS/noise-gain diagnostics and
  a configured gain limit.
- Moving Euler windows with index, stride, condition and RMS filters, plus
  structural-index-zero support without a fitted base level.
- Band-limited spectral source-population depth with slope uncertainty and R².
- Desktop and web grid interpretation panels for complete regular XYZ grids.

The equations, schemas, limitations, and independent reference coverage are
recorded in [KGM-M6 processing](KGM-M6_PROCESSING.md). GeoTIFF/SRTM loading,
CRS conversion, terrain outside the DEM, deriving per-station geodetic positions,
and direct IGRF controls in the legacy anomaly-profile panel remain deferred.

## Goal

Add reproducible profile-based processing and interpretation to the validated
forward and inversion engine. Every correction and transformation should carry
explicit units, coordinate metadata, and a record of its input assumptions.

## Proposed work packages

1. **Input schemas and provenance** — define station coordinates, elevation,
   gravity/magnetic channels, field metadata, units, missing-data behavior, and
   source-file provenance. Add small, versioned fixtures.
2. **Gravity corrections** — implement selected free-air, Bouguer slab, and
   terrain corrections only after agreeing on station datum, reduction density,
   elevation sign, and correction sign. Keep raw and corrected channels
   available in exported results.
3. **Magnetic corrections** — define supported diurnal and reference-field
   metadata, then implement corrections that can be reproduced from the saved
   input record. Preserve raw total field and the declared field vector.
4. **Regional/residual separation** — add documented polynomial and/or
   wavelength-based methods with masks, edge handling, and synthetic
   recovery tests. Show the chosen regional model and residual together.
5. **Continuation and derivatives** — implement upward continuation and
   horizontal/vertical derivatives with explicit units, observation geometry,
   sampling rules, and edge treatment. Validate against analytic Fourier or
   prism references.
6. **Analytical signal and source interpretation** — add the analytical signal
   and Euler deconvolution with structural-index controls, conditioning
   diagnostics, and synthetic source-location uncertainty tests.
7. **Low-latitude magnetic interpretation** — add documented methods for
   low-inclination data and demonstrate their limits with synthetic bodies and
   remanent magnetization cases.
8. **Application and reports** — expose each transform as a reproducible
   processing step, keep undoable inputs, and include parameters and provenance
   in project and PDF outputs.

## Acceptance gates

- Each correction preserves the raw channel and records its units, reference
  model, parameters, and sign convention.
- Synthetic tests recover analytic or independently computed results within
  stated tolerances, including edge and low-latitude cases.
- Noise, missing stations, and boundary effects are quantified.
- Outputs can be reproduced from the saved project and processing parameters.
- Desktop, web, and PDF smokes pass in CI before the processing interface is
  considered complete.

## Deferred follow-up decisions

- Supported DEM file readers and CRS conversion, including how station
  coordinates and DEM vertical datums are transformed.
- Per-station geodetic coordinate derivation and explicit raw-total-field input
  in the legacy survey import workflow; current magnetic profile channels are
  anomalies and are not eligible for IGRF subtraction.
- Multi-source and noisy-field uncertainty for Euler clustering and
  low-latitude transforms.
- Project persistence for regular-grid and interpretation outputs.

KGM-M7 is planned as a separate dependent pull request after M6.1 review. It
will focus on project navigation and recovery, a coherent geological editing
workflow, richer scientific plots and grid/GIS views, 3D prism visualization,
and report export while keeping the validated Python engine unchanged.
