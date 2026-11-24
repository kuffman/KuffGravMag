# KGM-M6 plan — geophysical data processing

KGM-M5 was merged to `main` in merge commit `53b8ddc`. KGM-M6 implementation
begins from that integrated baseline and is delivered in reviewable stages.

## M6.1 — validated profile-processing core

Implemented on `codex/kgm-m6-processing-core`:

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

The equations, input schemas, limitations, and benchmark coverage are recorded
in [KGM-M6 processing](KGM-M6_PROCESSING.md). Terrain corrections, IGRF/main-field
corrections, stabilized low-latitude reductions, and interactive Euler-grid
controls remain deferred pending validated inputs, independent references, or
a tested interpretation workflow.

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

## Decisions to settle before implementation

- Required gravity and magnetic input formats and coordinate reference.
- Elevation datum, terrain-data source, and default Bouguer density.
- Supported magnetic reference-field model and its date/location metadata.
- Whether regional separation defaults to a polynomial, a wavelength filter,
  or a user-selected method.
- How processing uncertainty and filter edge effects appear in the interface.

Do not start these work packages as part of KGM-M5. The current pull request
should include only this planning document for M6.
