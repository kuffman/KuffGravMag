# KGM-M9 architecture review and exchange decision

Reviewed 2026-11-25 before implementation. This review records the two repository baselines and the boundaries that the prototype must respect.

## Verified baselines

| Application | Repository | Clean task-copy base | Original local checkout |
|---|---|---|---|
| KuffGravMag | `kuffman/KuffGravMag` | `main` at `0d077f0b7e198584cd0a8bc6363c5603aea8305e` | `/Users/kuffouropoku/KuffGravMag`, `main` at `aad19b48bd105e2656aaeb18c82dd9b7e697e02a`, with existing staged, unstaged, and untracked changes |
| KuffGeoSeismic Lab | `kuffman/KuffGeoSeismic_Lab` | `main` at `d078275950fa1fadd4f02415899732ad083a912a` | `/Users/kuffouropoku/Developer/Projects/Web/KuffGeoSeismic Lab/frontend`, `feature/advanced-modelling-inversion` at `7e9cfaa74031587d93f91f76e430064595930993`, with existing staged, unstaged, and untracked changes |

The expected GravMag `main` SHA was verified through GitHub; the original checkout's local `origin/main` was stale. GitHub reports all four checks on the verified GravMag baseline successful: Python scientific suite, independent benchmarks/report, macOS arm64 desktop/web/PDF smoke, and Windows build. The supplied local baseline record reports 158 passing tests and four local Qt checks skipped because the offscreen plugin was unavailable.

The Seismic Lab `main` SHA was verified both through GitHub and the clean task copy. Its `main` checks include a successful build/scientific validation run; one older validation run is cancelled. No open PRs were present in either repository at review time.

Neither repository contains a top-level license file at the reviewed baselines. This proposal therefore does not assert a repository license or change license terms.

## Application boundaries

### KuffGravMag

- Python 3.11.18, `uv` lock, NumPy-based scientific engine; PyQt6 desktop and Streamlit web entry points. Desktop and web already read/write the native version-2 project format and `.kgmproj` bundles.
- A `Body2D` is a closed polygon in profile `x` and vertical `z`, extruded along strike `y`. The engine's vertical coordinate is positive down. Strike can be infinite or finite.
- Gravity uses density contrast in kg/m³ and returns positive-down `g_z` in mGal. Magnetics uses SI susceptibility, remanent intensity in A/m, inclination positive down, declination clockwise from north, and a declared inducing field. The engine has no seismic velocity or elastic-property model.
- Existing numerical gates cover the infinite-strike kernels, finite-strike prism integrals, independent references, synthetic inversion behavior, and a pinned IGRF-14 coefficient set. The inversion remains non-unique and uncertainty is local.

### KuffGeoSeismic Lab

- TypeScript/React 19/Vite web application, Node 24.22.1 via mise, Firebase-backed account projects and local browser cache. It has no native desktop application or Python engine.
- Native `.kgslab` project version 1 stores model width, ordered layers, a visual fault, and survey geometry. A layer has thickness in metres, P-wave velocity in m/s, dip in degrees, color, and a numeric application id. The model designer can draw dips/faults, but current analytical solvers explicitly model horizontal, homogeneous, isotropic layers; refraction requires velocity to increase strictly with depth. Reflection tools model primary arrivals and synthetic teaching records with stated omissions.
- The geology model has no density, susceptibility, remanence, S-wave velocity, elastic moduli, or declared CRS. Model depth is displayed positive down; survey station elevations and shot depths are separately represented and must not be conflated.
- The repository has locked npm dependencies, build/type-check, lint, Node scientific tests, emulator checks, and Playwright E2E workflows. Production deployment is a separate manual workflow and is outside this change.

## Main interoperability gaps

1. There is no shared versioned format. GravMag's native project v2 and Seismic Lab's `.kgslab` v1 are application-specific and encode different models.
2. The coordinate conventions differ: both use metres, but GravMag's engine and Seismic Lab's layer depths use positive-down conventions in different structures. GravMag observations use positive-down `z`; Seismic Lab survey receiver `z` is elevation. No current shared CRS, axis-order, or vertical-datum declaration exists.
3. GravMag models arbitrary profile polygons with infinite/finite strike and gravity/magnetic properties. Seismic Lab models ordered layers and P velocity; its current solvers do not simulate arbitrary polygon bodies, dips, faults, finite strike, or 3D geometry.
4. Physical-property absence must remain absence/unknown. A seismic velocity is not a density or magnetic property. Existing numeric defaults in either app are not exchange values and must not be applied during import.
5. Neither application has field-data validation for this cross-physics mapping. Synthetic checks can establish conversion and solver behavior only; they cannot establish geological interpretation or joint inversion validity.

## ADR-001: versioned, loss-aware JSON exchange

- Keep each scientific engine and native project format independent. Store identical schema and synthetic fixture files in both repositories so the contract remains reviewable with each adapter; do not add a shared service or a third repository.
- Use a JSON Schema 2020-12 contract with format identifier `KuffGeoSharedGeologicalModel`, contract version `1.0.0`, stable model/entity/interface IDs, source provenance, optional exact-byte SHA-256 source records, and namespaced extension objects.
- Declare horizontal CRS, axis order, unit, local transect frame, azimuth when known, vertical datum, vertical unit, and positive direction. The canonical local frame is right-handed: `x` along the transect, `y` left of travel, `z` positive up. When both local-transect and magnetic-field profile azimuths are known, they must agree. Prototype adapters accept local transect coordinates in metres and reject CRS/datum conversions they cannot perform.
- Represent interfaces as polylines and geological entities as either cross-section polygons with an explicit infinite/finite strike extent or explicit 3D polyhedra. A section polygon with infinite strike remains 2D; finite strike is a prism; arbitrary 3D geometry is never silently reduced to a section.
- Encode density as absolute or contrast, with the reference density explicit when absolute density is supplied. Encode susceptibility as dimensionless SI, remanence in A/m with orientation convention, P/S velocities in m/s, and optional elastic moduli in Pa. Each property is absent, explicitly unknown, or known; a known zero is a physical zero.
- Do not derive density, susceptibility, S-wave velocity, remanence, or elastic properties from another property. Do not use application defaults as substitutes for missing exchange values.
- Parse bounded JSON only, reject duplicate keys and non-finite values, enforce a 5 MiB UTF-8 size cap, 32-level JSON depth, 256 entities, 512 interfaces, 100000 total vertices, and a 5000000-operation geometry validation budget. Validate unique IDs and interface references, closed simple polygon rings, and 3D polyhedra as closed, connected, consistently oriented manifolds with non-zero volume. Reject unsupported schema revisions with a path-specific error. A future format migration must be an explicit versioned transform; unknown data is preserved only in bounded namespaced extensions. Every conversion either preserves a field, emits a specific loss warning, or rejects the model before mutation.

### Adapter entry points and limits

- KuffGravMag exposes shared-model import and export in the desktop **File** menu and the Streamlit sidebar. Native project version 2 remains readable and writable; it stores optional shared metadata and bounded per-body exchange properties. Infinite and finite-strike section polygons are supported. KuffGravMag rejects polyhedra and rejects unknown density contrast, susceptibility, or remanence when it needs those values for its existing forward solver. If the input omits an inducing field, it reports that the active KuffGravMag field remains in use; exports declare the active field.
- KuffGeoSeismic Lab exposes the actions in the Project Manager. Native `.kgslab` version 1 remains readable and writable with optional root and per-layer exchange metadata, including cloud persistence under the existing project size cap. Import requires continuous ordered layer polygons spanning the existing 20 m profile; it rejects arbitrary bodies, projected CRS conversion, and rescaling. It retains finite strike and dipped geometry as metadata but warns that current analytical seismic solvers do not simulate those dimensions. Current refraction requires increasing P-wave velocity and treats the last layer as a half-space; the stored final polygon thickness is retained for display and exchange only. Export rejects the app's enabled fault drawing because it is not a validated solver geometry.
- Neither adapter changes the other application's native project format or adds a shared service. The apps retain their independent forward engines. Seismic and elastic values imported by KuffGravMag are preserved as metadata; density and magnetic properties imported by Seismic Lab are preserved as metadata. No physical property is inferred from another.

## Prototype and scientific gates

The shared synthetic fixture is a horizontal, laterally continuous layered section with explicit P velocity, density contrast, SI susceptibility, zero-valued known remanence, local ground-surface datum, and infinite strike. It is intentionally representable by both engines. GravMag checks gravity and magnetic responses using its already validated kernels; Seismic Lab checks first arrivals from its existing analytical layered model. Conversion tests cover vertical sign, axes/units, stable IDs, unknown versus zero, and loss reporting. No field observations are introduced.

The adapters must reject arbitrary bodies when asked to create Seismic Lab analytical layers, report finite-strike or 3D loss if an import path cannot preserve it, and retain unsupported properties only as explicit metadata if that path is implemented. Joint inversion remains research-only until each forward mapping, coordinate convention, synthetic recovery, non-uniqueness, sensitivity, noise, and uncertainty gate has independent review. Field-data validation and any production release remain separate future gates.

M9.6 comparative visualization, M9.7 shared workspace/cloud behavior, M9.8 joint inversion research, M9.9 broad GIS/3D formats, and M9.10 field-data validation are deferred. Each requires its own design, validation evidence, and review; this PR pair does not claim those capabilities.

## Local implementation verification

Verified on 2026-11-25 before PR submission:

- KuffGravMag locked all-groups suite: **168 passed, 4 skipped**. The four existing desktop UI checks skip because this local environment cannot load the Qt offscreen platform plugin.
- KuffGravMag Python compile check and Streamlit/PDF smoke: passed. The standalone desktop smoke could not start for the same missing offscreen Qt plugin.
- KuffGeoSeismic Lab production build/type-check and lint: passed; Node suite: **537 passed, 0 failed, 0 skipped**.
- The shared JSON Schema passes Draft 2020-12 schema checks and the canonical synthetic fixture validates with zero schema errors.
- Direct cross-app smoke passed: Python export → TypeScript import/export → Python re-import preserved all three layers, the 500 kg/m³ density contrast, and the 1500 m/s P-wave value. Seismic Lab emitted its explicit final-half-space solver warning.
- Final preservation check matched both original checkout HEADs and dirty path lists to the recorded pre-work baseline. The two KuffGravMag synthetic CSV SHA-256 values are unchanged.
- These checks use synthetic data only. They do not establish field-data validity, geological interpretation, or joint-inversion validity. Hosted pull-request CI remains a separate check after the branches are submitted.

### PR #12 round-trip precision regression

The shared-model exporter compares reprojected vertex coordinates using a four-ULP tolerance that accounts for the vertical datum value. If the ring is unchanged within floating-point roundoff, it emits the exact imported ring and retains layer kind, stratigraphic order, and interface references. A larger vertex edit still exports as a body with the existing geometry-changed warning. Regression coverage exercises positive-up datum 1.1 m, exact ring preservation through native project save/load, and a real geometry edit.
