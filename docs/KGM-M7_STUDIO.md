# KGM-M7 Studio foundation

This change introduces the first integrated Studio workspace while keeping the validated Python scientific engine as the source of all forward, processing, and inversion calculations.

## Delivered

| Area | Current implementation |
|---|---|
| Project explorer | Shows survey observations, processing history, model bodies, and an inspected grid source. |
| Guided workflows | Actions link observed-profile import, survey processing, grid interpretation, body creation, gravity and magnetic inversion, CSV export, and PDF reporting to the existing workflows. A selected polynomial residual can be handed from survey processing into the model and inversion views with finite stations and elevation sign recorded. |
| Interactive model editor | Retains the draggable 2D cross-section editor and adds a synchronized 3D view that users can rotate and zoom. |
| Finite-prism visualization | Extrudes the closed 2D polygon between the declared finite strike bounds. Infinite-strike bodies are omitted from the 3D view and remain visible in the 2D section. |
| Persistence and recovery | Project JSON writes are atomic. A debounced local recovery snapshot contains model settings, observations, survey metadata, and processing history. Earlier project schema versions remain supported. |
| Reproducible report | PDF output includes the project snapshot SHA-256, observation counts, survey counts, and recorded processing steps alongside the existing model summary. |
| Cross-platform baseline | The desktop smoke now checks the explorer, workflow state, project save/load, grid tools, and 3D view. Existing macOS arm64 and Windows CI gates remain in place. |

## Staged work

- Project files record a grid source filename and SHA-256, but do not embed grid arrays. Reopening a project requires selecting the source file again; a portable project bundle is deferred.
- The grid tools accept regular XYZ data in the declared coordinate reference. CRS conversion, GIS basemaps, irregular-grid management, and persistent spatial layers are deferred.
- 3D rendering is a geometry viewer. Polygon editing remains in the 2D section; direct 3D vertex editing, multiple linked views, and geological surface meshes are deferred.
- The new explorer and workflow strip are desktop-only. The existing Streamlit workflow remains available, but full desktop/web parity and project navigation in the web interface are deferred.
- The report fingerprint covers the canonical saved project inputs and processing manifest. Reproducible data bundles with embedded source files and environment manifests are deferred.

## Validation

The M7 test additions verify finite-prism geometry bounds, the infinite-strike exclusion rule, recovery round trips, and stable project fingerprints. The desktop smoke verifies the project explorer, inversion action state, project save/load, and 3D model view. Scientific algorithms and their numerical regression fixtures remain unchanged.

See [the local Qt diagnostic](LOCAL_QT_DIAGNOSTIC.md) for the distinction between offscreen widget rendering and a physical macOS display session.
