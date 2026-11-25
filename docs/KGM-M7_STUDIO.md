# KGM-M7.1 Studio enhancement

This increment extends the Studio project workflow while leaving scientific calculations in the validated engine. Portable bundles carry their project and selected local source files; no basemaps or network services are requested.

## Delivered

| Area | Current implementation |
|---|---|
| Portable project bundle | `.kgmproj` is a ZIP with a versioned JSON manifest, project JSON, and content-addressed source attachments. Desktop and Streamlit can save and load the format; JSON projects remain supported. |
| Archive safety | The loader reads entries without extracting them, rejects traversal paths, duplicate files and JSON keys, symlinks, encryption, unsupported compression, suspicious compression ratios, unlisted files, and checksum mismatches. Limits are 64 MiB per archive and total expanded content, 32 MiB per attachment, 8 MiB for project JSON, and 64 entries. |
| Embedded data and provenance | Regular XYZ grid sources and local XYZ layers are attached by content hash. Project processing records include source filename, checksum, value units, source/target CRS, point count, and local-only status. Imported profile observations and survey data already serialize into project JSON. |
| CRS conversion | Local XYZ points can be transformed between declared CRS definitions through PROJ/pyproj with explicit X/Y axis order. Missing CRS, malformed points, non-finite values, and invalid transformations fail with a message. |
| Local spatial layers | Desktop and Streamlit load, display, and export local scattered XYZ layers. Transformed exports include source and target CRS metadata. No remote tiles, geocoding, or cloud upload are involved. |
| Linked model views | Selecting a body in the desktop project explorer updates the selected-body properties, 2D section highlight, and finite-prism highlight. Edits to polygon vertices in 2D or strike bounds in the shared properties update the 3D scene. Web controls retain the same model edit fields and now include a finite-prism preview. |
| Persistence and reporting | Project JSON remains atomic. `.kgmproj` uses atomic writes and hashes each attachment. Recovery snapshots and PDF report fingerprints continue to cover the canonical project inputs and processing history. |

## Limits and deferred work

- Only local XYZ text/CSV point layers are supported. GeoPackage, shapefile, raster pyramids, CRS database management, and basemap services are deferred.
- CRS conversion supports coordinate transformation, but does not reproject grids inside grid processing. Grid interpretation still requires complete regular coordinates in metres and an explicitly meaningful coordinate reference.
- Web finite-prism rendering is a static preview; the desktop Matplotlib view is rotatable. Direct 3D vertex dragging, geological surface meshes, multiple synchronized canvases, and full desktop/web workflow parity are deferred.
- Bundles are capped to keep archive handling bounded. A grid or attachment larger than 32 MiB must be reduced or referenced externally; large project environments are not embedded.
- Processing provenance records assumptions and file checksums. It does not claim that an input dataset is scientifically valid, licensed for redistribution, or externally verified.

## Validation

`tests/test_project_bundle.py` exercises deterministic round trips, identical-content attachments, traversal and duplicate-entry rejection, checksum validation, unsupported unlisted files, duplicate JSON keys, and compression-ratio limits. `tests/test_spatial_layers.py` checks axis-order behavior with synthetic Ghana UTM coordinates, source hashes, round trips, malformed rows, non-finite values, and bad CRS input. Desktop offscreen checks connect project explorer selection to 2D and 3D state and verify that shared strike-bound edits update the prism. The existing Windows packaging and macOS offscreen/web/PDF CI remain enabled.

See [the local Qt diagnostic](LOCAL_QT_DIAGNOSTIC.md) for the distinction between offscreen widget rendering and a physical macOS display session.
