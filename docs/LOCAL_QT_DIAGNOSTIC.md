# Local Qt diagnostic

## Environment

- Host and uv-managed Python: `arm64`
- Python: `3.11.18`
- PyQt: `6.12.1`
- Qt: `6.12.0`
- The Qt offscreen plugin is a Mach-O arm64 bundle. The Python executable is a Mach-O arm64 executable.

## Offscreen result

With `QT_DEBUG_PLUGINS=1` and `QT_QPA_PLATFORM=offscreen`, Qt found `libqoffscreen.dylib`, reported its `offscreen` key, and loaded the library. `QApplication` started successfully. The full `scripts/smoke_desktop.py` run passed on arm64, including the Studio explorer, 3D prism view, grid interpretation, and project save/load checks.

An `env -i` check with only the Python path, `PYTHONNOUSERSITE=1`, a temporary Matplotlib config directory, and `QT_QPA_PLATFORM=offscreen` also started Qt successfully. No `QT_*`, `DYLD_*`, `PYTHONPATH`, virtualenv, or Conda overrides were present in the normal shell environment during the smoke.

`otool -L` shows the plugin's QtGui and QtCore frameworks plus macOS AppKit, OpenGL, ImageIO, Metal, IOKit, Disk Arbitration, and Uniform Type Identifiers frameworks. The plugin loaded successfully in the same runtime, so no unresolved native dependency was observed.

The earlier offscreen startup failure did not reproduce in this fresh locked uv environment. Its original cause remains undetermined; no global Qt or Python settings were changed.

## Physical display limitation

With `QT_QPA_PLATFORM=cocoa`, `QT_DEBUG_PLUGINS=1` showed `libqcocoa.dylib` loading, but `QApplication.screens()` returned zero screens. macOS logged service and Launch Services notification warnings in this headless session. That prevents verifying a native on-screen launch here.

The offscreen smoke verifies widget creation and rendering only. It does not verify Cocoa display presentation, physical-screen interaction, or window placement. GitHub's macOS arm64 offscreen desktop/web/PDF job passed separately on PR #8's fix commit.

## Reproduction

```sh
uv sync --locked --group desktop
QT_DEBUG_PLUGINS=1 QT_QPA_PLATFORM=offscreen uv run --locked --group desktop python scripts/smoke_desktop.py
```
