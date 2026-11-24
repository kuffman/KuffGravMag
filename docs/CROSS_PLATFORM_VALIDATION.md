# KGM-M3.1 cross-platform validation

## macOS arm64 and Qt startup

The locked environment was checked on Apple Silicon with Python 3.11.18 arm64, PyQt 6.12.1, and Qt 6.12.0. Qt's configured plugin directory was inside the locked PyQt installation. The `libqcocoa.dylib` Cocoa platform plugin is an arm64 Mach-O binary.

With `QT_DEBUG_PLUGINS=1`, Qt found and loaded `libqcocoa.dylib`. The direct window launch then stopped with `Cannot create window: no screens available`. This run had no accessible desktop screen, so that message is a headless-session limitation rather than evidence that the Cocoa plugin is missing or built for the wrong architecture. Setting `QT_QPA_PLATFORM=offscreen` successfully created and rendered the complete desktop main window, including its Matplotlib canvas.

The desktop entry point now exits with an explanatory message when Qt reports no available screens. `scripts/smoke_desktop.py` exercises window construction with Qt's offscreen plugin. The macOS arm64 GitHub Actions job runs this smoke along with web and PDF checks. A native on-screen launch still needs a logged-in macOS desktop session.

Useful local diagnostics:

```sh
QT_DEBUG_PLUGINS=1 uv run --locked --group desktop python app.py
QT_QPA_PLATFORM=offscreen uv run --locked --group desktop python scripts/smoke_desktop.py
```

## Web, PDF, and Windows packaging

`scripts/smoke_web_pdf.py` renders the Streamlit app once and creates a minimal PDF report. The Windows workflow now uses the lockfile, runs the scientific suite, and builds on pull requests as well as pushes to the main branches. The first passing Windows Actions run is the packaging evidence for this milestone; a local Windows host was not available.
