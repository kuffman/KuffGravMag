"""Local-only viewer for an explicitly referenced XYZ spatial point layer."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout


class SpatialLayerDialog(QDialog):
    """Display one local XYZ layer without fetching basemaps or remote tiles."""

    def __init__(self, parent, layer):
        super().__init__(parent)
        self.layer = layer
        self.setWindowTitle(f"Local spatial layer — {layer.name}")
        self.resize(860, 680)
        self.figure = Figure(figsize=(8, 5.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.provenance = QLabel(
            f"{layer.x.size} points · {layer.source_crs} → {layer.target_crs} · "
            f"{layer.units} · SHA-256 {layer.source_sha256}"
        )
        self.provenance.setWordWrap(True)
        self.local_only_note = QLabel(
            "Local file layer only. No basemap or remote service is requested."
        )
        self.local_only_note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.provenance)
        layout.addWidget(self.local_only_note)

        axis = self.figure.add_subplot(111)
        points = axis.scatter(layer.x, layer.y, c=layer.values, cmap="viridis", s=24)
        self.figure.colorbar(points, ax=axis, label=layer.units)
        axis.set_xlabel(f"X ({layer.x_unit})")
        axis.set_ylabel(f"Y ({layer.y_unit})")
        axis.set_title(layer.name)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.2)
        self.figure.tight_layout()
        self.canvas.draw_idle()
