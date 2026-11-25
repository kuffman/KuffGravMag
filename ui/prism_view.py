"""Interactive 3D view of explicitly finite-strike model bodies."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from engine.visualization import finite_prism_faces


class FinitePrismView(QWidget):
    """Matplotlib 3D canvas for finite-strike bodies in the active model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("finitePrismView")
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.note = QLabel("Finite-strike geometry only. Infinite-strike bodies remain in the 2D section view.")
        self.note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.note)
        self.finite_body_count = 0
        self.active_index = None
        self.visible_body_indices = ()
        self.update_scene((), None)

    def update_scene(self, bodies, active_index=None):
        self.figure.clear()
        axis = self.figure.add_subplot(111, projection="3d")
        self.active_index = active_index
        finite_bodies = [
            (index, body) for index, body in enumerate(bodies)
            if getattr(body, "strike_mode", None) == "finite"
        ]
        self.visible_body_indices = tuple(index for index, _body in finite_bodies)
        self.finite_body_count = len(finite_bodies)
        warnings = []
        all_vertices = []
        for index, body in finite_bodies:
            try:
                faces = finite_prism_faces(body)
            except ValueError as exc:
                warnings.append(f"{body.name}: {exc}")
                continue
            all_vertices.extend(faces)
            collection = Poly3DCollection(
                faces,
                facecolor=body.color,
                edgecolor="#26364a",
                linewidth=1.1 if index == active_index else 0.65,
                alpha=0.78 if index == active_index else 0.48,
            )
            axis.add_collection3d(collection)
            center_x = float(np.mean(body.x[:-1]))
            center_y = (float(body.y_min) + float(body.y_max)) / 2.0
            center_z = float(np.mean(body.z[:-1]))
            axis.text(center_x, center_y, center_z, body.name, fontsize=8)

        if all_vertices:
            points = np.vstack(all_vertices)
            minima = np.min(points, axis=0)
            maxima = np.max(points, axis=0)
            spans = np.maximum(maxima - minima, 1.0)
            padding = spans * 0.06
            x_limits = (minima[0] - padding[0], maxima[0] + padding[0])
            y_limits = (minima[1] - padding[1], maxima[1] + padding[1])
            z_lower = min(0.0, minima[2] - padding[2])
            z_upper = maxima[2] + padding[2]
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_zlim(z_upper, z_lower)
            axis.set_box_aspect((
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                z_upper - z_lower,
            ))
            axis.set_title("Finite-strike prism geometry")
            axis.set_xlabel("Profile distance X (m)")
            axis.set_ylabel("Strike Y (m)")
            axis.set_zlabel("Depth Z (m, positive down)")
            axis.view_init(elev=23, azim=-58)
            axis.grid(True)
        else:
            axis.set_axis_off()
            axis.text2D(
                0.06, 0.90,
                "No finite-strike bodies to display.\n"
                "Set a body's strike model to Finite Strike to view its 3D prism.",
                transform=axis.transAxes,
                va="top",
            )
        self.note.setText(
            "Finite-strike geometry only. Infinite-strike bodies remain in the 2D section view."
            + ("  " + "  ".join(warnings) if warnings else "")
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()
