"""Interactive, assumption-visible grid interpretation dialog."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from engine.euler import euler_window_sweep
from engine.processing import regular_grid_from_xyz
from engine.spectral import (
    estimate_spectral_depth,
    stabilized_magnetic_transform_grid,
    transform_grid,
)


class GridInterpretationDialog(QDialog):
    """Inspect a regular XYZ grid and run explicit spectral/Euler workflows."""

    def __init__(self, parent, x, y, values, *, field_inclination_deg=55.0,
                 field_declination_deg=0.0):
        super().__init__(parent)
        self.setWindowTitle("KuffGravMag — Grid Interpretation")
        self.resize(1040, 820)
        self._x, self._y, self._values = map(lambda item: np.asarray(item, dtype=np.float64), (x, y, values))
        self.grid = None
        self.transform_result = None
        self.direction_result = None
        self._field_inc = float(field_inclination_deg)
        self._field_dec = float(field_declination_deg)

        root = QVBoxLayout(self)
        metadata = QFormLayout()
        self.coordinate_reference = QLineEdit("local projected grid (metres)")
        self.units = QComboBox()
        self.units.addItem("Magnetic total-field anomaly (nT)", "nT")
        self.units.addItem("Gravity anomaly (mGal)", "mGal")
        self.units.addItem("Other declared units", "input units")
        metadata.addRow("Grid coordinate reference:", self.coordinate_reference)
        metadata.addRow("Grid values:", self.units)
        root.addLayout(metadata)

        self.figure = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(self.canvas, stretch=2)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)
        self._build_spectral_tab()
        self._build_direction_tab()
        self._build_euler_depth_tab()

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(115)
        self.status.setPlaceholderText("Grid validation, parameters, and interpretation limits will appear here.")
        root.addWidget(self.status)
        self._show_input_grid()

    def _build_spectral_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.spectral_output = QComboBox()
        self.spectral_output.addItem("Upward continued field", "upward_continued")
        self.spectral_output.addItem("East derivative", "derivative_x")
        self.spectral_output.addItem("North derivative", "derivative_y")
        self.spectral_output.addItem("Vertical derivative (positive down)", "vertical_derivative_down")
        self.spectral_output.addItem("Analytic signal", "analytic_signal")
        self.spectral_output.addItem("Tilt angle", "tilt_angle_deg")
        self.height = QDoubleSpinBox()
        self.height.setRange(0.0, 1.0e7)
        self.height.setValue(0.0)
        self.height.setSuffix(" m")
        self.padding = QComboBox()
        self.padding.addItem("Reflect edges (mark for review)", "reflect")
        self.padding.addItem("Periodic grid (wrap at edges)", "none")
        form.addRow("Output:", self.spectral_output)
        form.addRow("Upward continuation:", self.height)
        form.addRow("Edge treatment:", self.padding)
        layout.addLayout(form)
        run = QPushButton("Calculate grid transform")
        run.clicked.connect(self.run_spectral_transform)
        layout.addWidget(run)
        self.spectral_caption = QLabel("Complete, regular grids only. Missing cells must be handled explicitly before transforming.")
        self.spectral_caption.setWordWrap(True)
        layout.addWidget(self.spectral_caption)
        self.tabs.addTab(tab, "Grid transforms")

    def _build_direction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.field_inc = self._angle_box(-90.0, 90.0, self._field_inc)
        self.field_dec = self._angle_box(-360.0, 360.0, self._field_dec)
        self.mag_inc = self._angle_box(-90.0, 90.0, self._field_inc)
        self.mag_dec = self._angle_box(-360.0, 360.0, self._field_dec)
        self.target_inc = self._angle_box(-90.0, 90.0, 90.0)
        self.target_dec = self._angle_box(-360.0, 360.0, 0.0)
        self.regularization = self._positive_box(0.001, 0.1, 0.0001)
        self.max_gain = self._positive_box(1.0, 100.0, 20.0)
        form.addRow("Inducing-field inclination / declination:", self._paired(self.field_inc, self.field_dec))
        form.addRow("Source magnetization inclination / declination:", self._paired(self.mag_inc, self.mag_dec))
        form.addRow("Target direction inclination / declination:", self._paired(self.target_inc, self.target_dec))
        form.addRow("Tikhonov regularization λ:", self.regularization)
        form.addRow("Maximum allowed Fourier gain:", self.max_gain)
        layout.addLayout(form)
        run = QPushButton("Calculate stabilized direction transform")
        run.clicked.connect(self.run_direction_transform)
        layout.addWidget(run)
        self.direction_caption = QLabel(
            "Target 90° inclination is reduction to the pole. The source direction must include remanent magnetization when known."
        )
        self.direction_caption.setWordWrap(True)
        layout.addWidget(self.direction_caption)
        self.tabs.addTab(tab, "Magnetic transform")

    def _build_euler_depth_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        euler_group = QGroupBox("Moving Euler windows")
        form = QFormLayout(euler_group)
        self.euler_window = QSpinBox()
        self.euler_window.setRange(3, 999)
        self.euler_window.setSingleStep(2)
        self.euler_window.setValue(15)
        self.euler_stride = QSpinBox()
        self.euler_stride.setRange(1, 999)
        self.euler_stride.setValue(5)
        self.structural_indices = QLineEdit("1, 2, 3")
        self.condition_filter = self._positive_box(1.0, 1e8, 1e4)
        self.rms_filter = self._positive_box(0.0, 1e6, 0.0)
        form.addRow("Window width (odd sample count):", self.euler_window)
        form.addRow("Window stride (samples):", self.euler_stride)
        form.addRow("Structural indices (comma-separated):", self.structural_indices)
        form.addRow("Maximum condition number:", self.condition_filter)
        form.addRow("Maximum RMS residual (0 disables):", self.rms_filter)
        euler_run = QPushButton("Run Euler window sweep")
        euler_run.clicked.connect(self.run_euler_sweep)
        form.addRow(euler_run)
        layout.addWidget(euler_group)

        depth_group = QGroupBox("Band-limited spectral depth estimate")
        depth_form = QFormLayout(depth_group)
        self.depth_k_min = self._positive_box(1e-6, 0.005, 1e-6)
        self.depth_k_max = self._positive_box(0.005, 0.1, 0.001)
        self.depth_bins = QSpinBox()
        self.depth_bins.setRange(4, 256)
        self.depth_bins.setValue(24)
        depth_form.addRow("Minimum radial wavenumber (rad/m):", self.depth_k_min)
        depth_form.addRow("Maximum radial wavenumber (rad/m):", self.depth_k_max)
        depth_form.addRow("Radial bins:", self.depth_bins)
        depth_run = QPushButton("Estimate spectral depth")
        depth_run.clicked.connect(self.run_depth_estimate)
        depth_form.addRow(depth_run)
        layout.addWidget(depth_group)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Euler / spectral depth")

    @staticmethod
    def _angle_box(minimum, maximum, value):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setValue(value)
        box.setSuffix("°")
        return box

    @staticmethod
    def _positive_box(minimum, maximum, value):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(6)
        box.setValue(value)
        return box

    @staticmethod
    def _paired(first, second):
        row = QHBoxLayout()
        row.addWidget(first)
        row.addWidget(second)
        container = QWidget()
        container.setLayout(row)
        return container

    def _load_regular_grid(self):
        self.grid = regular_grid_from_xyz(
            self._x, self._y, self._values,
            units=str(self.units.currentData()),
            coordinate_reference=self.coordinate_reference.text().strip(),
        )
        return self.grid

    def _show_input_grid(self):
        try:
            grid = self._load_regular_grid()
            self._plot(grid.values, "Observed XYZ grid")
            self.status.setPlainText(
                f"Complete regular grid: {grid.values.shape[1]} × {grid.values.shape[0]} cells; "
                f"spacing {grid.x_m[1]-grid.x_m[0]:g} × {grid.y_m[1]-grid.y_m[0]:g} m."
            )
        except Exception as exc:
            self.status.setPlainText(f"Grid is not ready for interpretation: {exc}")

    def _plot(self, values, title, *, accepted=None):
        grid = self._load_regular_grid()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(
            values, origin="lower", aspect="equal",
            extent=(grid.x_m[0], grid.x_m[-1], grid.y_m[0], grid.y_m[-1]),
            interpolation="nearest", cmap="RdBu_r",
        )
        self.figure.colorbar(image, ax=ax, label=str(self.units.currentData()))
        if accepted:
            xs = [item.result.x0_m for item in accepted]
            ys = [item.result.y0_m for item in accepted]
            depths = [item.result.depth_below_observation_m for item in accepted]
            ax.scatter(xs, ys, c=depths, cmap="viridis", edgecolors="black", s=24)
        ax.set_xlabel("X / easting (m)")
        ax.set_ylabel("Y / northing (m)")
        ax.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def run_spectral_transform(self):
        try:
            grid = self._load_regular_grid()
            result = transform_grid(
                grid.values, spacing_x_m=grid.x_m[1]-grid.x_m[0],
                spacing_y_m=grid.y_m[1]-grid.y_m[0],
                height_m=self.height.value(), padding=str(self.padding.currentData()), units=grid.units,
            )
            self.transform_result = result
            name = str(self.spectral_output.currentData())
            self._plot(getattr(result, name), name.replace("_", " "))
            self.status.setPlainText(
                f"Calculated {name}; reflect-edge review width {result.edge_effect_samples}. "
                + " ".join(result.assumptions)
            )
        except Exception as exc:
            self._show_error("Grid transform failed", exc)

    def run_direction_transform(self):
        try:
            grid = self._load_regular_grid()
            result = stabilized_magnetic_transform_grid(
                grid.values, spacing_x_m=grid.x_m[1]-grid.x_m[0],
                spacing_y_m=grid.y_m[1]-grid.y_m[0],
                field_inclination_deg=self.field_inc.value(), field_declination_deg=self.field_dec.value(),
                magnetization_inclination_deg=self.mag_inc.value(), magnetization_declination_deg=self.mag_dec.value(),
                target_inclination_deg=self.target_inc.value(), target_declination_deg=self.target_dec.value(),
                regularization=self.regularization.value(), padding=str(self.padding.currentData()),
                max_gain_limit=self.max_gain.value(),
            )
            self.direction_result = result
            self._plot(result.values, "Regularized magnetic direction transform")
            self.status.setPlainText(
                f"Maximum Fourier gain {result.max_gain:.4g}; RMS gain {result.rms_gain:.4g}; "
                f"fraction above unity {result.fraction_gain_above_one:.3%}. "
                + " ".join(result.assumptions)
            )
        except Exception as exc:
            self._show_error("Magnetic transform failed", exc)

    def run_euler_sweep(self):
        try:
            grid = self._load_regular_grid()
            if self.transform_result is None:
                self.transform_result = transform_grid(
                    grid.values, spacing_x_m=grid.x_m[1]-grid.x_m[0],
                    spacing_y_m=grid.y_m[1]-grid.y_m[0], padding=str(self.padding.currentData()), units=grid.units,
                )
            indices = [float(value.strip()) for value in self.structural_indices.text().split(",") if value.strip()]
            rms_limit = self.rms_filter.value() or None
            results = euler_window_sweep(
                grid.x_m, grid.y_m, grid.values,
                self.transform_result.derivative_x, self.transform_result.derivative_y,
                -self.transform_result.vertical_derivative_down,
                structural_indices=indices,
                window_size_samples=self.euler_window.value(), stride_samples=self.euler_stride.value(),
                condition_limit=self.condition_filter.value(), max_rms_residual=rms_limit,
                mask=~self.transform_result.edge_mask,
            )
            accepted = [item for item in results if item.accepted]
            self._plot(grid.values, "Accepted Euler source locations" if accepted else "Euler sweep: no accepted solutions", accepted=accepted)
            self.status.setPlainText(
                f"Euler windows: {len(results)} evaluated, {len(accepted)} passed the declared index, "
                f"condition, and RMS filters. Structural index and window selection are user choices; "
                "multiple sources, remanence, edge effects, and derivative noise may invalidate estimates."
            )
        except Exception as exc:
            self._show_error("Euler sweep failed", exc)

    def run_depth_estimate(self):
        try:
            grid = self._load_regular_grid()
            result = estimate_spectral_depth(
                grid.values, spacing_x_m=grid.x_m[1]-grid.x_m[0],
                spacing_y_m=grid.y_m[1]-grid.y_m[0],
                wavenumber_band_rad_per_m=(self.depth_k_min.value(), self.depth_k_max.value()),
                radial_bin_count=self.depth_bins.value(),
            )
            self.status.setPlainText(
                f"Depth {result.depth_m:.4g} ± {result.standard_error_m:.3g} m (linear-fit 1σ), "
                f"R²={result.r_squared:.4f}, bins={result.bin_count}. "
                + " ".join(result.assumptions)
            )
        except Exception as exc:
            self._show_error("Spectral depth estimate failed", exc)

    def _show_error(self, title, error):
        self.status.setPlainText(f"{title}: {error}")
        QMessageBox.warning(self, title, str(error))

