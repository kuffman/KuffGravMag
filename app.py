import sys
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from PyQt6.QtCore import QStandardPaths, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QDoubleSpinBox, QPushButton, QGroupBox, QListWidget,
    QComboBox, QLabel, QFileDialog, QMessageBox, QDialog, QCheckBox,
    QLineEdit, QSpinBox, QTextEdit, QDialogButtonBox, QGridLayout,
    QDockWidget, QTabWidget, QInputDialog,
)
from PyQt6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from engine.forward import Body2D, PotentialField2D
from engine.geological_exchange import export_kgm_model, import_shared_model, validate_shared_model
from engine.report import generate_pdf_report
from engine.grid import extract_profile_from_xyz
from engine.inversion import invert_body_parameters
from engine.project import (
    load_project as read_project,
    project_to_dict,
    save_project as write_project,
)
from engine.project_bundle import (
    MAX_ATTACHMENT_BYTES,
    load_project_bundle,
    save_project_bundle,
)
from engine.spatial import load_xyz_layer
from engine.survey import SurveyProfile, BaseStationSeries, export_processing_bundle
from engine.processing import (
    apply_gravity_reductions,
    apply_magnetic_diurnal_correction,
    separate_polynomial_regional,
    interpolate_profile,
    RegularProfile,
)
from engine.spectral import transform_profile
from ui.grid_interpretation import GridInterpretationDialog
from ui.prism_view import FinitePrismView
from ui.project_recovery import (
    clear_recovery_snapshot,
    load_recovery_snapshot,
    recovery_is_newer,
    save_recovery_snapshot,
)
from ui.studio_workspace import StudioWorkspace
from ui.studio_workflow import prepare_model_observations
from ui.academic_lab import AcademicLabDialog
from ui.spatial_layer_dialog import SpatialLayerDialog

COLOR_OPTIONS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']


class MapProfileDialog(QDialog):
    def __init__(self, parent, x, y, v, name="Grid Data"):
        super().__init__(parent)
        self.setWindowTitle(f"Extract Profile from {name}")
        self.resize(800, 650)
        self.layout = QVBoxLayout(self)

        self.info_label = QLabel("<b>Step 1:</b> Click starting point (A) | <b>Step 2:</b> Click ending point (B)")
        self.layout.addWidget(self.info_label)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        xi = np.linspace(x.min(), x.max(), 150)
        yi = np.linspace(y.min(), y.max(), 150)
        XI, YI = np.meshgrid(xi, yi)
        VI = griddata((x, y), v, (XI, YI), method='linear')

        cmap_plot = self.ax.pcolormesh(XI, YI, VI, shading='auto', cmap='jet')
        self.figure.colorbar(cmap_plot, ax=self.ax, label="Amplitude")
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("Click two points to define profile transect")
        self.canvas.draw()

        self.x_data, self.y_data, self.v_data = x, y, v
        self.pts = []
        self.profile_dist = None
        self.profile_v = None

        self.canvas.mpl_connect('button_press_event', self.on_click)

        btn_box = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Profile Slice")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self.accept)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_confirm)
        self.layout.addLayout(btn_box)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if len(self.pts) < 2:
            self.pts.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ko', markersize=8)
            self.ax.text(event.xdata, event.ydata, f"  {'A' if len(self.pts) == 1 else 'B'}", fontweight='bold')

            if len(self.pts) == 2:
                self.ax.plot([self.pts[0][0], self.pts[1][0]], [self.pts[0][1], self.pts[1][1]], 'k-', lw=2)
                self.profile_dist, self.profile_v = extract_profile_from_xyz(
                    self.x_data, self.y_data, self.v_data,
                    self.pts[0][0], self.pts[0][1], self.pts[1][0], self.pts[1][1]
                )
                self.info_label.setText(
                    "<span style='color:green;'><b>Profile extracted! Click 'Confirm Profile Slice'.</b></span>")
                self.btn_confirm.setEnabled(True)
            self.canvas.draw()


class SurveyProcessingDialog(QDialog):
    """Load a strict profile schema, run reversible processing, and export provenance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KuffGravMag — Survey Processing")
        self.resize(760, 720)
        self.profile = None
        self.base_station = None
        self.products = []
        self.regular_profiles = []
        self.model_inputs = []

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self.profile_path = QLineEdit()
        browse_profile = QPushButton("Browse…")
        browse_profile.clicked.connect(self.browse_profile)
        profile_row = QHBoxLayout()
        profile_row.addWidget(self.profile_path)
        profile_row.addWidget(browse_profile)
        form.addRow("Survey CSV:", profile_row)

        self.coordinate_reference = QLineEdit("local profile coordinates")
        form.addRow("Coordinate reference:", self.coordinate_reference)
        self.profile_azimuth = QDoubleSpinBox()
        self.profile_azimuth.setRange(-180.0, 180.0)
        self.profile_azimuth.setDecimals(4)
        form.addRow("Profile azimuth (deg):", self.profile_azimuth)
        self.elevation_datum = QLineEdit("local survey benchmark")
        form.addRow("Elevation datum:", self.elevation_datum)
        self.reduction_datum_m = QDoubleSpinBox()
        self.reduction_datum_m.setRange(-100000.0, 100000.0)
        self.reduction_datum_m.setDecimals(3)
        self.reduction_datum_m.setSuffix(" m")
        form.addRow("Gravity reduction datum:", self.reduction_datum_m)
        self.include_bouguer = QCheckBox("Apply declared infinite-slab correction")
        self.include_bouguer.setChecked(False)
        self.bouguer_density = QDoubleSpinBox()
        self.bouguer_density.setRange(1.0, 10000.0)
        self.bouguer_density.setValue(2670.0)
        self.bouguer_density.setDecimals(1)
        self.bouguer_density.setSuffix(" kg/m³")
        form.addRow("Bouguer slab:", self.include_bouguer)
        form.addRow("Slab density:", self.bouguer_density)

        self.channel = QComboBox()
        self.channel.addItem("Gravity (mGal)", "gravity")
        self.channel.addItem("Magnetic anomaly (nT)", "magnetic")
        form.addRow("Profile channel:", self.channel)
        self.degree = QSpinBox()
        self.degree.setRange(0, 5)
        self.degree.setValue(1)
        form.addRow("Regional polynomial degree:", self.degree)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.001, 1.0e7)
        self.spacing.setDecimals(3)
        self.spacing.setValue(100.0)
        self.spacing.setSuffix(" m")
        form.addRow("Regular profile spacing:", self.spacing)
        self.height = QDoubleSpinBox()
        self.height.setRange(0.0, 1.0e7)
        self.height.setDecimals(3)
        self.height.setSuffix(" m")
        form.addRow("Upward continuation height:", self.height)
        self.padding = QComboBox()
        self.padding.addItem("Reflect padding (review edges)", "reflect")
        self.padding.addItem("Periodic ends (no padding)", "none")
        form.addRow("Spectral edge treatment:", self.padding)

        base_row = QHBoxLayout()
        self.base_path = QLineEdit()
        browse_base = QPushButton("Load base CSV…")
        browse_base.clicked.connect(self.browse_base_station)
        base_row.addWidget(self.base_path)
        base_row.addWidget(browse_base)
        form.addRow("Magnetic base station:", base_row)
        self.reference_time = QLineEdit()
        self.reference_time.setPlaceholderText("UTC ISO-8601, e.g. 2026-01-01T12:00:00Z")
        form.addRow("Diurnal reference time:", self.reference_time)

        load_btn = QPushButton("Load survey profile")
        load_btn.clicked.connect(self.load_profile)
        root.addWidget(load_btn)

        operations = QGridLayout()
        for index, (title, callback) in enumerate((
            ("Gravity reductions", self.apply_gravity),
            ("Magnetic diurnal", self.apply_diurnal),
            ("Regional / residual", self.apply_regional),
            ("Resample profile", self.resample),
            ("Spectral transforms", self.apply_spectral),
            ("Export processing ZIP…", self.export_bundle),
        )):
            button = QPushButton(title)
            button.clicked.connect(callback)
            operations.addWidget(button, index // 3, index % 3)
        root.addLayout(operations)

        handoff_group = QGroupBox("Model handoff")
        handoff_layout = QVBoxLayout(handoff_group)
        self.model_input_combo = QComboBox()
        self.model_input_combo.addItem("Keep current model observations", None)
        handoff_layout.addWidget(self.model_input_combo)
        handoff_note = QLabel(
            "Only declared polynomial residuals are offered. Selecting one transfers its finite stations to the model and inversion views."
        )
        handoff_note.setWordWrap(True)
        handoff_layout.addWidget(handoff_note)
        root.addWidget(handoff_group)

        self.preview_figure = Figure(figsize=(7, 2.4), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        root.addWidget(self.preview_canvas)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Loaded profile details and processing assumptions will appear here.")
        root.addWidget(self.log, stretch=1)
        self.note = QLabel(
            "The processing package records raw channels, units, assumptions, parameters, and hashes. "
            "DEM terrain residuals and IGRF-14 raw total-field corrections are available through the processing API; "
            "this profile panel lacks station map coordinates, and its magnetic channel is already an anomaly."
        )
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Keep survey in project")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def browse_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load versioned survey profile", "", "CSV (*.csv)")
        if path:
            self.profile_path.setText(path)

    def browse_base_station(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load magnetic base-station series", "", "CSV (*.csv)")
        if path:
            try:
                self.base_station = BaseStationSeries.from_csv(path)
                self.base_path.setText(path)
                self._log(f"Loaded base station: {len(self.base_station.time_utc)} UTC readings.")
            except Exception as exc:
                QMessageBox.warning(self, "Base-station input error", str(exc))

    def load_profile(self):
        path = self.profile_path.text().strip()
        try:
            if not path:
                raise ValueError("Choose a survey CSV first.")
            metadata = {
                "coordinate_reference": self.coordinate_reference.text(),
                "profile_azimuth_deg": self.profile_azimuth.value(),
                "elevation_datum": self.elevation_datum.text(),
            }
            self.profile = SurveyProfile.from_csv(path, metadata=metadata)
            self.products.clear()
            self.regular_profiles.clear()
            self.model_inputs.clear()
            self.model_input_combo.clear()
            self.model_input_combo.addItem("Keep current model observations", None)
            channels = []
            if self.profile.gravity_mgal is not None:
                channels.append(("Gravity (mGal)", "gravity"))
            if self.profile.magnetic_anomaly_nt is not None:
                channels.append(("Magnetic anomaly (nT)", "magnetic"))
            self.channel.clear()
            for label, key in channels:
                self.channel.addItem(label, key)
            self._log(
                f"Loaded {self.profile.distance_m.size} stations from {self.profile.provenance['source_name']} "
                f"(SHA-256 {self.profile.provenance['source_sha256']}).\n"
                f"Coordinate reference: {self.profile.metadata['coordinate_reference']}; "
                f"azimuth: {self.profile.metadata['profile_azimuth_deg']}°; "
                f"elevation datum: {self.profile.metadata['elevation_datum']}."
            )
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Survey input error", str(exc))

    def _require_profile(self):
        if self.profile is None:
            raise ValueError("Load a versioned survey profile first.")
        return self.profile

    def _selected_channel(self):
        profile = self._require_profile()
        if self.channel.currentData() == "gravity":
            return profile.gravity_mgal, "mGal"
        return profile.magnetic_anomaly_nt, "nT total-field anomaly"

    def _log(self, message):
        self.log.append(message)

    def apply_gravity(self):
        try:
            product = apply_gravity_reductions(
                self._require_profile(),
                datum_elevation_m=self.reduction_datum_m.value(),
                bouguer_density_kg_m3=self.bouguer_density.value() if self.include_bouguer.isChecked() else None,
            )
            self.products.append(product)
            self._log("Applied gravity reductions.\n" + "\n".join(product.assumptions))
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Gravity processing error", str(exc))

    def apply_diurnal(self):
        try:
            profile = self._require_profile()
            if self.base_station is None:
                raise ValueError("Load a base-station CSV with time_utc,total_field_nt first.")
            product = apply_magnetic_diurnal_correction(
                profile, self.base_station, reference_time_utc=self.reference_time.text()
            )
            self.products.append(product)
            self._log("Applied magnetic diurnal correction.\n" + "\n".join(product.assumptions))
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Magnetic processing error", str(exc))

    def apply_regional(self):
        try:
            values, units = self._selected_channel()
            product = separate_polynomial_regional(
                self.profile.distance_m, values, degree=self.degree.value(), units=units
            )
            self.products.append(product)
            channel = str(self.channel.currentData())
            residual_name = next(name for name in product.columns if name.startswith("residual_"))
            self.model_inputs.append({
                "channel": channel,
                "source_column": residual_name,
                "values": product.columns[residual_name],
                "units": product.units[residual_name],
                "label": f"{channel.title()} residual · degree {self.degree.value()}",
            })
            self.model_input_combo.addItem(self.model_inputs[-1]["label"], len(self.model_inputs) - 1)
            self.model_input_combo.setCurrentIndex(self.model_input_combo.count() - 1)
            self._log(f"Fit degree-{self.degree.value()} regional and residual.\n" + "\n".join(product.assumptions))
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Regional separation error", str(exc))

    def selected_model_input(self):
        selected = self.model_input_combo.currentData()
        if selected is None or selected < 0 or selected >= len(self.model_inputs):
            return None
        return self.model_inputs[int(selected)]

    def resample(self):
        try:
            values, units = self._selected_channel()
            source = self._require_profile()
            regular = interpolate_profile(
                source.distance_m, values, spacing_m=self.spacing.value(), units=units
            )
            channel_name = str(self.channel.currentData())
            regular = RegularProfile(
                regular.distance_m,
                regular.values,
                regular.units,
                name=f"{channel_name}_linear_interpolation",
                parameters=regular.parameters,
                assumptions=regular.assumptions,
            )
            self.regular_profiles.append(regular)
            self._log(f"Created {regular.distance_m.size} regular samples at {regular.spacing_m:g} m spacing.")
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Interpolation error", str(exc))

    def apply_spectral(self):
        try:
            if not self.regular_profiles:
                raise ValueError("Resample the selected channel first; Fourier transforms require regular sampling.")
            regular = self.regular_profiles[-1]
            if not regular.name.startswith(str(self.channel.currentData())):
                raise ValueError("Resample the currently selected channel before transforming it.")
            result = transform_profile(
                regular.values,
                spacing_m=regular.spacing_m,
                height_m=self.height.value(),
                padding=str(self.padding.currentData()),
                units=regular.units,
            )
            parameters = {
                "source_regular_profile": regular.name,
                "height_m": self.height.value(),
                "padding": self.padding.currentData(),
                "spacing_m": regular.spacing_m,
                "transform_assumptions": list(result.assumptions),
                "edge_effect_samples": result.edge_effect_samples,
            }
            for output_name in (
                "upward_continued", "horizontal_derivative", "vertical_derivative_down",
                "analytic_signal", "tilt_angle_deg",
            ):
                self.regular_profiles.append(RegularProfile(
                    regular.distance_m,
                    getattr(result, output_name),
                    result.units[output_name],
                    name=f"{regular.name}_{output_name}",
                    parameters=parameters,
                    assumptions=result.assumptions,
                ))
            self._log(
                "Calculated upward continuation, horizontal/vertical derivatives, analytic signal, and tilt. "
                f"Review the first/last {result.edge_effect_samples} samples for padding effects.\n"
                + "\n".join(result.assumptions)
            )
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Spectral processing error", str(exc))

    def export_bundle(self):
        try:
            profile = self._require_profile()
            path, _ = QFileDialog.getSaveFileName(
                self, "Export reproducible processing bundle", "KuffGravMag_Processing.zip", "ZIP (*.zip)"
            )
            if not path:
                return
            from pathlib import Path
            Path(path).write_bytes(export_processing_bundle(profile, self.products, regular_profiles=self.regular_profiles))
            QMessageBox.information(self, "Processing export", f"Saved raw inputs, outputs, and provenance to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Processing export error", str(exc))

    def _refresh_plot(self):
        self.preview_figure.clear()
        if self.profile is None:
            self.preview_canvas.draw_idle()
            return
        axis = self.preview_figure.add_subplot(111)
        values, units = self._selected_channel()
        axis.plot(self.profile.distance_m, values, color="black", marker=".", label="Raw channel")
        for product in self.products:
            for name, result in product.columns.items():
                if product.units[name] == units:
                    axis.plot(self.profile.distance_m, result, label=name)
        for regular in self.regular_profiles:
            if regular.name.startswith(str(self.channel.currentData())):
                axis.plot(regular.distance_m, regular.values, label=regular.name)
        axis.set_xlabel("Profile distance (m)")
        axis.set_ylabel(units)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2, loc="best")
        self.preview_figure.tight_layout()
        self.preview_canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, *, recovery_path=None, enable_recovery=True):
        super().__init__()
        self.setWindowTitle("KuffGravMag Studio — Potential Field Workbench")
        self.resize(1580, 980)
        self.setStyleSheet("""
            QMainWindow { background: #f4f7fb; }
            QDockWidget::title { background: #e8edf5; padding: 7px; font-weight: 600; }
            QGroupBox { border: 1px solid #d6deea; border-radius: 6px; margin-top: 9px; padding: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #24466f; }
            QPushButton { padding: 6px 8px; border: 1px solid #c8d3e2; border-radius: 4px; background: #ffffff; }
            QPushButton:hover { background: #eaf2fc; border-color: #7898bd; }
            QTabBar::tab { padding: 8px 14px; }
            QTreeWidget { border: 1px solid #d6deea; border-radius: 4px; background: #ffffff; }
        """)
        self._initializing = True
        self._loading_project = False
        self.project_path = None
        self.project_dirty = False
        self.shared_model_metadata = None
        self.grid_source_path = None
        self.grid_source_name = None
        self.grid_attachment_name = None
        self.project_attachments = {}
        self.spatial_layer = None
        self.spatial_layer_attachment_name = None
        self._recovery_enabled = bool(enable_recovery)
        if recovery_path is None:
            data_location = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            if not data_location:
                data_location = str(Path(tempfile.gettempdir()) / "KuffGravMag-Studio")
            recovery_path = Path(data_location) / "recovery.kgm.json"
        self._recovery_file = Path(recovery_path)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1200)
        self._autosave_timer.timeout.connect(self._write_recovery_snapshot)

        self.x_obs = np.linspace(-1000, 1000, 250)
        self.z_obs = np.zeros_like(self.x_obs)
        self.obs_gz = None
        self.obs_dT = None
        self.survey_profile = None
        self.processing_products = []
        self.regular_profiles = []
        self.processing_steps = []
        self.dragging_vertex = None

        # Default Bodies
        b1_x = np.array([-400, -100, 0, -300, -400], dtype=float)
        b1_z = np.array([200, 200, 500, 450, 200], dtype=float)
        body1 = Body2D("Dike / Intrusion 1", b1_x, b1_z, density=400.0, k_succ=0.03, y_min=-400, y_max=400,
                       color='tab:blue')

        b2_x = np.array([100, 400, 450, 150, 100], dtype=float)
        b2_z = np.array([300, 300, 650, 600, 300], dtype=float)
        body2 = Body2D("Basement High 2", b2_x, b2_z, density=-200.0, k_succ=0.01, J_r=2.0, I_r=40.0, y_min=-1000,
                       y_max=1000, color='tab:orange')

        self.bodies = [body1, body2]
        self.active_body_idx = 0

        self.create_menu_bar()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel: Controls
        left_panel = QVBoxLayout()

        global_group = QGroupBox("Regional Ambient Field")
        global_layout = QFormLayout(global_group)

        self.spin_F = QDoubleSpinBox()
        self.spin_F.setRange(10000, 70000)
        self.spin_F.setValue(48000)
        self.spin_F.setSuffix(" nT")
        self.spin_F.valueChanged.connect(self.on_param_changed)
        global_layout.addRow("Earth Field (F):", self.spin_F)

        self.spin_Inc = QDoubleSpinBox()
        self.spin_Inc.setRange(-90, 90)
        self.spin_Inc.setValue(55)
        self.spin_Inc.setSuffix(" deg")
        self.spin_Inc.valueChanged.connect(self.on_param_changed)
        global_layout.addRow("Field Inclination:", self.spin_Inc)

        self.spin_Dec = QDoubleSpinBox()
        self.spin_Dec.setRange(-180, 180)
        self.spin_Dec.setValue(0)
        self.spin_Dec.setSuffix(" deg")
        self.spin_Dec.valueChanged.connect(self.on_param_changed)
        global_layout.addRow("Field Declination:", self.spin_Dec)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(-180, 180)
        self.spin_alpha.setValue(0)
        self.spin_alpha.setSuffix(" deg")
        self.spin_alpha.valueChanged.connect(self.on_param_changed)
        global_layout.addRow("Profile Azimuth:", self.spin_alpha)

        left_panel.addWidget(global_group)

        self.gravity_status_label = QLabel()
        self.gravity_status_label.setWordWrap(True)
        left_panel.addWidget(self.gravity_status_label)

        self.magnetic_status_label = QLabel()
        self.magnetic_status_label.setWordWrap(True)
        left_panel.addWidget(self.magnetic_status_label)

        list_group = QGroupBox("Subsurface Bodies (2.5D)")
        list_layout = QVBoxLayout(list_group)

        self.body_list = QListWidget()
        self.body_list.currentRowChanged.connect(self.on_body_selected)
        list_layout.addWidget(self.body_list)

        btn_box = QHBoxLayout()
        btn_add = QPushButton("Add Body")
        btn_add.clicked.connect(self.add_body)
        btn_box.addWidget(btn_add)

        btn_remove = QPushButton("Remove Body")
        btn_remove.clicked.connect(self.remove_body)
        btn_box.addWidget(btn_remove)

        list_layout.addLayout(btn_box)
        left_panel.addWidget(list_group)

        self.body_props_group = QGroupBox("Selected Body Properties")
        props_layout = QFormLayout(self.body_props_group)

        self.spin_rho = QDoubleSpinBox()
        self.spin_rho.setRange(-2000, 2000)
        self.spin_rho.setValue(300)
        self.spin_rho.setSuffix(" kg/m³")
        self.spin_rho.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Density Contrast:", self.spin_rho)

        self.spin_k = QDoubleSpinBox()
        self.spin_k.setRange(0, 1)
        self.spin_k.setDecimals(4)
        self.spin_k.setValue(0.02)
        self.spin_k.setSuffix(" SI")
        self.spin_k.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Susceptibility:", self.spin_k)

        self.spin_Jr = QDoubleSpinBox()
        self.spin_Jr.setRange(0, 100)
        self.spin_Jr.setValue(0.0)
        self.spin_Jr.setSuffix(" A/m")
        self.spin_Jr.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Remanence (Jr):", self.spin_Jr)

        self.spin_Ir = QDoubleSpinBox()
        self.spin_Ir.setRange(-90, 90)
        self.spin_Ir.setValue(0.0)
        self.spin_Ir.setSuffix(" deg")
        self.spin_Ir.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Remanent Inc (Ir):", self.spin_Ir)

        self.spin_Dr = QDoubleSpinBox()
        self.spin_Dr.setRange(-180, 180)
        self.spin_Dr.setValue(0.0)
        self.spin_Dr.setSuffix(" deg")
        self.spin_Dr.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Remanent Dec (Dr):", self.spin_Dr)

        self.combo_strike_mode = QComboBox()
        self.combo_strike_mode.addItem("Infinite Strike", "infinite")
        self.combo_strike_mode.addItem("Finite Strike", "finite")
        self.combo_strike_mode.currentIndexChanged.connect(self.update_active_body_props)
        props_layout.addRow("Strike Model:", self.combo_strike_mode)

        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1.0e7, 1.0e7)
        self.spin_ymin.setValue(-500)
        self.spin_ymin.setSuffix(" m")
        self.spin_ymin.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Strike Y-Min:", self.spin_ymin)

        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-1.0e7, 1.0e7)
        self.spin_ymax.setValue(500)
        self.spin_ymax.setSuffix(" m")
        self.spin_ymax.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Strike Y-Max:", self.spin_ymax)

        self.combo_color = QComboBox()
        self.combo_color.addItems(COLOR_OPTIONS)
        self.combo_color.currentIndexChanged.connect(self.update_active_body_props)
        props_layout.addRow("Display Color:", self.combo_color)

        left_panel.addWidget(self.body_props_group)

        # Inversion & Optimization Panel
        inv_group = QGroupBox("Model Optimization & Inversion")
        inv_layout = QVBoxLayout(inv_group)

        self.btn_invert_grav = QPushButton("⚙️ Run Gravity Inversion (Fit Observed)")
        self.btn_invert_grav.clicked.connect(lambda: self.run_inversion('gravity'))
        inv_layout.addWidget(self.btn_invert_grav)

        self.btn_invert_mag = QPushButton("⚙️ Run Magnetic Inversion (Fit Observed)")
        self.btn_invert_mag.clicked.connect(lambda: self.run_inversion('magnetics'))
        inv_layout.addWidget(self.btn_invert_mag)

        inversion_options = QFormLayout()
        self.spin_inversion_sigma = QDoubleSpinBox()
        self.spin_inversion_sigma.setRange(1e-6, 1e6)
        self.spin_inversion_sigma.setDecimals(6)
        self.spin_inversion_sigma.setValue(1.0)
        self.spin_inversion_sigma.setToolTip("Observation standard deviation in mGal for gravity or nT for magnetics")
        inversion_options.addRow("Observation sigma (mGal / nT):", self.spin_inversion_sigma)

        self.spin_inversion_regularization = QDoubleSpinBox()
        self.spin_inversion_regularization.setRange(0.0, 1e6)
        self.spin_inversion_regularization.setDecimals(4)
        self.spin_inversion_regularization.setValue(0.0)
        self.spin_inversion_regularization.setToolTip("Dimensionless regularization strength lambda")
        inversion_options.addRow("Regularization lambda:", self.spin_inversion_regularization)

        self.check_fit_geometry = QCheckBox("Fit constrained depth shift and scale")
        self.check_fit_geometry.setChecked(False)
        inversion_options.addRow(self.check_fit_geometry)
        inv_layout.addLayout(inversion_options)

        left_panel.addWidget(inv_group)

        pdf_btn = QPushButton("📄 Generate PDF Report")
        pdf_btn.clicked.connect(self.generate_pdf)
        left_panel.addWidget(pdf_btn)

        main_layout.addLayout(left_panel, stretch=1)

        # Right Panel: Plot Canvas
        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("profilePlotCanvas")
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setObjectName("analysisWorkspaceTabs")
        self.analysis_tabs.setDocumentMode(True)
        profile_page = QWidget()
        profile_layout = QVBoxLayout(profile_page)
        profile_layout.setContentsMargins(4, 4, 4, 4)
        self.profile_toolbar = NavigationToolbar2QT(self.canvas, profile_page)
        profile_layout.addWidget(self.profile_toolbar)
        profile_layout.addWidget(self.canvas, stretch=1)
        self.analysis_tabs.addTab(profile_page, "Profiles and 2D section")
        self.prism_view = FinitePrismView(self)
        self.analysis_tabs.addTab(self.prism_view, "3D finite-prism view")
        main_layout.addWidget(self.analysis_tabs, stretch=3)

        self.studio_workspace = StudioWorkspace(self)
        self.studio_workspace.action_requested.connect(self._handle_studio_action)
        self.studio_workspace.item_activated.connect(self._activate_project_item)
        self.project_dock = QDockWidget("Project Explorer", self)
        self.project_dock.setObjectName("projectExplorerDock")
        self.project_dock.setWidget(self.studio_workspace)
        self.project_dock.setMinimumWidth(250)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.project_dock)
        self.resizeDocks([self.project_dock], [285], Qt.Orientation.Horizontal)
        self.menuBar().addMenu("View").addAction(self.project_dock.toggleViewAction())

        self.ax_grav = self.figure.add_subplot(311)
        self.ax_mag = self.figure.add_subplot(312)
        self.ax_body = self.figure.add_subplot(313)
        self.figure.tight_layout()

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.refresh_body_list()
        self.recalculate_and_plot()
        self._initializing = False
        self.refresh_project_explorer()
        self.statusBar().showMessage("Ready — project recovery is enabled" if self._recovery_enabled else "Ready")
        if self._recovery_enabled:
            QTimer.singleShot(0, self._offer_recovery)

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        self.save_project_action = QAction("Save Project…", self)
        self.save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(self.save_project_action)

        self.load_project_action = QAction("Load Project…", self)
        self.load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(self.load_project_action)

        file_menu.addSeparator()

        import_grid_action = QAction("Import 3D Map & Slice Profile (XYZ)", self)
        import_grid_action.triggered.connect(self.import_3d_grid_csv)
        file_menu.addAction(import_grid_action)

        file_menu.addSeparator()
        self.import_shared_model_action = QAction("Import Shared Geological Model…", self)
        self.import_shared_model_action.triggered.connect(self.import_shared_geological_model)
        file_menu.addAction(self.import_shared_model_action)
        self.export_shared_model_action = QAction("Export Shared Geological Model…", self)
        self.export_shared_model_action.triggered.connect(self.export_shared_geological_model)
        file_menu.addAction(self.export_shared_model_action)

        self.interpret_grid_action = QAction("Interpret Regular 3D Grid (XYZ)…", self)
        self.interpret_grid_action.triggered.connect(self.open_grid_interpretation)
        file_menu.addAction(self.interpret_grid_action)

        self.import_obs_action = QAction("Import 2D Observed Line (CSV)", self)
        self.import_obs_action.triggered.connect(self.import_observed_csv)
        file_menu.addAction(self.import_obs_action)

        self.process_survey_action = QAction("Process Versioned Survey Profile…", self)
        self.process_survey_action.triggered.connect(self.open_survey_processing)
        file_menu.addAction(self.process_survey_action)

        self.export_calc_action = QAction("Export Calculated Results (CSV)", self)
        self.export_calc_action.triggered.connect(self.export_results_csv)
        file_menu.addAction(self.export_calc_action)

        file_menu.addSeparator()

        self.pdf_report_action = QAction("Generate Reproducible PDF Report", self)
        self.pdf_report_action.triggered.connect(self.generate_pdf)
        file_menu.addAction(self.pdf_report_action)

        toolbar = self.addToolBar("Studio workflow")
        toolbar.setObjectName("studioWorkflowToolbar")
        toolbar.setMovable(False)
        for action in (
            self.save_project_action,
            self.load_project_action,
            self.import_obs_action,
            self.process_survey_action,
            self.interpret_grid_action,
            self.export_calc_action,
            self.pdf_report_action,
        ):
            toolbar.addAction(action)

    def _handle_studio_action(self, action):
        handlers = {
            "import_observations": self.import_observed_csv,
            "process_survey": self.open_survey_processing,
            "interpret_grid": self.open_grid_interpretation,
            "academic_lab": self.open_academic_lab,
            "load_spatial_layer": self.load_local_spatial_layer,
            "add_body": self.add_body,
            "invert_gravity": lambda: self.run_inversion("gravity"),
            "invert_magnetic": lambda: self.run_inversion("magnetics"),
            "export_results": self.export_results_csv,
            "generate_report": self.generate_pdf,
        }
        handler = handlers.get(action)
        if handler is not None:
            handler()

    def _activate_project_item(self, item_key):
        if isinstance(item_key, tuple) and len(item_key) == 2 and item_key[0] == "body":
            index = int(item_key[1])
            if 0 <= index < self.body_list.count():
                self.body_list.setCurrentRow(index)
                self.analysis_tabs.setCurrentIndex(0)
            return
        if item_key == "survey":
            self.open_survey_processing()
        elif item_key == "grid":
            self.open_grid_interpretation()
        elif item_key == "spatial_layer":
            self.open_local_spatial_layer()
        elif item_key == "reports":
            self.generate_pdf()

    def refresh_project_explorer(self):
        if not hasattr(self, "studio_workspace"):
            return
        observed = self.obs_gz if self.obs_gz is not None else self.obs_dT
        observation_count = 0 if observed is None else int(len(observed))
        inversion_ready = (
            observed is not None
            and observation_count >= 2
            and bool(np.all(np.isfinite(observed)))
        )
        project_name = self.project_path.stem if self.project_path is not None else "Untitled project"
        grid_name = (
            Path(self.grid_source_path).name
            if self.grid_source_path is not None
            else self.grid_source_name
        )
        self.studio_workspace.update_project(
            project_name=project_name,
            project_path=self.project_path,
            dirty=self.project_dirty,
            bodies=self.bodies,
            active_body_index=self.active_body_idx,
            survey_profile=self.survey_profile,
            observation_count=observation_count,
            inversion_ready=inversion_ready,
            processing_steps=self.processing_steps,
            grid_name=grid_name,
            spatial_layer_name=(self.spatial_layer.name if self.spatial_layer is not None else None),
        )

    def _current_observations(self):
        return {
            "x": self.x_obs,
            "z": self.z_obs,
            "gravity": self.obs_gz,
            "magnetic": self.obs_dT,
        }

    def _store_project_attachment(self, source_name, payload):
        """Keep imported local bytes under a path-free, content-addressed name."""
        if len(payload) > MAX_ATTACHMENT_BYTES:
            raise ValueError("Portable project attachments are limited to 32 MiB each")
        digest = hashlib.sha256(payload).hexdigest()
        basename = Path(str(source_name)).name
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in basename
        ).strip("._")[:100]
        if not safe_name:
            safe_name = "local-data.bin"
        attachment_name = f"{digest[:12]}-{safe_name}"
        self.project_attachments[attachment_name] = bytes(payload)
        return attachment_name

    def _mark_project_dirty(self):
        if self._initializing or self._loading_project:
            return
        self.project_dirty = True
        self.refresh_project_explorer()
        self.statusBar().showMessage("Unsaved changes — a recovery snapshot will be saved automatically")
        if self._recovery_enabled:
            self._autosave_timer.start()

    def _mark_project_clean(self, *, clear_recovery=True):
        self._autosave_timer.stop()
        self.project_dirty = False
        same_as_recovery = (
            self.project_path is not None
            and self.project_path.resolve() == self._recovery_file.resolve()
        )
        if clear_recovery and self._recovery_enabled and not same_as_recovery:
            try:
                clear_recovery_snapshot(self._recovery_file)
            except OSError:
                pass
        self.refresh_project_explorer()
        self.statusBar().showMessage("Project saved", 5000)

    def _write_recovery_snapshot(self):
        if not self._recovery_enabled or not self.project_dirty:
            return False
        try:
            save_recovery_snapshot(
                self._recovery_file,
                self.bodies,
                self.field_params(),
                self._current_observations(),
                survey_profile=self.survey_profile,
                processing_steps=self.processing_steps,
                shared_model_metadata=self.shared_model_metadata,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Recovery snapshot failed: {exc}")
            return False
        self.refresh_project_explorer()
        self.statusBar().showMessage("Recovery snapshot saved", 5000)
        return True

    def _offer_recovery(self):
        if not self._recovery_enabled or not recovery_is_newer(self._recovery_file, self.project_path):
            return
        answer = QMessageBox.question(
            self,
            "Recover previous session?",
            "A newer recovery snapshot is available. Restore its model, observations, survey metadata, and processing history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                project = load_recovery_snapshot(self._recovery_file)
                self._apply_project(project, project_path=None, recovered=True)
            except Exception as exc:
                QMessageBox.warning(self, "Recovery unavailable", f"Could not restore the recovery snapshot:\n{exc}")
        else:
            try:
                clear_recovery_snapshot(self._recovery_file)
            except OSError:
                pass

    def _apply_project(self, project, *, project_path, recovered=False, attachments=None):
        if not project["bodies"]:
            raise ValueError("Project has no bodies")
        self._loading_project = True
        try:
            self.bodies = project["bodies"]
            self.shared_model_metadata = project.get("shared_model_metadata")
            fields = project["field_params"]
            for widget, key, default in (
                (self.spin_F, "F", 48_000.0),
                (self.spin_Inc, "Inc", 55.0),
                (self.spin_Dec, "Dec", 0.0),
                (self.spin_alpha, "Alpha", 0.0),
            ):
                widget.blockSignals(True)
                widget.setValue(fields.get(key, default))
                widget.blockSignals(False)
            observations = project["observations"]
            if observations["x"] is not None and observations["z"] is not None:
                self.x_obs = observations["x"]
                self.z_obs = observations["z"]
            self.obs_gz = observations["gravity"]
            self.obs_dT = observations["magnetic"]
            self.survey_profile = project["survey_profile"]
            self.processing_steps = project["processing_steps"]
            self.processing_products = []
            self.regular_profiles = []
            self.project_attachments = dict(attachments or {})
            self.grid_source_path = None
            self.grid_source_name = None
            self.grid_attachment_name = None
            self.spatial_layer = None
            self.spatial_layer_attachment_name = None
            spatial_layer_restored = False
            for step in reversed(self.processing_steps):
                if not isinstance(step, dict):
                    continue
                parameters = step.get("parameters", {})
                if not isinstance(parameters, dict):
                    continue
                if step.get("name") == "regular_grid_interpretation_source":
                    self.grid_source_name = parameters.get("source_name")
                    self.grid_attachment_name = parameters.get("attachment_name")
                elif step.get("name") == "local_spatial_layer" and not spatial_layer_restored:
                    attachment_name = parameters.get("attachment_name")
                    raw = self.project_attachments.get(attachment_name)
                    if raw is not None:
                        self.spatial_layer = load_xyz_layer(
                            raw,
                            source_crs=parameters.get("source_crs", ""),
                            target_crs=parameters.get("target_crs"),
                            units=parameters.get("units", "input units"),
                            name=parameters.get("source_name", attachment_name),
                        )
                        self.spatial_layer_attachment_name = attachment_name
                        spatial_layer_restored = True
            self.active_body_idx = 0
            self.refresh_body_list()
            self.recalculate_and_plot()
        finally:
            self._loading_project = False

        self.project_path = None if recovered else Path(project_path)
        self.project_dirty = bool(recovered)
        if recovered:
            self.statusBar().showMessage("Session recovered — save the project to keep it as a named file")
            if self._recovery_enabled:
                self._autosave_timer.start()
        else:
            self._mark_project_clean()
        self.refresh_project_explorer()

    def open_survey_processing(self):
        dialog = SurveyProcessingDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.profile is None:
            return
        selected_input = dialog.selected_model_input()
        model_observations = None
        handoff_record = None
        if selected_input is not None:
            model_observations, handoff_record = prepare_model_observations(
                dialog.profile,
                selected_input["channel"],
                selected_input["values"],
                source_column=selected_input["source_column"],
                units=selected_input["units"],
            )
        self.survey_profile = dialog.profile
        self.processing_products = list(dialog.products)
        self.regular_profiles = list(dialog.regular_profiles)
        self.processing_steps = (
            [product.to_manifest() for product in self.processing_products]
            + [
                {
                    "name": item.name,
                    "parameters": dict(item.parameters or {}),
                    "assumptions": list(item.assumptions),
                    "spacing_m": item.spacing_m,
                    "units": item.units,
                }
                for item in self.regular_profiles
            ]
        )
        if model_observations is not None:
            self.x_obs = model_observations["x"]
            self.z_obs = model_observations["z"]
            self.obs_gz = model_observations["gravity"]
            self.obs_dT = model_observations["magnetic"]
            self.processing_steps.append(handoff_record)
            self.recalculate_and_plot()
        self._mark_project_dirty()
        QMessageBox.information(
            self,
            "Survey processing saved",
            f"Saved {self.survey_profile.distance_m.size} raw survey stations and "
            f"{len(self.processing_steps)} processing records in the project state."
            + (" The selected regional residual is now active in the model and inversion views." if model_observations is not None else ""),
        )

    def run_inversion(self, field_type):
        target_data = self.obs_gz if field_type == 'gravity' else self.obs_dT
        if target_data is None:
            QMessageBox.warning(self, "Missing Observed Data",
                                f"Please import observed {field_type} CSV data first before running inversion.")
            return

        if not self.bodies:
            return

        active_body = self.bodies[self.active_body_idx]
        field_params = {
            'F': self.spin_F.value(),
            'Inc': self.spin_Inc.value(),
            'Dec': self.spin_Dec.value(),
            'alpha': self.spin_alpha.value(),
        }

        result = invert_body_parameters(
            active_body,
            self.x_obs,
            self.z_obs,
            target_data,
            all_bodies=self.bodies,
            field_type=field_type,
            field_params=field_params,
            observation_sigma=self.spin_inversion_sigma.value(),
            regularization=self.spin_inversion_regularization.value(),
            fit_geometry=self.check_fit_geometry.isChecked(),
        )
        if result.success:
            self.load_active_body_props()
            self.recalculate_and_plot()
            diagnostics = (
                f"Initial RMS: {result.initial_rms:.6g}\n"
                f"Final RMS: {result.final_rms:.6g}\n"
                f"Weighted RMS: {result.weighted_rms:.6g}\n"
                f"Evaluations: {result.function_evaluations}\n"
                f"Locally identifiable: {'yes' if result.identifiable else 'no'}"
            )
            if result.standard_errors:
                errors = ", ".join(
                    f"{name} +/- {value:.4g}" for name, value in result.standard_errors.items()
                )
                diagnostics += f"\nApproximate 1-sigma errors: {errors}"
            box = QMessageBox.information if result.identifiable else QMessageBox.warning
            box(
                self,
                "Inversion Converged" if result.identifiable else "Fit Converged; Model Is Non-unique",
                f"{result.message}\n\n{diagnostics}",
            )
        else:
            QMessageBox.critical(
                self,
                "Inversion Failed",
                f"{result.status}: {result.message}\nThe starting model was preserved.",
            )

    def on_press(self, event):
        if event.inaxes != self.ax_body or event.button != 1: return
        click_x, click_z = event.xdata, event.ydata
        if click_x is None or click_z is None: return

        ax_xlim = self.ax_body.get_xlim()
        ax_zlim = self.ax_body.get_ylim()
        x_span = abs(ax_xlim[1] - ax_xlim[0])
        z_span = abs(ax_zlim[1] - ax_zlim[0])

        min_dist = float('inf')
        found_target = None
        for b_idx, body in enumerate(self.bodies):
            for v_idx in range(len(body.x) - 1):
                dx = (body.x[v_idx] - click_x) / x_span
                dz = (body.z[v_idx] - click_z) / z_span
                dist = np.hypot(dx, dz)
                if dist < 0.04 and dist < min_dist:
                    min_dist = dist
                    found_target = (b_idx, v_idx)

        if found_target is not None:
            self.dragging_vertex = found_target
            self.active_body_idx = found_target[0]
            self.refresh_body_list()

    def on_motion(self, event):
        if self.dragging_vertex is None or event.inaxes != self.ax_body: return
        new_x, new_z = event.xdata, event.ydata
        if new_x is None or new_z is None: return

        b_idx, v_idx = self.dragging_vertex
        body = self.bodies[b_idx]
        body.x[v_idx] = new_x
        body.z[v_idx] = new_z
        if v_idx == 0:
            body.x[-1] = new_x
            body.z[-1] = new_z
        self.recalculate_and_plot()

    def on_release(self, event):
        self.dragging_vertex = None

    def import_3d_grid_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import 3D Grid Map (XYZ)", "",
                                                   "CSV Files (*.csv);;Text Files (*.txt)")
        if not file_path: return
        try:
            x, y, v = [], [], []
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith('#'): continue
                    try:
                        x.append(float(row[0]))
                        y.append(float(row[1]))
                        v.append(float(row[2]))
                    except ValueError:
                        continue  # Safely skip headers or malformed text rows

            dialog = MapProfileDialog(self, np.array(x), np.array(y), np.array(v))
            if dialog.exec() and dialog.profile_dist is not None:
                self.x_obs = dialog.profile_dist
                self.z_obs = np.zeros_like(self.x_obs)
                self.obs_gz = dialog.profile_v
                self.recalculate_and_plot()
                QMessageBox.information(self, "Success", "2D Profile extracted from 3D Map.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not process grid:\n{str(e)}")

    def open_grid_interpretation(self):
        try:
            file_path = None
            attachment_name = self.grid_attachment_name
            if attachment_name and attachment_name in self.project_attachments:
                payload = self.project_attachments[attachment_name]
                source_name = self.grid_source_name or attachment_name
            else:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Interpret regular XYZ grid", "",
                    "CSV / text grids (*.csv *.txt *.dat);;All files (*)"
                )
                if not file_path:
                    return
                source = Path(file_path)
                if source.stat().st_size > MAX_ATTACHMENT_BYTES:
                    raise ValueError("Portable grid attachments are limited to 32 MiB")
                with source.open("rb") as stream:
                    payload = stream.read(MAX_ATTACHMENT_BYTES + 1)
                source_name = source.name
            x, y, values = [], [], []
            for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    row = stripped.replace(",", " ").split()
                    x.append(float(row[0]))
                    y.append(float(row[1]))
                    values.append(float(row[2]))
                except (ValueError, IndexError):
                    if not x and not values:
                        continue  # A single leading header row is accepted.
                    raise ValueError(f"Grid row {line_number} is not a numeric XYZ record")
            if len(values) < 64:
                raise ValueError("Regular grid interpretation needs at least 8 by 8 complete XYZ cells.")
            dialog = GridInterpretationDialog(
                self, np.asarray(x), np.asarray(y), np.asarray(values),
                field_inclination_deg=self.spin_Inc.value(), field_declination_deg=self.spin_Dec.value(),
            )
            dialog.exec()
            self.grid_source_path = Path(file_path) if file_path else None
            self.grid_source_name = source_name
            attachment_name = attachment_name or self._store_project_attachment(source_name, payload)
            self.grid_attachment_name = attachment_name
            self.processing_steps.append({
                "name": "regular_grid_interpretation_source",
                "parameters": {
                    "source_name": source_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "attachment_name": attachment_name,
                    "units": str(dialog.units.currentData()),
                    "coordinate_reference": dialog.coordinate_reference.text().strip(),
                },
            })
            self._mark_project_dirty()
        except Exception as exc:
            QMessageBox.critical(self, "Grid interpretation input error", str(exc))

    def open_academic_lab(self):
        dialog = AcademicLabDialog(self)
        dialog.exec()

    def load_local_spatial_layer(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load local XYZ spatial layer", "",
            "XYZ / CSV layers (*.xyz *.csv *.txt);;All files (*)",
        )
        if not file_path:
            return
        try:
            source = Path(file_path)
            if source.stat().st_size > MAX_ATTACHMENT_BYTES:
                raise ValueError("Portable spatial layer attachments are limited to 32 MiB")
            with source.open("rb") as stream:
                payload = stream.read(MAX_ATTACHMENT_BYTES + 1)
            source_crs, accepted = QInputDialog.getText(
                self,
                "Spatial layer source CRS",
                "Enter the source CRS (for example EPSG:32630):",
            )
            if not accepted:
                return
            target_crs, accepted = QInputDialog.getText(
                self,
                "Spatial layer project CRS",
                "Enter the target CRS (use the source CRS to keep coordinates unchanged):",
                text=source_crs,
            )
            if not accepted:
                return
            units, accepted = QInputDialog.getText(
                self,
                "Spatial layer value units",
                "Enter the value units (for example mGal or nT):",
                text="input units",
            )
            if not accepted:
                return
            layer = load_xyz_layer(
                payload,
                source_crs=source_crs,
                target_crs=target_crs,
                units=units,
                name=Path(file_path).name,
            )
            attachment_name = self._store_project_attachment(Path(file_path).name, payload)
            self.spatial_layer = layer
            self.spatial_layer_attachment_name = attachment_name
            self.processing_steps.append({
                "name": "local_spatial_layer",
                "parameters": {
                    "source_name": layer.name,
                    "attachment_name": attachment_name,
                    "source_sha256": layer.source_sha256,
                    "source_crs": layer.source_crs,
                    "target_crs": layer.target_crs,
                    "units": layer.units,
                    "point_count": int(layer.x.size),
                    "local_only": True,
                },
            })
            self._mark_project_dirty()
            self.open_local_spatial_layer()
        except Exception as exc:
            QMessageBox.warning(self, "Spatial layer import error", str(exc))

    def open_local_spatial_layer(self):
        if self.spatial_layer is None:
            QMessageBox.information(self, "Local spatial layer", "Load a local XYZ layer first.")
            return
        dialog = SpatialLayerDialog(self, self.spatial_layer)
        dialog.exec()

    def import_observed_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Observed Profile", "",
                                                   "CSV Files (*.csv);;Text Files (*.txt)")
        if not file_path: return
        try:
            source = Path(file_path)
            if source.stat().st_size > MAX_ATTACHMENT_BYTES:
                raise ValueError("Portable observation attachments are limited to 32 MiB")
            with source.open("rb") as stream:
                payload = stream.read(MAX_ATTACHMENT_BYTES + 1)
            x_vals, gz_vals, dT_vals, z_vals = [], [], [], []
            reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
            for row in reader:
                if not row or row[0].startswith('#'): continue
                try:
                    nums = [float(val) for val in row]
                except ValueError:
                    continue  # Safely skip headers or malformed text rows

                x_vals.append(nums[0])
                gz_vals.append(nums[1] if len(nums) > 1 else 0.0)
                dT_vals.append(nums[2] if len(nums) > 2 else 0.0)
                z_vals.append(nums[3] if len(nums) > 3 else 0.0)
            self.x_obs = np.array(x_vals, dtype=float)
            self.z_obs = np.array(z_vals, dtype=float) if z_vals and any(z_vals) else np.zeros_like(self.x_obs)
            self.obs_gz = np.array(gz_vals, dtype=float) if len(gz_vals) == len(self.x_obs) else None
            self.obs_dT = np.array(dT_vals, dtype=float) if len(dT_vals) == len(self.x_obs) else None
            attachment_name = self._store_project_attachment(source.name, payload)
            self.processing_steps.append({
                "name": "observed_profile_import",
                "parameters": {
                    "source_name": source.name,
                    "attachment_name": attachment_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "columns": ["distance_m", "gravity_mgal", "magnetic_nT", "elevation_m"],
                    "local_only": True,
                },
            })
            self.recalculate_and_plot()
            QMessageBox.information(self, "Success", f"Loaded profile with {len(self.x_obs)} points.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not read file:\n{str(e)}")

    def export_results_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "KuffGravMag_Results.csv",
                                                   "CSV Files (*.csv)")
        if not file_path: return
        engine = PotentialField2D(self.bodies)
        gz_calc = engine.calculate_gravity(self.x_obs, self.z_obs)
        _, _, dT_calc = engine.calculate_magnetics(
            self.x_obs, self.z_obs, self.spin_F.value(), self.spin_Inc.value(),
            self.spin_Dec.value(), self.spin_alpha.value(),
        )
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Distance_m", "Calculated_gz_mGal", "Calculated_dT_nT"])
                for x, gz, dT in zip(self.x_obs, gz_calc, dT_calc):
                    writer.writerow([round(x, 2), round(gz, 4), round(dT, 4)])
            QMessageBox.information(self, "Export Successful", f"Saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not save file:\n{str(e)}")

    def generate_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "KuffGravMag_Report.pdf", "PDF (*.pdf)")
        if not file_path: return

        try:
            # 1. Safely get the macOS approved temporary directory
            temp_fig = os.path.join(tempfile.gettempdir(), "temp_report_fig.png")

            # 2. Save the figure safely inside the try block
            self.figure.savefig(temp_fig, dpi=300, bbox_inches='tight')

            # 3. Generate the actual PDF
            generate_pdf_report(
                file_path,
                temp_fig,
                self.bodies,
                self.field_params(),
                observations=self._current_observations(),
                survey_profile=self.survey_profile,
                processing_steps=self.processing_steps,
                project_name=self.project_path.stem if self.project_path else "Unsaved project",
            )

            # 4. Clean up the temporary image
            if os.path.exists(temp_fig):
                os.remove(temp_fig)

            QMessageBox.information(self, "Success", f"PDF saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not build PDF:\n{str(e)}")

    def field_params(self):
        return {
            "F": self.spin_F.value(),
            "Inc": self.spin_Inc.value(),
            "Dec": self.spin_Dec.value(),
            "Alpha": self.spin_alpha.value(),
        }

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save KuffGravMag Project", "KuffGravMag_Project.kgmproj",
            "Portable KuffGravMag project (*.kgmproj);;JSON project (*.json)",
        )
        if not file_path:
            return
        try:
            self._save_project_to_path(file_path)
            QMessageBox.information(self, "Project Saved", f"Saved project to {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Project Save Error", f"Could not save project:\n{exc}")

    def _save_project_to_path(self, file_path):
        path = Path(file_path)
        project_data = project_to_dict(
            self.bodies,
            self.field_params(),
            self._current_observations(),
            survey_profile=self.survey_profile,
            processing_steps=self.processing_steps,
            shared_model_metadata=self.shared_model_metadata,
        )
        if path.suffix.lower() == ".kgmproj":
            save_project_bundle(path, project_data, self.project_attachments)
        else:
            write_project(
                path,
                self.bodies,
                self.field_params(),
                self._current_observations(),
                survey_profile=self.survey_profile,
                processing_steps=self.processing_steps,
                shared_model_metadata=self.shared_model_metadata,
            )
        self.project_path = path
        self._mark_project_clean()
        return path

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load KuffGravMag Project", "KuffGravMag_Project.json",
            "KuffGravMag projects (*.kgmproj *.json)",
        )
        if not file_path:
            return
        try:
            if Path(file_path).suffix.lower() == ".kgmproj":
                bundle = load_project_bundle(file_path)
                project = bundle["project"]
                attachments = bundle["attachments"]
            else:
                project = read_project(file_path)
                attachments = {}
            self._apply_project(project, project_path=file_path, attachments=attachments)
            QMessageBox.information(self, "Project Loaded", f"Loaded project from {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Project Load Error", f"Could not load project:\n{exc}")

    def import_shared_geological_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Shared Geological Model", "",
            "Shared geological models (*.kgsm.json *.json);;JSON files (*.json)",
        )
        if not file_path:
            return
        try:
            result = import_shared_model(Path(file_path))
            if not result.bodies:
                raise ValueError("Shared geological model contains no supported 2D polygons")
            self._loading_project = True
            try:
                self.bodies = list(result.bodies)
                self.shared_model_metadata = result.model_metadata
                if result.magnetic_field is not None:
                    field = result.magnetic_field
                    for widget, value in (
                        (self.spin_F, field["intensity_nT"]),
                        (self.spin_Inc, field["inclination_deg"]),
                        (self.spin_Dec, field["declination_deg"]),
                        (self.spin_alpha, ((field["profileAzimuth_deg"] + 180.0) % 360.0) - 180.0),
                    ):
                        widget.blockSignals(True)
                        widget.setValue(value)
                        widget.blockSignals(False)
                self.active_body_idx = 0
                self.refresh_body_list()
                self.load_active_body_props()
                self.recalculate_and_plot()
            finally:
                self._loading_project = False
            self.project_path = None
            self._mark_project_dirty()
            self.refresh_project_explorer()
            detail = "\n\n".join(result.warnings) if result.warnings else "No conversion warnings."
            QMessageBox.information(
                self, "Shared Geological Model Imported",
                f"Imported {len(result.bodies)} supported polygon body/bodies. Existing observed data remains attached to the project.\n\n{detail}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Shared Model Import Error", f"Could not import the shared geological model:\n{exc}")

    def export_shared_geological_model(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Shared Geological Model", "KuffGravMag_Shared_Model.kgsm.json",
            "Shared geological model (*.kgsm.json);;JSON file (*.json)",
        )
        if not file_path:
            return
        try:
            field = {
                "intensity_nT": self.spin_F.value(),
                "inclination_deg": self.spin_Inc.value(),
                "declination_deg": self.spin_Dec.value(),
                "profileAzimuth_deg": self.spin_alpha.value() % 360.0,
            }
            model, warnings = export_kgm_model(self.bodies, field, model_metadata=self.shared_model_metadata)
            validate_shared_model(model)
            Path(file_path).write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            detail = "\n\n".join(warnings) if warnings else "No conversion warnings."
            QMessageBox.information(self, "Shared Geological Model Exported", f"Saved {file_path}.\n\n{detail}")
        except Exception as exc:
            QMessageBox.critical(self, "Shared Model Export Error", f"Could not export the shared geological model:\n{exc}")

    def closeEvent(self, event):
        if self.project_dirty and self._recovery_enabled:
            self._write_recovery_snapshot()
        event.accept()

    def refresh_body_list(self):
        self.body_list.blockSignals(True)
        self.body_list.clear()
        for b in self.bodies: self.body_list.addItem(b.name)
        self.body_list.blockSignals(False)
        if self.bodies:
            idx = min(self.active_body_idx, len(self.bodies) - 1)
            self.body_list.setCurrentRow(idx)
            self.load_active_body_props()

    def on_body_selected(self, row):
        if 0 <= row < len(self.bodies):
            self.active_body_idx = row
            self.load_active_body_props()
            self.recalculate_and_plot(mark_dirty=False)
            QTimer.singleShot(0, self.refresh_project_explorer)

    def load_active_body_props(self):
        if not self.bodies or self.active_body_idx >= len(self.bodies): return
        body = self.bodies[self.active_body_idx]
        self.body_props_group.setTitle(f"Properties: {body.name}")
        for spin, val in [(self.spin_rho, body.density), (self.spin_k, body.k_succ),
                          (self.spin_Jr, body.J_r), (self.spin_Ir, body.I_r),
                          (self.spin_Dr, body.D_r)]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        mode_index = self.combo_strike_mode.findData(body.strike_mode)
        self.combo_strike_mode.blockSignals(True)
        self.combo_strike_mode.setCurrentIndex(max(0, mode_index))
        self.combo_strike_mode.blockSignals(False)
        for spin, val, fallback in (
            (self.spin_ymin, body.y_min, -500.0),
            (self.spin_ymax, body.y_max, 500.0),
        ):
            spin.blockSignals(True)
            spin.setValue(val if np.isfinite(val) else fallback)
            spin.blockSignals(False)
        finite_strike = body.strike_mode == "finite"
        self.spin_ymin.setEnabled(finite_strike)
        self.spin_ymax.setEnabled(finite_strike)
        self.combo_color.blockSignals(True)
        c_idx = COLOR_OPTIONS.index(body.color) if body.color in COLOR_OPTIONS else 0
        self.combo_color.setCurrentIndex(c_idx)
        self.combo_color.blockSignals(False)

    def update_active_body_props(self):
        if not self.bodies or self.active_body_idx >= len(self.bodies): return
        body = self.bodies[self.active_body_idx]
        body.density = self.spin_rho.value()
        body.k_succ = self.spin_k.value()
        body.J_r = self.spin_Jr.value()
        body.I_r = self.spin_Ir.value()
        body.D_r = self.spin_Dr.value()
        body.strike_mode = self.combo_strike_mode.currentData()
        if body.strike_mode == "finite":
            body.y_min = self.spin_ymin.value()
            body.y_max = self.spin_ymax.value()
        body.color = self.combo_color.currentText()
        self.recalculate_and_plot()

    def add_body(self):
        n = len(self.bodies) + 1
        x_new = np.array([-100, 100, 150, -50, -100], dtype=float) + (n * 50)
        z_new = np.array([400, 400, 700, 650, 400], dtype=float)
        new_body = Body2D(f"Body {n}", x_new, z_new, density=250.0, k_succ=0.015, y_min=-500, y_max=500,
                          color=COLOR_OPTIONS[n % len(COLOR_OPTIONS)])
        self.bodies.append(new_body)
        self.active_body_idx = len(self.bodies) - 1
        self.refresh_body_list()
        self.recalculate_and_plot()

    def remove_body(self):
        if len(self.bodies) <= 1: return
        self.bodies.pop(self.active_body_idx)
        self.active_body_idx = max(0, self.active_body_idx - 1)
        self.refresh_body_list()
        self.recalculate_and_plot()

    def on_param_changed(self):
        self.recalculate_and_plot()

    def recalculate_and_plot(self, *, mark_dirty=True):
        engine = PotentialField2D(self.bodies)
        gz_total = engine.calculate_gravity(self.x_obs, self.z_obs)
        _, _, _, dT_total = engine.calculate_magnetics_components(
            self.x_obs, self.z_obs, self.spin_F.value(), self.spin_Inc.value(),
            self.spin_Dec.value(), self.spin_alpha.value(),
        )
        self.plot_results(gz_total, dT_total)
        if mark_dirty:
            self._mark_project_dirty()

    def plot_results(self, gz, dT):
        self.ax_grav.clear()
        self.ax_mag.clear()
        self.ax_body.clear()

        uses_finite_strike = any(body.strike_mode == "finite" for body in self.bodies)
        uses_infinite_strike = any(body.strike_mode == "infinite" for body in self.bodies)
        if uses_finite_strike:
            gravity_label = (
                "Mixed Infinite/Finite Strike gravity"
                if uses_infinite_strike else "Finite Strike gravity (validated prism model)"
            )
            finite_text = (
                "Finite Strike gravity uses closed-form polygonal-prism face integrals "
                "validated against rectangular-prism and independent volume references."
            )
            if uses_infinite_strike:
                finite_text += " Infinite Strike bodies use the benchmarked positive-down 2D kernel."
            self.gravity_status_label.setText("Gravity: " + finite_text)
            self.gravity_status_label.setStyleSheet("color: #276749;")
        else:
            gravity_label = "Infinite Strike gravity (benchmarked 2D model)"
            self.gravity_status_label.setText(
                "Gravity: Infinite Strike mode uses the benchmarked positive-down 2D kernel."
            )
            self.gravity_status_label.setStyleSheet("color: #276749;")

        if uses_finite_strike and uses_infinite_strike:
            self.magnetic_status_label.setText(
                "Magnetics: mixed Infinite and Finite Strike bodies; finite bodies use "
                "closed-form 3D prism face integrals; total-field output is linearized."
            )
        elif uses_finite_strike:
            self.magnetic_status_label.setText(
                "Magnetics: Finite Strike mode uses closed-form 3D prism face integrals; "
                "uniform magnetization, no demagnetization, linearized total field."
            )
        else:
            self.magnetic_status_label.setText(
                "Magnetics: Infinite Strike mode uses the benchmarked 2D boundary-charge kernel."
            )
        self.magnetic_status_label.setStyleSheet("color: #276749;")

        if self.obs_gz is not None:
            self.ax_grav.plot(self.x_obs, self.obs_gz, 'r.', label='Observed (g)')
        self.ax_grav.plot(self.x_obs, gz, 'b-', lw=1.8, label=gravity_label)
        self.ax_grav.set_ylabel("Gravity (mGal)")
        self.ax_grav.grid(True)
        self.ax_grav.legend(loc='upper right')

        if self.obs_dT is not None:
            self.ax_mag.plot(self.x_obs, self.obs_dT, 'r.', label='Observed (T)')
        magnetic_label = (
            "Finite Strike calculated dT"
            if uses_finite_strike else "Infinite Strike calculated dT"
        )
        self.ax_mag.plot(self.x_obs, dT, 'm-', lw=1.8, label=magnetic_label)
        self.ax_mag.set_ylabel("Magnetics (nT)")
        self.ax_mag.grid(True)
        self.ax_mag.legend(loc='upper right')

        if np.any(self.z_obs != 0):
            self.ax_body.plot(self.x_obs, self.z_obs, 'k--', lw=1.5, label='Topography')

        for idx, body in enumerate(self.bodies):
            is_active = (idx == self.active_body_idx)
            self.ax_body.plot(body.x, body.z, '-o', color=body.color, lw=2.5 if is_active else 1.0,
                              label=f"{body.name} ({body.strike_mode}: Y=[{body.y_min:.0f},{body.y_max:.0f}])",
                              markersize=6)
            self.ax_body.fill(body.x, body.z, color=body.color, alpha=0.7 if is_active else 0.3)

        self.ax_body.invert_yaxis()
        self.ax_body.set_xlabel("Distance (m)")
        self.ax_body.set_ylabel("Depth (m)")
        self.ax_body.grid(True)
        self.ax_body.legend(loc='lower right', fontsize='small')
        self.canvas.draw()
        if hasattr(self, "prism_view"):
            self.prism_view.update_scene(self.bodies, self.active_body_idx)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("KuffGravMag")
    app.setApplicationName("KuffGravMag Studio")
    if not app.screens():
        sys.stderr.write(
            "KuffGravMag could not find an active display. The Qt platform plugin "
            "loaded, but this process has no available screens. Launch the desktop "
            "app from a logged-in graphical session.\n"
        )
        raise SystemExit(2)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
