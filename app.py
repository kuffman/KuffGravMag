import sys
import csv
import os
import tempfile
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QDoubleSpinBox, QPushButton, QGroupBox, QListWidget,
    QComboBox, QLabel, QFileDialog, QMessageBox, QDialog, QCheckBox,
    QLineEdit, QSpinBox, QTextEdit, QDialogButtonBox, QGridLayout
)
from PyQt6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from engine.forward import Body2D, PotentialField2D
from engine.report import generate_pdf_report
from engine.grid import extract_profile_from_xyz
from engine.inversion import invert_body_parameters
from engine.project import load_project as read_project, save_project as write_project
from engine.survey import SurveyProfile, BaseStationSeries, export_processing_bundle
from engine.processing import (
    apply_gravity_reductions,
    apply_magnetic_diurnal_correction,
    separate_polynomial_regional,
    interpolate_profile,
    RegularProfile,
)
from engine.spectral import transform_profile

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

        self.preview_figure = Figure(figsize=(7, 2.4), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        root.addWidget(self.preview_canvas)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Loaded profile details and processing assumptions will appear here.")
        root.addWidget(self.log, stretch=1)
        self.note = QLabel(
            "The processing package records raw channels, units, assumptions, parameters, and hashes. "
            "Terrain corrections and IGRF/main-field subtraction are not included."
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
            self._log(f"Fit degree-{self.degree.value()} regional and residual.\n" + "\n".join(product.assumptions))
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.warning(self, "Regional separation error", str(exc))

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KuffGravMag — 2.5D Potential Field Workbench & Inversion")
        self.resize(1350, 920)

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
        main_layout.addWidget(self.canvas, stretch=3)

        self.ax_grav = self.figure.add_subplot(311)
        self.ax_mag = self.figure.add_subplot(312)
        self.ax_body = self.figure.add_subplot(313)
        self.figure.tight_layout()

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.refresh_body_list()
        self.recalculate_and_plot()

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        save_project_action = QAction("Save Project…", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        load_project_action = QAction("Load Project…", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)

        file_menu.addSeparator()

        import_grid_action = QAction("Import 3D Map & Slice Profile (XYZ)", self)
        import_grid_action.triggered.connect(self.import_3d_grid_csv)
        file_menu.addAction(import_grid_action)

        import_obs_action = QAction("Import 2D Observed Line (CSV)", self)
        import_obs_action.triggered.connect(self.import_observed_csv)
        file_menu.addAction(import_obs_action)

        process_survey_action = QAction("Process Versioned Survey Profile…", self)
        process_survey_action.triggered.connect(self.open_survey_processing)
        file_menu.addAction(process_survey_action)

        export_calc_action = QAction("Export Calculated Results (CSV)", self)
        export_calc_action.triggered.connect(self.export_results_csv)
        file_menu.addAction(export_calc_action)

        file_menu.addSeparator()

        pdf_report_action = QAction("Generate Executive PDF Report", self)
        pdf_report_action.triggered.connect(self.generate_pdf)
        file_menu.addAction(pdf_report_action)

    def open_survey_processing(self):
        dialog = SurveyProcessingDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.profile is None:
            return
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
        QMessageBox.information(
            self,
            "Survey processing saved",
            f"Saved {self.survey_profile.distance_m.size} raw survey stations and "
            f"{len(self.processing_steps)} processing records in the project state.",
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

    def import_observed_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Observed Profile", "",
                                                   "CSV Files (*.csv);;Text Files (*.txt)")
        if not file_path: return
        try:
            x_vals, gz_vals, dT_vals, z_vals = [], [], [], []
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
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
            generate_pdf_report(file_path, temp_fig, self.bodies, self.field_params())

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
            self, "Save KuffGravMag Project", "KuffGravMag_Project.json",
            "KuffGravMag project (*.json)",
        )
        if not file_path:
            return
        observations = {
            "x": self.x_obs,
            "z": self.z_obs,
            "gravity": self.obs_gz,
            "magnetic": self.obs_dT,
        }
        try:
            write_project(
                file_path,
                self.bodies,
                self.field_params(),
                observations,
                survey_profile=self.survey_profile,
                processing_steps=self.processing_steps,
            )
            QMessageBox.information(self, "Project Saved", f"Saved project to {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Project Save Error", f"Could not save project:\n{exc}")

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load KuffGravMag Project", "KuffGravMag_Project.json",
            "KuffGravMag project (*.json)",
        )
        if not file_path:
            return
        try:
            project = read_project(file_path)
            if not project["bodies"]:
                raise ValueError("Project has no bodies")
            self.bodies = project["bodies"]
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
            self.active_body_idx = 0
            self.refresh_body_list()
            self.recalculate_and_plot()
            QMessageBox.information(self, "Project Loaded", f"Loaded project from {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Project Load Error", f"Could not load project:\n{exc}")

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

    def recalculate_and_plot(self):
        engine = PotentialField2D(self.bodies)
        gz_total = engine.calculate_gravity(self.x_obs, self.z_obs)
        _, _, _, dT_total = engine.calculate_magnetics_components(
            self.x_obs, self.z_obs, self.spin_F.value(), self.spin_Inc.value(),
            self.spin_Dec.value(), self.spin_alpha.value(),
        )
        self.plot_results(gz_total, dT_total)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
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
