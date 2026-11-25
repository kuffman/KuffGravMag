"""Interactive Qt dialog for synthetic gravity and magnetic teaching models."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engine.academic import (
    MODEL_EQUATIONS,
    MODEL_LIMITATIONS,
    export_teaching_experiment_csv,
    invert_sphere_gravity_density_depth,
    simulate_teaching_experiment,
    sphere_density_depth_tradeoff,
)


class AcademicLabDialog(QDialog):
    """Run seeded analytical experiments, fit a sphere, and export provenance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Academic & Research Laboratory — Synthetic Models")
        self.resize(1180, 820)
        self.experiment = None
        self.inversion_result = None

        outer = QVBoxLayout(self)
        disclaimer = QLabel(
            "Synthetic teaching data only. The example coordinate anchor is EPSG:32630 near 3°W, 6°N; "
            "it is not a measured Ghana survey or a claim about local geology."
        )
        disclaimer.setWordWrap(True)
        outer.addWidget(disclaimer)

        content = QHBoxLayout()
        outer.addLayout(content, stretch=1)
        controls = QWidget(self)
        form = QFormLayout(controls)
        self.model = QComboBox()
        for label, key in (
            ("Sphere", "sphere"),
            ("Infinite horizontal cylinder", "cylinder"),
            ("Infinite-strike vertical dyke", "dyke"),
        ):
            self.model.addItem(label, key)
        form.addRow("Analytical source:", self.model)
        self.field = QComboBox()
        self.field.addItem("Gravity", "gravity")
        self.field.addItem("Magnetic total-field anomaly", "magnetics")
        form.addRow("Response:", self.field)

        self.radius = self._double_spin(1.0, 5_000.0, 120.0, 1.0)
        form.addRow("Sphere / cylinder radius (m):", self.radius)
        self.center_depth = self._double_spin(2.0, 20_000.0, 420.0, 10.0)
        form.addRow("Circle centre depth (m):", self.center_depth)
        self.density = self._double_spin(-5_000.0, 5_000.0, 550.0, 25.0, decimals=2)
        form.addRow("Density contrast (kg/m³):", self.density)
        self.dyke_width = self._double_spin(1.0, 5_000.0, 150.0, 10.0)
        form.addRow("Dyke width (m):", self.dyke_width)
        self.dyke_top = self._double_spin(0.0, 10_000.0, 180.0, 10.0)
        form.addRow("Dyke top depth (m):", self.dyke_top)
        self.dyke_bottom = self._double_spin(1.0, 20_000.0, 600.0, 10.0)
        form.addRow("Dyke bottom depth (m):", self.dyke_bottom)

        self.field_strength = self._double_spin(10_000.0, 70_000.0, 49_000.0, 500.0)
        form.addRow("Ambient field (nT):", self.field_strength)
        self.field_inclination = self._double_spin(-90.0, 90.0, 8.0, 1.0)
        form.addRow("Field inclination (deg):", self.field_inclination)
        self.field_declination = self._double_spin(-180.0, 180.0, -2.0, 1.0)
        form.addRow("Field declination (deg):", self.field_declination)
        self.profile_azimuth = self._double_spin(-180.0, 180.0, 90.0, 1.0)
        form.addRow("Profile azimuth (deg):", self.profile_azimuth)
        self.susceptibility = self._double_spin(-1.0, 1.0, 0.03, 0.005, decimals=5)
        form.addRow("Susceptibility (SI):", self.susceptibility)
        self.remanence = self._double_spin(0.0, 100.0, 0.0, 0.1)
        form.addRow("Remanence (A/m):", self.remanence)
        self.remanent_inclination = self._double_spin(-90.0, 90.0, 20.0, 1.0)
        form.addRow("Remanent inclination (deg):", self.remanent_inclination)
        self.remanent_declination = self._double_spin(-180.0, 180.0, 30.0, 1.0)
        form.addRow("Remanent declination (deg):", self.remanent_declination)

        self.noise_sigma = self._double_spin(0.0, 100_000.0, 0.04, 0.01, decimals=4)
        form.addRow("Gaussian noise σ (mGal / nT):", self.noise_sigma)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(20261124)
        form.addRow("Random seed:", self.seed)
        self.station_count = QSpinBox()
        self.station_count.setRange(11, 1001)
        self.station_count.setSingleStep(10)
        self.station_count.setValue(201)
        form.addRow("Profile stations:", self.station_count)
        self.alternative_depth = self._double_spin(5.0, 20_000.0, 650.0, 10.0)
        form.addRow("Deeper alternative sphere depth (m):", self.alternative_depth)
        content.addWidget(controls, stretch=0)

        plot_container = QWidget(self)
        plot_layout = QVBoxLayout(plot_container)
        self.figure = Figure(figsize=(7.5, 5.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.status = QLabel("Choose a source and run a seeded experiment.")
        self.status.setWordWrap(True)
        self.equation_note = QLabel()
        self.equation_note.setWordWrap(True)
        self.equation_note.setAlignment(Qt.AlignmentFlag.AlignTop)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=1)
        plot_layout.addWidget(self.status)
        plot_layout.addWidget(self.equation_note)
        content.addWidget(plot_container, stretch=1)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Run seeded experiment")
        self.run_button.clicked.connect(self._run_from_controls)
        actions.addWidget(self.run_button)
        self.export_button = QPushButton("Export synthetic CSV…")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        actions.addWidget(self.export_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        outer.addLayout(actions)

        self.model.currentIndexChanged.connect(self._update_control_state)
        self.field.currentIndexChanged.connect(self._update_control_state)
        self._update_control_state()

    @staticmethod
    def _double_spin(minimum, maximum, value, step, *, decimals=2):
        widget = QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setDecimals(int(decimals))
        widget.setSingleStep(float(step))
        widget.setValue(float(value))
        return widget

    def _update_control_state(self, *_args):
        is_dyke = self.model.currentData() == "dyke"
        is_magnetic = self.field.currentData() == "magnetics"
        for widget in (self.radius, self.center_depth):
            widget.setEnabled(not is_dyke)
        for widget in (self.dyke_width, self.dyke_top, self.dyke_bottom):
            widget.setEnabled(is_dyke)
        for widget in (
            self.field_strength,
            self.field_inclination,
            self.field_declination,
            self.profile_azimuth,
            self.susceptibility,
            self.remanence,
            self.remanent_inclination,
            self.remanent_declination,
        ):
            widget.setEnabled(is_magnetic)
        self.alternative_depth.setEnabled(not is_dyke and not is_magnetic)

    def run_experiment(self):
        x = np.linspace(-1_200.0, 1_200.0, self.station_count.value())
        model_kind = self.model.currentData()
        field_type = self.field.currentData()
        parameters = {
            "density_contrast_kg_m3": self.density.value(),
            "center_x_m": 0.0,
            "center_depth_m": self.center_depth.value(),
            "radius_m": self.radius.value(),
            "dyke_width_m": self.dyke_width.value(),
            "dyke_top_m": self.dyke_top.value(),
            "dyke_bottom_m": self.dyke_bottom.value(),
            "field_strength_nT": self.field_strength.value(),
            "field_inclination_deg": self.field_inclination.value(),
            "field_declination_deg": self.field_declination.value(),
            "profile_azimuth_deg": self.profile_azimuth.value(),
            "susceptibility_si": self.susceptibility.value(),
            "remanence_a_m": self.remanence.value(),
            "remanent_inclination_deg": self.remanent_inclination.value(),
            "remanent_declination_deg": self.remanent_declination.value(),
        }
        self.experiment = simulate_teaching_experiment(
            model_kind,
            field_type,
            x_m=x,
            noise_sigma=self.noise_sigma.value(),
            seed=self.seed.value(),
            **parameters,
        )
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(x, self.experiment.clean_response, color="tab:blue", label="Noise-free model")
        axis.plot(
            x,
            self.experiment.observed_response,
            "o",
            color="tab:orange",
            markersize=3,
            label="Synthetic observations",
        )
        units = self.experiment.response_units
        self.inversion_result = None
        tradeoff_text = ""
        if model_kind == "sphere" and field_type == "gravity":
            alternative_depth = max(
                self.alternative_depth.value(),
                self.center_depth.value() + 1.0e-3,
                self.radius.value() + 1.0e-3,
            )
            alternative = sphere_density_depth_tradeoff(
                x,
                radius_m=self.radius.value(),
                shallow_density_kg_m3=self.density.value(),
                shallow_depth_m=self.center_depth.value(),
                deep_depth_m=alternative_depth,
            )
            axis.plot(
                x,
                alternative["deep_response_mgal"],
                "--",
                color="tab:green",
                label="Deeper sphere with adjusted density (same center station)",
            )
            fit_density = self.density.value() * 0.7
            if abs(fit_density) < 1.0:
                fit_density = 200.0
            initial_depth = min(self.center_depth.value() + self.radius.value(), 9_000.0)
            self.inversion_result = invert_sphere_gravity_density_depth(
                x,
                self.experiment.observed_response,
                radius_m=self.radius.value(),
                observation_sigma_mgal=max(self.noise_sigma.value(), 0.05),
                initial_density_kg_m3=fit_density,
                initial_depth_m=max(initial_depth, self.radius.value() + 0.1),
            )
            fit = self.inversion_result
            uncertainty = (
                "local σ: "
                f"density {fit.standard_errors['density_kg_m3']:.3g} kg/m³; "
                f"depth {fit.standard_errors['center_depth_m']:.3g} m"
                if fit.standard_errors else "local uncertainty unavailable (rank deficient or no residual degrees of freedom)"
            )
            tradeoff_text = (
                f"Sphere fit: ρ={fit.density_kg_m3:.2f} kg/m³, depth={fit.center_depth_m:.2f} m; "
                f"rank {fit.numerical_rank}/2, identifiable={fit.identifiable}, {uncertainty}. "
                f"Density-depth alternative at {alternative_depth:.1f} m matches the center station within "
                f"{abs(alternative['single_station_difference_mgal']):.3g} mGal but has "
                f"{alternative['profile_relative_rms_difference']:.3f} relative profile RMS difference."
            )
        axis.set_xlabel("Profile distance X (m)")
        axis.set_ylabel(f"Response ({units})")
        axis.set_title(f"Synthetic {model_kind} {field_type} experiment · seed {self.seed.value()}")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status.setText(
            f"Seed {self.experiment.seed}; σ={self.experiment.noise_sigma:g} {units}. "
            f"Synthetic inputs and assumptions are embedded in the CSV export. {tradeoff_text}"
        )
        equation = MODEL_EQUATIONS[f"{model_kind}_{field_type}"]
        self.equation_note.setText(
            f"Equation: {equation}<br><br>Limitations: " + " ".join(MODEL_LIMITATIONS)
        )
        self.export_button.setEnabled(True)
        return self.experiment

    def _run_from_controls(self):
        try:
            self.run_experiment()
        except Exception as exc:
            QMessageBox.warning(self, "Synthetic experiment unavailable", str(exc))

    def _export(self):
        if self.experiment is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export reproducible synthetic experiment",
            "KuffGravMag_Synthetic_Experiment.csv",
            "CSV files (*.csv)",
        )
        if not filename:
            return
        try:
            with open(filename, "wb") as output:
                output.write(export_teaching_experiment_csv(self.experiment))
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
