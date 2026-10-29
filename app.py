import sys
import csv
import os
import tempfile
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QDoubleSpinBox, QPushButton, QGroupBox, QListWidget,
    QComboBox, QLabel, QFileDialog, QMessageBox, QDialog
)
from PyQt6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from engine.forward import Body2D, PotentialField2D
from engine.report import generate_pdf_report
from engine.grid import extract_profile_from_xyz
from engine.inversion import invert_body_parameters

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KuffGravMag — 2.5D Potential Field Workbench & Inversion")
        self.resize(1350, 920)

        self.x_obs = np.linspace(-1000, 1000, 250)
        self.z_obs = np.zeros_like(self.x_obs)
        self.obs_gz = None
        self.obs_dT = None
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

        left_panel.addWidget(global_group)

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

        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-10000, 0)
        self.spin_ymin.setValue(-500)
        self.spin_ymin.setSuffix(" m")
        self.spin_ymin.valueChanged.connect(self.update_active_body_props)
        props_layout.addRow("Strike Y-Min:", self.spin_ymin)

        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(0, 10000)
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

        import_grid_action = QAction("Import 3D Map & Slice Profile (XYZ)", self)
        import_grid_action.triggered.connect(self.import_3d_grid_csv)
        file_menu.addAction(import_grid_action)

        import_obs_action = QAction("Import 2D Observed Line (CSV)", self)
        import_obs_action.triggered.connect(self.import_observed_csv)
        file_menu.addAction(import_obs_action)

        export_calc_action = QAction("Export Calculated Results (CSV)", self)
        export_calc_action.triggered.connect(self.export_results_csv)
        file_menu.addAction(export_calc_action)

        file_menu.addSeparator()

        pdf_report_action = QAction("Generate Executive PDF Report", self)
        pdf_report_action.triggered.connect(self.generate_pdf)
        file_menu.addAction(pdf_report_action)

    def run_inversion(self, field_type):
        target_data = self.obs_gz if field_type == 'gravity' else self.obs_dT
        if target_data is None:
            QMessageBox.warning(self, "Missing Observed Data",
                                f"Please import observed {field_type} CSV data first before running inversion.")
            return

        if not self.bodies:
            return

        active_body = self.bodies[self.active_body_idx]
        field_params = {'F': self.spin_F.value(), 'Inc': self.spin_Inc.value()}

        success = invert_body_parameters(active_body, self.x_obs, self.z_obs, target_data, field_type, field_params)
        if success:
            self.load_active_body_props()
            self.recalculate_and_plot()
            QMessageBox.information(self, "Inversion Successful",
                                    f"Successfully optimized {active_body.name} parameters to fit observed {field_type} data!")
        else:
            QMessageBox.critical(self, "Inversion Failed", "Optimizer could not converge on a valid model solution.")

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
        _, _, dT_calc = engine.calculate_magnetics(self.x_obs, self.z_obs, self.spin_F.value(), self.spin_Inc.value(),
                                                   0.0, 0.0)
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
            generate_pdf_report(file_path, temp_fig, self.bodies,
                                {'F': self.spin_F.value(), 'Inc': self.spin_Inc.value()})

            # 4. Clean up the temporary image
            if os.path.exists(temp_fig):
                os.remove(temp_fig)

            QMessageBox.information(self, "Success", f"PDF saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not build PDF:\n{str(e)}")

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
        for spin, val in [(self.spin_rho, body.density), (self.spin_k, body.k_succ), (self.spin_Jr, body.J_r),
                          (self.spin_Ir, body.I_r), (self.spin_ymin, body.y_min), (self.spin_ymax, body.y_max)]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
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
        _, _, dT_total = engine.calculate_magnetics(self.x_obs, self.z_obs, self.spin_F.value(), self.spin_Inc.value(),
                                                    0.0, 0.0)
        self.plot_results(gz_total, dT_total)

    def plot_results(self, gz, dT):
        self.ax_grav.clear()
        self.ax_mag.clear()
        self.ax_body.clear()

        if self.obs_gz is not None:
            self.ax_grav.plot(self.x_obs, self.obs_gz, 'r.', label='Observed (g)')
        self.ax_grav.plot(self.x_obs, gz, 'b-', lw=1.8, label='2.5D Calculated (g)')
        self.ax_grav.set_ylabel("Gravity (mGal)")
        self.ax_grav.grid(True)
        self.ax_grav.legend(loc='upper right')

        if self.obs_dT is not None:
            self.ax_mag.plot(self.x_obs, self.obs_dT, 'r.', label='Observed (T)')
        self.ax_mag.plot(self.x_obs, dT, 'm-', lw=1.8, label='2.5D Calculated (T)')
        self.ax_mag.set_ylabel("Magnetics (nT)")
        self.ax_mag.grid(True)
        self.ax_mag.legend(loc='upper right')

        if np.any(self.z_obs != 0):
            self.ax_body.plot(self.x_obs, self.z_obs, 'k--', lw=1.5, label='Topography')

        for idx, body in enumerate(self.bodies):
            is_active = (idx == self.active_body_idx)
            self.ax_body.plot(body.x, body.z, '-o', color=body.color, lw=2.5 if is_active else 1.0,
                              label=f"{body.name} (Y:[{body.y_min:.0f},{body.y_max:.0f}])", markersize=6)
            self.ax_body.fill(body.x, body.z, color=body.color, alpha=0.7 if is_active else 0.3)

        self.ax_body.invert_yaxis()
        self.ax_body.set_xlabel("Distance (m)")
        self.ax_body.set_ylabel("Depth (m)")
        self.ax_body.grid(True)
        self.ax_body.legend(loc='lower right', fontsize='small')
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())