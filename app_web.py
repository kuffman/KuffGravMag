import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import griddata

# Core computational backend engine imports
from engine.forward import Body2D, PotentialField2D
from engine.inversion import invert_body_parameters
from engine.grid import extract_profile_from_xyz
from engine.report import generate_pdf_report

COLOR_OPTIONS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

st.set_page_config(page_title="KuffGravMag Workbench", layout="wide")

# --- Session State Initialization ---
if 'bodies' not in st.session_state:
    b1_x = np.array([-400, -100, 0, -300, -400], dtype=float)
    b1_z = np.array([200, 200, 500, 450, 200], dtype=float)
    body1 = Body2D("Dike / Intrusion 1", b1_x, b1_z, density=400.0, k_succ=0.03, y_min=-400, y_max=400,
                   color='tab:blue')

    b2_x = np.array([100, 400, 450, 150, 100], dtype=float)
    b2_z = np.array([300, 300, 650, 600, 300], dtype=float)
    body2 = Body2D("Basement High 2", b2_x, b2_z, density=-200.0, k_succ=0.01, J_r=2.0, I_r=40.0, y_min=-1000,
                   y_max=1000, color='tab:orange')

    st.session_state.bodies = [body1, body2]

if 'active_idx' not in st.session_state:
    st.session_state.active_idx = 0

if 'x_obs' not in st.session_state:
    st.session_state.x_obs = np.linspace(-1000, 1000, 250)
    st.session_state.z_obs = np.zeros_like(st.session_state.x_obs)
    st.session_state.obs_gz = None
    st.session_state.obs_dT = None

st.title("🌐 KuffGravMag — 2.5D Mobile Workbench")

# --- Sidebar Controls ---
st.sidebar.header("🌍 Regional Ambient Field")
spin_F = st.sidebar.number_input("Earth Field (F in nT)", min_value=10000.0, max_value=70000.0, value=48000.0,
                                 step=500.0)
spin_Inc = st.sidebar.number_input("Field Inclination (deg)", min_value=-90.0, max_value=90.0, value=55.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("🧱 Subsurface Bodies (2.5D)")

# Body Selection
body_names = [b.name for b in st.session_state.bodies]
selected_idx = st.sidebar.selectbox("Active Body Target", range(len(body_names)), format_func=lambda i: body_names[i],
                                    index=min(st.session_state.active_idx, len(body_names) - 1))
st.session_state.active_idx = selected_idx
active_body = st.session_state.bodies[st.session_state.active_idx]

# Add / Remove Buttons
col_add, col_rem = st.sidebar.columns(2)
if col_add.button("➕ Add Body"):
    n = len(st.session_state.bodies) + 1
    x_new = np.array([-100, 100, 150, -50, -100], dtype=float) + (n * 50)
    z_new = np.array([400, 400, 700, 650, 400], dtype=float)
    new_body = Body2D(f"Body {n}", x_new, z_new, density=250.0, k_succ=0.015, y_min=-500, y_max=500,
                      color=COLOR_OPTIONS[n % len(COLOR_OPTIONS)])
    st.session_state.bodies.append(new_body)
    st.session_state.active_idx = len(st.session_state.bodies) - 1
    st.rerun()

if col_rem.button("➖ Remove Body") and len(st.session_state.bodies) > 1:
    st.session_state.bodies.pop(st.session_state.active_idx)
    st.session_state.active_idx = max(0, st.session_state.active_idx - 1)
    st.rerun()

# Active Body Properties
with st.sidebar.expander(f"⚙️ Properties: {active_body.name}", expanded=True):
    active_body.density = st.number_input("Density Contrast (kg/m³)", value=float(active_body.density), step=10.0)
    active_body.k_succ = st.number_input("Susceptibility (SI)", value=float(active_body.k_succ), format="%.4f",
                                         step=0.005)
    active_body.J_r = st.number_input("Remanence Jr (A/m)", value=float(active_body.J_r), step=0.5)
    active_body.I_r = st.number_input("Remanent Inc Ir (deg)", value=float(active_body.I_r), step=5.0)
    active_body.y_min = st.number_input("Strike Y-Min (m)", value=float(active_body.y_min), step=100.0)
    active_body.y_max = st.number_input("Strike Y-Max (m)", value=float(active_body.y_max), step=100.0)

    c_idx = COLOR_OPTIONS.index(active_body.color) if active_body.color in COLOR_OPTIONS else 0
    active_body.color = st.selectbox("Display Color", COLOR_OPTIONS, index=c_idx)

# Active Body Vertex Editor
with st.sidebar.expander("📐 Edit Active Body Vertices", expanded=False):
    n_verts = len(active_body.x) - 1
    new_x, new_z = [], []
    for i in range(n_verts):
        col_vx, col_vz = st.columns(2)
        # Using active_idx in key prevents DuplicateWidgetID crashes if bodies have the same name
        vx = col_vx.number_input(f"V{i + 1} X (m)", value=float(active_body.x[i]),
                                 key=f"v_{st.session_state.active_idx}_{i}_x")
        vz = col_vz.number_input(f"V{i + 1} Z (m)", value=float(active_body.z[i]),
                                 key=f"v_{st.session_state.active_idx}_{i}_z")
        new_x.append(vx)
        new_z.append(vz)

    new_x.append(new_x[0])
    new_z.append(new_z[0])
    active_body.update_polygon(new_x, new_z)

# --- Data Import & Slicing ---
st.sidebar.markdown("---")
st.sidebar.header("📂 Data Import & Slicing")
data_import_type = st.sidebar.radio("Import Type",
                                    ["None / Synthetic", "2D Observed Line (CSV)", "3D Grid Map Transect (XYZ)"])
uploaded_file = st.sidebar.file_uploader("Upload Data File", type=["csv", "txt", "dat"])

grid_spatial_data = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=r'\s+|,', engine='python').dropna()
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if data_import_type == "2D Observed Line (CSV)":
            x_vals = df.iloc[:, 0].values
            gz_vals = df.iloc[:, 1].values if df.shape[1] > 1 else None
            dT_vals = df.iloc[:, 2].values if df.shape[1] > 2 else None
            z_vals = df.iloc[:, 3].values if df.shape[1] > 3 else np.zeros_like(x_vals)

            st.session_state.x_obs = x_vals
            st.session_state.z_obs = z_vals
            st.session_state.obs_gz = gz_vals
            st.session_state.obs_dT = dT_vals
            st.sidebar.success(f"Loaded 2D Line: {len(x_vals)} points")

        elif data_import_type == "3D Grid Map Transect (XYZ)":
            if df.shape[1] >= 3:
                grid_spatial_data = (df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values)
                st.sidebar.success("3D Grid map parsed successfully!")
            else:
                st.sidebar.error("3D Grid requires at least 3 columns (X, Y, Value).")

    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# --- Inversion Panel ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Optimization & Inversion")

if st.sidebar.button("🚀 Run Gravity Inversion"):
    if st.session_state.obs_gz is None:
        st.sidebar.error("Import observed gravity data first!")
    else:
        field_params = {'F': spin_F, 'Inc': spin_Inc}
        with st.spinner(f"Inverting gravity for {active_body.name}..."):
            # Try to pass all_bodies if backend supports it, else fallback to standard args
            try:
                success = invert_body_parameters(active_body, st.session_state.x_obs, st.session_state.z_obs,
                                                 st.session_state.obs_gz, all_bodies=st.session_state.bodies,
                                                 field_type='gravity', field_params=field_params)
            except TypeError:
                success = invert_body_parameters(active_body, st.session_state.x_obs, st.session_state.z_obs,
                                                 st.session_state.obs_gz, field_type='gravity',
                                                 field_params=field_params)

            if success:
                st.sidebar.success("Gravity inversion converged!")
            else:
                st.sidebar.error("Inversion failed to converge.")

if st.sidebar.button("🚀 Run Magnetic Inversion"):
    if st.session_state.obs_dT is None:
        st.sidebar.error("Import observed magnetic data first!")
    else:
        field_params = {'F': spin_F, 'Inc': spin_Inc}
        with st.spinner(f"Inverting magnetics for {active_body.name}..."):
            try:
                success = invert_body_parameters(active_body, st.session_state.x_obs, st.session_state.z_obs,
                                                 st.session_state.obs_dT, all_bodies=st.session_state.bodies,
                                                 field_type='magnetics', field_params=field_params)
            except TypeError:
                success = invert_body_parameters(active_body, st.session_state.x_obs, st.session_state.z_obs,
                                                 st.session_state.obs_dT, field_type='magnetics',
                                                 field_params=field_params)

            if success:
                st.sidebar.success("Magnetic inversion converged!")
            else:
                st.sidebar.error("Inversion failed to converge.")

# --- Main Layout Execution (Tabs Fix) ---
col_left, col_right = st.columns([1, 1.2])

with col_left:
    tab_model, tab_map = st.tabs(["Subsurface 2.5D Model", "3D Map Transect Slicer"])

    with tab_map:
        if data_import_type == "3D Grid Map Transect (XYZ)" and grid_spatial_data is not None:
            gx, gy, gv = grid_spatial_data
            st.markdown("**1. Define A → B Transect**")
            col_a, col_b = st.columns(2)
            x1 = col_a.number_input("Point A (X1)", value=float(gx.min()))
            y1 = col_a.number_input("Point A (Y1)", value=float(gy.min()))
            x2 = col_b.number_input("Point B (X2)", value=float(gx.max()))
            y2 = col_b.number_input("Point B (Y2)", value=float(gy.max()))

            field_target = st.radio("Extract profile as:", ["Gravity Data", "Magnetic Data"], horizontal=True)

            if st.button("✂️ Extract Profile Transect"):
                profile_dist, profile_v = extract_profile_from_xyz(gx, gy, gv, x1, y1, x2, y2)
                st.session_state.x_obs = profile_dist
                st.session_state.z_obs = np.zeros_like(profile_dist)
                if field_target == "Gravity Data":
                    st.session_state.obs_gz = profile_v
                    st.session_state.obs_dT = None
                else:
                    st.session_state.obs_dT = profile_v
                    st.session_state.obs_gz = None
                st.success("Transect extracted! Switch to 'Subsurface 2.5D Model' tab to view.")

            fig_map, ax_map = plt.subplots(figsize=(6, 5))
            xi = np.linspace(gx.min(), gx.max(), 150)
            yi = np.linspace(gy.min(), gy.max(), 150)
            XI, YI = np.meshgrid(xi, yi)
            VI = griddata((gx, gy), gv, (XI, YI), method='linear')

            mesh = ax_map.pcolormesh(XI, YI, VI, shading='auto', cmap='jet')
            fig_map.colorbar(mesh, ax=ax_map, label="Amplitude")
            ax_map.plot([x1, x2], [y1, y2], 'k-', lw=2.5, marker='o', markersize=8)
            ax_map.text(x1, y1, " A", fontweight='bold', color='black')
            ax_map.text(x2, y2, " B", fontweight='bold', color='black')
            ax_map.set_xlabel("X (m)")
            ax_map.set_ylabel("Y (m)")
            ax_map.set_aspect('equal', 'box')
            st.pyplot(fig_map)
            plt.close(fig_map)
        else:
            st.info("Upload a 3D XYZ Grid file in the sidebar to use the map slicer.")

    with tab_model:
        fig_mod, ax_mod = plt.subplots(figsize=(6, 5))
        if np.any(st.session_state.z_obs != 0):
            ax_mod.plot(st.session_state.x_obs, st.session_state.z_obs, 'k--', lw=1.5, label='Topography')

        for idx, body in enumerate(st.session_state.bodies):
            is_active = (idx == st.session_state.active_idx)
            ax_mod.plot(body.x, body.z, '-o', color=body.color, lw=2.5 if is_active else 1.0,
                        label=f"{body.name} (Y:[{body.y_min:.0f},{body.y_max:.0f}])", markersize=5)
            ax_mod.fill(body.x, body.z, color=body.color, alpha=0.7 if is_active else 0.3)

        ax_mod.invert_yaxis()
        ax_mod.set_xlabel("Distance (m)")
        ax_mod.set_ylabel("Depth (m)")
        ax_mod.grid(True, linestyle='--', alpha=0.6)
        ax_mod.legend(loc='lower right', fontsize='small')
        st.pyplot(fig_mod)
        plt.close(fig_mod)

# Forward Field Calculations
engine = PotentialField2D(st.session_state.bodies)
gz_calc = engine.calculate_gravity(st.session_state.x_obs, st.session_state.z_obs)
_, _, dT_calc = engine.calculate_magnetics(st.session_state.x_obs, st.session_state.z_obs, spin_F, spin_Inc, 0.0, 0.0)

# Right Column: Field Plots
with col_right:
    st.subheader("Model Field Response & Fit")
    fig_res, (ax_g, ax_m) = plt.subplots(2, 1, figsize=(6, 5.7), sharex=True)

    # Gravity Plot
    if st.session_state.obs_gz is not None:
        ax_g.plot(st.session_state.x_obs, st.session_state.obs_gz, 'r.', label='Observed (g)')
    ax_g.plot(st.session_state.x_obs, gz_calc, 'b-', lw=1.8, label='2.5D Calculated (g)')
    ax_g.set_ylabel("Gravity (mGal)")
    ax_g.grid(True, linestyle='--', alpha=0.6)
    ax_g.legend(loc='upper right')

    # Magnetics Plot
    if st.session_state.obs_dT is not None:
        ax_m.plot(st.session_state.x_obs, st.session_state.obs_dT, 'r.', label='Observed (T)')
    ax_m.plot(st.session_state.x_obs, dT_calc, 'm-', lw=1.8, label='2.5D Calculated (T)')
    ax_m.set_xlabel("Distance (m)")
    ax_m.set_ylabel("Magnetics (nT)")
    ax_m.grid(True, linestyle='--', alpha=0.6)
    ax_m.legend(loc='upper right')

    st.pyplot(fig_res)
    plt.close(fig_res)

# --- Reporting and Export Actions ---
st.markdown("---")
col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])

# Export CSV
csv_df = pd.DataFrame({
    'Distance_m': st.session_state.x_obs,
    'Calculated_gz_mGal': gz_calc,
    'Calculated_dT_nT': dT_calc
})
csv_data = csv_df.to_csv(index=False).encode('utf-8')
col_exp1.download_button("💾 Export Calculated Results (CSV)", data=csv_data, file_name="KuffGravMag_Results.csv",
                         mime="text/csv")

# Fixed PDF Generation Workflow
if col_exp2.button("📄 Prepare Executive PDF Report"):
    with st.spinner("Generating PDF Report..."):
        try:
            temp_fig_path = os.path.join(tempfile.gettempdir(), "temp_report_fig.png")
            temp_pdf_path = os.path.join(tempfile.gettempdir(), "KuffGravMag_Report.pdf")

            # Create combined figure for report
            report_fig, (r_ax1, r_ax2, r_ax3) = plt.subplots(3, 1, figsize=(8, 7))
            if st.session_state.obs_gz is not None:
                r_ax1.plot(st.session_state.x_obs, st.session_state.obs_gz, 'r.', label='Observed')
            r_ax1.plot(st.session_state.x_obs, gz_calc, 'b-', label='Calculated')
            r_ax1.set_ylabel("Gravity (mGal)")
            r_ax1.grid(True)

            if st.session_state.obs_dT is not None:
                r_ax2.plot(st.session_state.x_obs, st.session_state.obs_dT, 'r.', label='Observed')
            r_ax2.plot(st.session_state.x_obs, dT_calc, 'm-', label='Calculated')
            r_ax2.set_ylabel("Magnetics (nT)")
            r_ax2.grid(True)

            for body in st.session_state.bodies:
                r_ax3.plot(body.x, body.z, '-o', color=body.color)
                r_ax3.fill(body.x, body.z, color=body.color, alpha=0.5)
            r_ax3.invert_yaxis()
            r_ax3.set_ylabel("Depth (m)")
            r_ax3.grid(True)

            report_fig.tight_layout()
            report_fig.savefig(temp_fig_path, dpi=300)
            plt.close(report_fig)

            # Generate PDF via engine
            generate_pdf_report(temp_pdf_path, temp_fig_path, st.session_state.bodies, {'F': spin_F, 'Inc': spin_Inc})

            # Save path to session state to keep download button alive
            st.session_state.pdf_ready_path = temp_pdf_path
            st.success("Report prepared successfully!")
        except Exception as e:
            st.error(f"Could not build PDF Report: {e}")

# Reveal the download button securely
if 'pdf_ready_path' in st.session_state and os.path.exists(st.session_state.pdf_ready_path):
    with open(st.session_state.pdf_ready_path, "rb") as f:
        col_exp3.download_button("📥 Click to Download PDF", f, file_name="KuffGravMag_Report.pdf",
                                 mime="application/pdf")