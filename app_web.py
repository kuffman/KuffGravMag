import os
import tempfile
import json
import hashlib
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
from engine.project import load_project as read_project, project_to_dict
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

for key, default in (("field_F", 48000.0), ("field_Inc", 55.0),
                     ("field_Dec", 0.0), ("field_alpha", 0.0)):
    if key not in st.session_state:
        st.session_state[key] = default

project_upload = st.sidebar.file_uploader(
    "Load KuffGravMag project (JSON)", type=["json"], key="project_upload"
)
if project_upload is not None:
    project_bytes = project_upload.getvalue()
    project_digest = hashlib.sha256(project_bytes).hexdigest()
    if st.session_state.get("loaded_project_digest") != project_digest:
        try:
            project = read_project(json.loads(project_bytes.decode("utf-8")))
            if not project["bodies"]:
                raise ValueError("Project has no bodies")
            st.session_state.bodies = project["bodies"]
            st.session_state.active_idx = 0
            field_values = project["field_params"]
            st.session_state.field_F = field_values.get("F", 48000.0)
            st.session_state.field_Inc = field_values.get("Inc", 55.0)
            st.session_state.field_Dec = field_values.get("Dec", 0.0)
            st.session_state.field_alpha = field_values.get("Alpha", 0.0)
            observations = project["observations"]
            if observations["x"] is not None and observations["z"] is not None:
                st.session_state.x_obs = observations["x"]
                st.session_state.z_obs = observations["z"]
            st.session_state.obs_gz = observations["gravity"]
            st.session_state.obs_dT = observations["magnetic"]
            st.session_state.m6_profile = project["survey_profile"]
            st.session_state.m6_processing_steps = project["processing_steps"]
            st.session_state.m6_products = []
            st.session_state.m6_regular_profiles = []
            st.session_state.loaded_project_digest = project_digest
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Could not load project: {exc}")

if 'x_obs' not in st.session_state:
    st.session_state.x_obs = np.linspace(-1000, 1000, 250)
    st.session_state.z_obs = np.zeros_like(st.session_state.x_obs)
    st.session_state.obs_gz = None
    st.session_state.obs_dT = None

for key, default in (
    ("m6_profile", None),
    ("m6_base_station", None),
    ("m6_products", []),
    ("m6_regular_profiles", []),
    ("m6_processing_steps", []),
):
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🌐 KuffGravMag — 2.5D Mobile Workbench")

# --- Sidebar Controls ---
st.sidebar.header("🌍 Regional Ambient Field")
spin_F = st.sidebar.number_input("Earth Field (F in nT)", min_value=10000.0, max_value=70000.0,
                                 step=500.0, key="field_F")
spin_Inc = st.sidebar.number_input("Field Inclination (deg)", min_value=-90.0, max_value=90.0,
                                   step=1.0, key="field_Inc")
spin_Dec = st.sidebar.number_input("Field Declination (deg)", min_value=-180.0, max_value=180.0,
                                   step=1.0, key="field_Dec")
spin_alpha = st.sidebar.number_input("Profile Azimuth (deg)", min_value=-180.0, max_value=180.0,
                                     step=1.0, key="field_alpha")

st.sidebar.markdown("---")
st.sidebar.header("🧱 Subsurface Bodies (2.5D)")

# Body Selection
body_names = [b.name for b in st.session_state.bodies]
selected_idx = st.sidebar.selectbox("Active Body Target", range(len(body_names)), format_func=lambda i: body_names[i],
                                    index=min(st.session_state.active_idx, len(body_names) - 1))
st.session_state.active_idx = selected_idx
active_body = st.session_state.bodies[st.session_state.active_idx]
project_key = st.session_state.get("loaded_project_digest", "default")
body_key = f"{project_key}_{st.session_state.active_idx}"

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
    active_body.density = st.number_input("Density Contrast (kg/m³)", value=float(active_body.density),
                                          step=10.0, key=f"density_{body_key}")
    active_body.k_succ = st.number_input("Susceptibility (SI)", value=float(active_body.k_succ),
                                         format="%.4f", step=0.005, key=f"susceptibility_{body_key}")
    active_body.J_r = st.number_input("Remanence Jr (A/m)", value=float(active_body.J_r),
                                      step=0.5, key=f"remanence_{body_key}")
    active_body.I_r = st.number_input("Remanent Inc Ir (deg)", value=float(active_body.I_r),
                                      step=5.0, key=f"remanent_inc_{body_key}")
    active_body.D_r = st.number_input("Remanent Dec Dr (deg)", value=float(active_body.D_r),
                                      step=5.0, key=f"remanent_dec_{body_key}")
    strike_mode_label = st.selectbox(
        "Strike Model", ["Infinite Strike", "Finite Strike"],
        index=0 if active_body.strike_mode == "infinite" else 1,
        key=f"strike_mode_{body_key}",
    )
    active_body.strike_mode = "infinite" if strike_mode_label == "Infinite Strike" else "finite"
    if active_body.strike_mode == "finite":
        active_body.y_min = st.number_input(
            "Strike Y-Min (m)", min_value=-10_000_000.0, max_value=10_000_000.0,
            value=float(active_body.y_min if np.isfinite(active_body.y_min) else -500.0),
            step=100.0, key=f"strike_min_{body_key}",
        )
        active_body.y_max = st.number_input(
            "Strike Y-Max (m)", min_value=-10_000_000.0, max_value=10_000_000.0,
            value=float(active_body.y_max if np.isfinite(active_body.y_max) else 500.0),
            step=100.0, key=f"strike_max_{body_key}",
        )

    c_idx = COLOR_OPTIONS.index(active_body.color) if active_body.color in COLOR_OPTIONS else 0
    active_body.color = st.selectbox("Display Color", COLOR_OPTIONS, index=c_idx,
                                     key=f"color_{body_key}")

if any(body.strike_mode == "finite" for body in st.session_state.bodies):
    st.sidebar.warning(
        "Finite Strike gravity and magnetics use closed-form prism face integrals; "
        "validated against rectangular-prism and independent volume references."
    )
else:
    st.sidebar.success(
        "Infinite Strike gravity and magnetics use benchmarked 2D kernels."
    )

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

with st.sidebar.expander("Inversion controls", expanded=False):
    inversion_sigma = st.number_input(
        "Observation standard deviation (mGal / nT)",
        min_value=1e-6,
        value=1.0,
        format="%.6g",
    )
    inversion_regularization = st.number_input(
        "Regularization lambda",
        min_value=0.0,
        value=0.0,
        format="%.6g",
    )
    fit_inversion_geometry = st.checkbox("Fit constrained depth shift and scale", value=False)


def _m6_processing_manifest(products, regular_profiles):
    steps = [product.to_manifest() for product in products]
    steps.extend({
        "name": item.name,
        "parameters": dict(item.parameters or {}),
        "assumptions": list(item.assumptions),
        "spacing_m": item.spacing_m,
        "units": item.units,
    } for item in regular_profiles)
    return steps


def _render_m6_processing_panel():
    with st.sidebar.expander("🧭 Survey processing and provenance (M6)", expanded=False):
        upload = st.file_uploader("Versioned survey profile CSV", type=["csv"], key="m6_survey_csv")
        coordinate_reference = st.text_input("Coordinate reference", "local profile coordinates", key="m6_crs")
        azimuth = st.number_input("Profile azimuth (degrees)", -180.0, 180.0, 0.0, key="m6_azimuth")
        elevation_datum = st.text_input("Elevation datum", "local survey benchmark", key="m6_elevation_datum")
        if st.button("Load versioned survey", key="m6_load_survey"):
            try:
                if upload is None:
                    raise ValueError("Choose a profile CSV first.")
                st.session_state.m6_profile = SurveyProfile.from_csv(
                    upload.getvalue(),
                    metadata={"coordinate_reference": coordinate_reference,
                              "profile_azimuth_deg": azimuth, "elevation_datum": elevation_datum},
                    source_name=upload.name,
                )
                st.session_state.m6_products = []
                st.session_state.m6_regular_profiles = []
                st.session_state.m6_processing_steps = []
            except Exception as exc:
                st.error(f"Could not load survey profile: {exc}")

        profile = st.session_state.m6_profile
        if profile is None:
            st.caption("CSV columns: distance_m, elevation_m, gravity_mgal and/or magnetic_anomaly_nt.")
            return
        st.caption(
            f"{profile.distance_m.size} stations · {profile.provenance['source_name']} · "
            f"SHA-256 {profile.provenance['source_sha256'][:16]}…"
        )
        choices = {}
        if profile.gravity_mgal is not None:
            choices["Gravity (mGal)"] = ("gravity", profile.gravity_mgal, "mGal")
        if profile.magnetic_anomaly_nt is not None:
            choices["Magnetic anomaly (nT)"] = (
                "magnetic", profile.magnetic_anomaly_nt, "nT total-field anomaly"
            )
        label = st.selectbox("Processing channel", list(choices), key="m6_channel_selection")
        channel_name, channel_values, units = choices[label]
        datum = st.number_input("Gravity reduction datum (m)", value=0.0, key="m6_datum")
        include_bouguer = st.checkbox("Apply infinite Bouguer slab", key="m6_include_bouguer")
        density = st.number_input("Slab density (kg/m³)", min_value=1.0, value=2670.0, key="m6_density")
        if st.button("Apply gravity reductions", key="m6_apply_gravity"):
            try:
                st.session_state.m6_products.append(apply_gravity_reductions(
                    profile, datum_elevation_m=datum,
                    bouguer_density_kg_m3=density if include_bouguer else None,
                ))
            except Exception as exc:
                st.error(f"Gravity correction failed: {exc}")

        base_upload = st.file_uploader("Magnetic base-station CSV", type=["csv"], key="m6_base_csv")
        if base_upload is not None and st.button("Load base station", key="m6_load_base"):
            try:
                st.session_state.m6_base_station = BaseStationSeries.from_csv(
                    base_upload.getvalue(), source_name=base_upload.name
                )
            except Exception as exc:
                st.error(f"Could not load base station: {exc}")
        reference_time = st.text_input(
            "Diurnal reference time (UTC ISO-8601)", key="m6_reference_time",
            placeholder="2026-01-01T12:00:00Z",
        )
        if st.button("Apply magnetic diurnal correction", key="m6_apply_diurnal"):
            try:
                if st.session_state.m6_base_station is None:
                    raise ValueError("Load a base-station CSV first.")
                st.session_state.m6_products.append(apply_magnetic_diurnal_correction(
                    profile, st.session_state.m6_base_station, reference_time_utc=reference_time
                ))
            except Exception as exc:
                st.error(f"Diurnal correction failed: {exc}")

        degree = st.number_input("Polynomial regional degree", 0, 5, 1, key="m6_degree")
        if st.button("Separate regional and residual", key="m6_regional"):
            try:
                st.session_state.m6_products.append(separate_polynomial_regional(
                    profile.distance_m, channel_values, degree=int(degree), units=units
                ))
            except Exception as exc:
                st.error(f"Regional separation failed: {exc}")

        spacing = st.number_input("Regular sampling interval (m)", min_value=0.001, value=100.0, key="m6_spacing")
        if st.button("Interpolate selected channel", key="m6_interpolate"):
            try:
                regular = interpolate_profile(profile.distance_m, channel_values, spacing_m=spacing, units=units)
                st.session_state.m6_regular_profiles.append(RegularProfile(
                    regular.distance_m, regular.values, regular.units,
                    name=f"{channel_name}_linear_interpolation",
                    parameters=regular.parameters, assumptions=regular.assumptions,
                ))
            except Exception as exc:
                st.error(f"Interpolation failed: {exc}")

        if st.session_state.m6_regular_profiles:
            height = st.number_input("Upward continuation height (m)", min_value=0.0, value=0.0, key="m6_height")
            padding = st.selectbox(
                "Spectral edge treatment", ["reflect", "none"],
                format_func=lambda value: "Reflect padding (review edges)" if value == "reflect" else "Periodic ends",
                key="m6_padding",
            )
            if st.button("Calculate profile spectral transforms", key="m6_spectral"):
                try:
                    regular = st.session_state.m6_regular_profiles[-1]
                    if not regular.name.startswith(channel_name):
                        raise ValueError("Interpolate the selected channel before transforming it.")
                    result = transform_profile(
                        regular.values, spacing_m=regular.spacing_m, height_m=height,
                        padding=padding, units=regular.units,
                    )
                    parameters = {"source_regular_profile": regular.name, "height_m": height,
                                  "padding": padding, "spacing_m": regular.spacing_m,
                                  "edge_effect_samples": result.edge_effect_samples}
                    for output_name in (
                        "upward_continued", "horizontal_derivative", "vertical_derivative_down",
                        "analytic_signal", "tilt_angle_deg",
                    ):
                        st.session_state.m6_regular_profiles.append(RegularProfile(
                            regular.distance_m, getattr(result, output_name), result.units[output_name],
                            name=f"{regular.name}_{output_name}", parameters=parameters,
                            assumptions=result.assumptions,
                        ))
                    st.info(f"Review the first and last {result.edge_effect_samples} samples for padding effects.")
                except Exception as exc:
                    st.error(f"Spectral transform failed: {exc}")

        st.session_state.m6_processing_steps = _m6_processing_manifest(
            st.session_state.m6_products, st.session_state.m6_regular_profiles
        )
        if st.session_state.m6_processing_steps:
            st.write("Processing history", st.session_state.m6_processing_steps)
        chart_data = {"Raw channel": channel_values}
        for product in st.session_state.m6_products:
            for name, values in product.columns.items():
                if product.units[name] == channel_units:
                    chart_data[name] = values
        st.line_chart(pd.DataFrame(chart_data, index=profile.distance_m), height=240)
        matching_regular = [
            item for item in st.session_state.m6_regular_profiles
            if item.name.startswith(channel_name)
        ]
        if matching_regular:
            regular = matching_regular[-1]
            st.line_chart(pd.DataFrame({regular.name: regular.values}, index=regular.distance_m), height=200)
        bundle = export_processing_bundle(
            profile, st.session_state.m6_products,
            regular_profiles=st.session_state.m6_regular_profiles,
        )
        st.download_button(
            "Download raw data, outputs, and provenance (.zip)", data=bundle,
            file_name="KuffGravMag_Processing.zip", mime="application/zip", key="m6_bundle",
        )
        st.caption("Terrain corrections and IGRF/main-field subtraction are not included.")


def _show_inversion_result(label, result):
    if not result.success:
        st.sidebar.error(f"{label} inversion stopped ({result.status}): {result.message}")
        st.sidebar.caption("The starting model was preserved.")
        return

    if result.identifiable:
        st.sidebar.success(f"{label} inversion converged.")
    else:
        st.sidebar.warning(f"{label} fit converged, but the data do not identify a unique model.")
    st.sidebar.caption(
        f"RMS {result.initial_rms:.5g} → {result.final_rms:.5g}; "
        f"weighted RMS {result.weighted_rms:.5g}; "
        f"{result.function_evaluations} evaluations; "
        f"rank {result.numerical_rank}/{result.parameter_count}."
    )
    if result.standard_errors:
        uncertainty = ", ".join(
            f"{name} ± {value:.4g}" for name, value in result.standard_errors.items()
        )
        st.sidebar.caption(f"Approximate local 1σ errors: {uncertainty}")
    elif result.message:
        st.sidebar.caption(result.message)

_render_m6_processing_panel()

if st.sidebar.button("🚀 Run Gravity Inversion"):
    if st.session_state.obs_gz is None:
        st.sidebar.error("Import observed gravity data first!")
    else:
        field_params = {"F": spin_F, "Inc": spin_Inc, "Dec": spin_Dec, "alpha": spin_alpha}
        with st.spinner(f"Inverting gravity for {active_body.name}..."):
            result = invert_body_parameters(
                active_body,
                st.session_state.x_obs,
                st.session_state.z_obs,
                st.session_state.obs_gz,
                all_bodies=st.session_state.bodies,
                field_type="gravity",
                field_params=field_params,
                observation_sigma=inversion_sigma,
                regularization=inversion_regularization,
                fit_geometry=fit_inversion_geometry,
            )
        _show_inversion_result("Gravity", result)

if st.sidebar.button("🚀 Run Magnetic Inversion"):
    if st.session_state.obs_dT is None:
        st.sidebar.error("Import observed magnetic data first!")
    else:
        field_params = {"F": spin_F, "Inc": spin_Inc, "Dec": spin_Dec, "alpha": spin_alpha}
        with st.spinner(f"Inverting magnetics for {active_body.name}..."):
            result = invert_body_parameters(
                active_body,
                st.session_state.x_obs,
                st.session_state.z_obs,
                st.session_state.obs_dT,
                all_bodies=st.session_state.bodies,
                field_type="magnetics",
                field_params=field_params,
                observation_sigma=inversion_sigma,
                regularization=inversion_regularization,
                fit_geometry=fit_inversion_geometry,
            )
        _show_inversion_result("Magnetic", result)

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
_, _, _, dT_calc = engine.calculate_magnetics_components(
    st.session_state.x_obs, st.session_state.z_obs, spin_F, spin_Inc, spin_Dec, spin_alpha
)
finite_strike_gravity = any(body.strike_mode == "finite" for body in st.session_state.bodies)
infinite_strike_gravity = any(body.strike_mode == "infinite" for body in st.session_state.bodies)
gravity_plot_label = "Finite Strike gravity (validated prism model)" if finite_strike_gravity else (
    "Infinite Strike gravity (benchmarked 2D model)"
)

# Right Column: Field Plots
with col_right:
    st.subheader("Model Field Response & Fit")
    fig_res, (ax_g, ax_m) = plt.subplots(2, 1, figsize=(6, 5.7), sharex=True)

    # Gravity Plot
    if finite_strike_gravity:
        st.info(
            "Finite Strike gravity and magnetics use closed-form polygonal-prism face integrals, "
            "validated against rectangular-prism and independent volume references. "
            "Magnetics assume uniform magnetization and no self-demagnetization."
        )
    elif infinite_strike_gravity:
        st.success("Infinite Strike gravity and magnetics use benchmarked 2D kernels.")
    if st.session_state.obs_gz is not None:
        ax_g.plot(st.session_state.x_obs, st.session_state.obs_gz, 'r.', label='Observed (g)')
    ax_g.plot(st.session_state.x_obs, gz_calc, 'b-', lw=1.8, label=gravity_plot_label)
    ax_g.set_ylabel("Gravity (mGal)")
    ax_g.grid(True, linestyle='--', alpha=0.6)
    ax_g.legend(loc='upper right')

    # Magnetics Plot
    if st.session_state.obs_dT is not None:
        ax_m.plot(st.session_state.x_obs, st.session_state.obs_dT, 'r.', label='Observed (T)')
    magnetic_plot_label = (
        "Finite Strike calculated dT"
        if finite_strike_gravity else "Infinite Strike calculated dT"
    )
    ax_m.plot(st.session_state.x_obs, dT_calc, 'm-', lw=1.8, label=magnetic_plot_label)
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

project_observations = {
    "x": st.session_state.x_obs,
    "z": st.session_state.z_obs,
    "gravity": st.session_state.obs_gz,
    "magnetic": st.session_state.obs_dT,
}
project_fields = {"F": spin_F, "Inc": spin_Inc, "Dec": spin_Dec, "Alpha": spin_alpha}
project_json = json.dumps(project_to_dict(
    st.session_state.bodies,
    project_fields,
    project_observations,
    survey_profile=st.session_state.m6_profile,
    processing_steps=st.session_state.m6_processing_steps,
), indent=2, allow_nan=False)
st.sidebar.download_button(
    "Save KuffGravMag Project (JSON)", data=project_json,
    file_name="KuffGravMag_Project.json", mime="application/json",
)

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
            generate_pdf_report(temp_pdf_path, temp_fig_path, st.session_state.bodies, project_fields)

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
