import io
import os
import tempfile
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import streamlit as st
from scipy.interpolate import griddata

# Core computational backend engine imports
from engine.forward import Body2D, PotentialField2D
from engine.inversion import invert_body_parameters
from engine.grid import extract_profile_from_xyz
from engine.report import generate_pdf_report
from engine.project import load_project as read_project, project_to_dict
from engine.project_bundle import (
    MAX_ATTACHMENT_BYTES,
    build_project_bundle,
    load_project_bundle,
)
from engine.spatial import load_xyz_layer, spatial_layer_csv
from engine.survey import SurveyProfile, BaseStationSeries, export_processing_bundle
from engine.processing import (
    apply_gravity_reductions,
    apply_magnetic_diurnal_correction,
    separate_polynomial_regional,
    interpolate_profile,
    RegularProfile,
)
from engine.spectral import (
    estimate_spectral_depth,
    stabilized_magnetic_transform_grid,
    transform_grid,
    transform_profile,
)
from engine.euler import euler_window_sweep
from engine.processing import regular_grid_from_xyz
from engine.visualization import finite_prism_faces

COLOR_OPTIONS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

st.set_page_config(page_title="KuffGravMag Workbench", layout="wide")


def _restore_spatial_layer(processing_steps, attachments):
    for step in reversed(processing_steps or []):
        if not isinstance(step, dict) or step.get("name") != "local_spatial_layer":
            continue
        parameters = step.get("parameters", {})
        if not isinstance(parameters, dict):
            continue
        raw = attachments.get(parameters.get("attachment_name"))
        if raw is None:
            continue
        return load_xyz_layer(
            raw,
            source_crs=parameters.get("source_crs", ""),
            target_crs=parameters.get("target_crs"),
            units=parameters.get("units", "input units"),
            name=parameters.get("source_name", "Local XYZ layer"),
        )
    return None

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

for key, default in (
    ("project_attachments", {}),
    ("spatial_layer", None),
    ("project_grid_bytes", None),
    ("project_grid_name", None),
    ("m71_source_steps", []),
):
    if key not in st.session_state:
        st.session_state[key] = default

project_upload = st.sidebar.file_uploader(
    "Load KuffGravMag project (JSON or portable bundle)",
    type=["json", "kgmproj"], key="project_upload"
)
if project_upload is not None:
    project_bytes = project_upload.getvalue()
    project_digest = hashlib.sha256(project_bytes).hexdigest()
    if st.session_state.get("loaded_project_digest") != project_digest:
        try:
            if project_upload.name.lower().endswith(".kgmproj"):
                bundle = load_project_bundle(project_bytes)
                project = bundle["project"]
                attachments = bundle["attachments"]
            else:
                project = read_project(json.loads(project_bytes.decode("utf-8")))
                attachments = {}
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
            st.session_state.m71_source_steps = list(project["processing_steps"])
            st.session_state.m6_products = []
            st.session_state.m6_regular_profiles = []
            st.session_state.project_attachments = attachments
            st.session_state.project_grid_bytes = None
            st.session_state.project_grid_name = None
            for step in reversed(project["processing_steps"]):
                if not isinstance(step, dict) or step.get("name") != "regular_grid_interpretation_source":
                    continue
                parameters = step.get("parameters", {})
                if not isinstance(parameters, dict):
                    continue
                attachment_name = parameters.get("attachment_name")
                if attachment_name in attachments:
                    st.session_state.project_grid_bytes = attachments[attachment_name]
                    st.session_state.project_grid_name = parameters.get("source_name", attachment_name)
                    st.session_state.data_import_type = "3D Grid Map Transect (XYZ)"
                    break
            st.session_state.spatial_layer = _restore_spatial_layer(
                project["processing_steps"], attachments
            )
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

local_layer_upload = st.sidebar.file_uploader(
    "Load local XYZ spatial layer", type=["csv", "xyz", "txt"], key="m71_local_layer_upload"
)
source_crs = st.sidebar.text_input(
    "Spatial layer source CRS", value="", placeholder="For example EPSG:32630", key="m71_source_crs"
)
target_crs = st.sidebar.text_input(
    "Spatial layer target CRS", value="", placeholder="Blank keeps source coordinates", key="m71_target_crs"
)
spatial_units = st.sidebar.text_input(
    "Spatial layer value units", value="input units", key="m71_spatial_units"
)
if st.sidebar.button("Import local layer", key="m71_import_layer"):
    try:
        if local_layer_upload is None:
            raise ValueError("Choose an XYZ/CSV file first")
        raw = local_layer_upload.getvalue()
        layer = load_xyz_layer(
            raw,
            source_crs=source_crs,
            target_crs=target_crs or None,
            units=spatial_units,
            name=local_layer_upload.name,
        )
        digest = hashlib.sha256(raw).hexdigest()
        safe_basename = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in local_layer_upload.name
        ).strip("._")[:100] or "local-data.xyz"
        attachment_name = f"{digest[:12]}-{safe_basename}"
        st.session_state.project_attachments[attachment_name] = raw
        st.session_state.spatial_layer = layer
        source_step = {
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
        }
        st.session_state.m71_source_steps.append(source_step)
    except Exception as exc:
        st.sidebar.error(f"Could not load local layer: {exc}")

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
                                    ["None / Synthetic", "2D Observed Line (CSV)", "3D Grid Map Transect (XYZ)"],
                                    key="data_import_type")
uploaded_file = st.sidebar.file_uploader("Upload Data File", type=["csv", "txt", "dat"])

grid_spatial_data = None

if uploaded_file is not None:
    try:
        uploaded_payload = uploaded_file.getvalue()
        if len(uploaded_payload) > MAX_ATTACHMENT_BYTES:
            raise ValueError("Portable source files are limited to 32 MiB")
        df = pd.read_csv(io.BytesIO(uploaded_payload), sep=r'\s+|,', engine='python').dropna()
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
            digest = hashlib.sha256(uploaded_payload).hexdigest()
            safe_basename = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in uploaded_file.name
            ).strip("._")[:100] or "observed-profile.csv"
            attachment_name = f"{digest[:12]}-{safe_basename}"
            st.session_state.project_attachments[attachment_name] = uploaded_payload
            if st.session_state.get("m71_observed_source_digest") != digest:
                st.session_state.m71_source_steps.append({
                    "name": "observed_profile_import",
                    "parameters": {
                        "source_name": uploaded_file.name,
                        "attachment_name": attachment_name,
                        "sha256": digest,
                        "columns": ["distance_m", "gravity_mgal", "magnetic_nT", "elevation_m"],
                        "local_only": True,
                    },
                })
                st.session_state.m71_observed_source_digest = digest

        elif data_import_type == "3D Grid Map Transect (XYZ)":
            if df.shape[1] >= 3:
                grid_spatial_data = (df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values)
                st.sidebar.success("3D Grid map parsed successfully!")
                digest = hashlib.sha256(uploaded_payload).hexdigest()
                safe_basename = "".join(
                    character if character.isalnum() or character in "._-" else "_"
                    for character in uploaded_file.name
                ).strip("._")[:100] or "grid.xyz"
                attachment_name = f"{digest[:12]}-{safe_basename}"
                st.session_state.project_attachments[attachment_name] = uploaded_payload
                st.session_state.project_grid_bytes = uploaded_payload
                st.session_state.project_grid_name = uploaded_file.name
                if st.session_state.get("m71_grid_source_digest") != digest:
                    st.session_state.m71_source_steps.append({
                        "name": "regular_grid_interpretation_source",
                        "parameters": {
                            "source_name": uploaded_file.name,
                            "attachment_name": attachment_name,
                            "sha256": digest,
                            "size_bytes": len(uploaded_payload),
                            "coordinate_reference": st.session_state.get("grid_crs", ""),
                            "units": st.session_state.get("grid_units", "input units"),
                            "local_only": True,
                        },
                    })
                    st.session_state.m71_grid_source_digest = digest
            else:
                st.sidebar.error("3D Grid requires at least 3 columns (X, Y, Value).")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")
elif data_import_type == "3D Grid Map Transect (XYZ)" and st.session_state.project_grid_bytes:
    try:
        df = pd.read_csv(
            io.BytesIO(st.session_state.project_grid_bytes),
            sep=r'\s+|,', engine='python',
        ).dropna()
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if df.shape[1] < 3:
            raise ValueError("Embedded grid requires X, Y, and value columns")
        grid_spatial_data = (df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values)
        st.sidebar.success(f"Restored embedded grid: {st.session_state.project_grid_name}")
    except Exception as e:
        st.sidebar.error(f"Could not read embedded grid: {e}")

st.session_state.m6_processing_steps = list(st.session_state.m71_source_steps)

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
                st.session_state.m71_source_steps = [
                    step for step in st.session_state.m71_source_steps
                    if isinstance(step, dict) and step.get("name") in {
                        "observed_profile_import",
                        "regular_grid_interpretation_source",
                        "local_spatial_layer",
                    }
                ]
                st.session_state.m6_processing_steps = list(st.session_state.m71_source_steps)
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

        st.session_state.m6_processing_steps = list(st.session_state.m71_source_steps) + _m6_processing_manifest(
            st.session_state.m6_products, st.session_state.m6_regular_profiles
        )
        if st.session_state.m6_processing_steps:
            st.write("Processing history", st.session_state.m6_processing_steps)
        chart_data = {"Raw channel": channel_values}
        for product in st.session_state.m6_products:
            for name, values in product.columns.items():
                if product.units[name] == units:
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
        st.caption(
            "DEM terrain residuals and IGRF-14 raw total-field corrections are available through the processing API. "
            "This profile panel lacks station map coordinates, and its magnetic channel is already an anomaly."
        )


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
    tab_model, tab_map, tab_grid, tab_spatial, tab_3d = st.tabs([
        "Subsurface 2.5D Model", "3D Map Transect Slicer", "Grid Interpretation",
        "Local Spatial Layer", "Finite Strike 3D Model",
    ])

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

    with tab_grid:
        if data_import_type != "3D Grid Map Transect (XYZ)" or grid_spatial_data is None:
            st.info("Choose a regular XYZ grid in the sidebar to open the grid interpretation workflow.")
        else:
            gx, gy, gv = grid_spatial_data
            grid_units_label = st.selectbox("Grid value units", ["nT", "mGal", "other input units"], key="grid_units")
            grid_units = "input units" if grid_units_label == "other input units" else grid_units_label
            grid_crs = st.text_input("Grid coordinate reference", "local projected grid (metres)", key="grid_crs")
            for step in reversed(st.session_state.m71_source_steps):
                if not isinstance(step, dict) or step.get("name") != "regular_grid_interpretation_source":
                    continue
                parameters = step.get("parameters", {})
                if not isinstance(parameters, dict):
                    continue
                if parameters.get("attachment_name") in st.session_state.project_attachments:
                    parameters["coordinate_reference"] = grid_crs
                    parameters["units"] = grid_units
                    break
            try:
                grid = regular_grid_from_xyz(
                    gx, gy, gv, units=grid_units, coordinate_reference=grid_crs
                )
                st.caption(
                    f"Complete grid: {grid.values.shape[1]} × {grid.values.shape[0]} cells · "
                    f"spacing {grid.x_m[1]-grid.x_m[0]:g} × {grid.y_m[1]-grid.y_m[0]:g} m · {grid.coordinate_reference}"
                )
                fig_grid, ax_grid = plt.subplots(figsize=(6, 4.5))
                mesh_grid = ax_grid.imshow(
                    grid.values, origin="lower", aspect="equal",
                    extent=(grid.x_m[0], grid.x_m[-1], grid.y_m[0], grid.y_m[-1]),
                    cmap="RdBu_r",
                )
                fig_grid.colorbar(mesh_grid, ax=ax_grid, label=grid.units)
                ax_grid.set_xlabel("X / easting (m)")
                ax_grid.set_ylabel("Y / northing (m)")
                ax_grid.set_title("Observed grid")
                st.pyplot(fig_grid)
                plt.close(fig_grid)

                st.markdown("#### Spectral grid transforms")
                output_name = st.selectbox(
                    "Transform output",
                    ["upward_continued", "derivative_x", "derivative_y", "vertical_derivative_down", "analytic_signal", "tilt_angle_deg"],
                    format_func=lambda value: value.replace("_", " "),
                    key="grid_transform_output",
                )
                grid_height = st.number_input("Upward continuation height (m)", min_value=0.0, value=0.0, key="grid_transform_height")
                grid_padding = st.selectbox(
                    "Spectral edge treatment", ["reflect", "none"],
                    format_func=lambda value: "Reflect padding (review marked edges)" if value == "reflect" else "Periodic grid (wrap at boundaries)",
                    key="grid_transform_padding",
                )
                if st.button("Calculate selected grid transform", key="grid_transform_run"):
                    try:
                        st.session_state.m61_grid_transform = transform_grid(
                            grid.values,
                            spacing_x_m=grid.x_m[1] - grid.x_m[0],
                            spacing_y_m=grid.y_m[1] - grid.y_m[0],
                            height_m=grid_height,
                            padding=grid_padding,
                            units=grid.units,
                        )
                        st.session_state.m61_grid_metadata = {
                            "coordinate_reference": grid.coordinate_reference,
                            "units": grid.units,
                            "spacing_x_m": grid.x_m[1] - grid.x_m[0],
                            "spacing_y_m": grid.y_m[1] - grid.y_m[0],
                        }
                    except Exception as exc:
                        st.error(f"Grid transform failed: {exc}")
                transform_result = st.session_state.get("m61_grid_transform")
                if transform_result is not None:
                    transformed = getattr(transform_result, output_name)
                    fig_result, ax_result = plt.subplots(figsize=(6, 4))
                    mesh_result = ax_result.imshow(
                        transformed, origin="lower", aspect="equal",
                        extent=(grid.x_m[0], grid.x_m[-1], grid.y_m[0], grid.y_m[-1]), cmap="RdBu_r",
                    )
                    fig_result.colorbar(mesh_result, ax=ax_result, label=transform_result.units[output_name])
                    ax_result.set_title(output_name.replace("_", " "))
                    ax_result.set_xlabel("X / easting (m)")
                    ax_result.set_ylabel("Y / northing (m)")
                    st.pyplot(fig_result)
                    plt.close(fig_result)
                    st.caption(
                        f"Reflect edge-review widths: {transform_result.edge_effect_samples}. "
                        + " ".join(transform_result.assumptions)
                    )

                with st.expander("Stabilized magnetic direction transform", expanded=False):
                    direction_a, direction_b = st.columns(2)
                    field_inc = direction_a.number_input("Field inclination (°)", -90.0, 90.0, float(spin_Inc), key="grid_field_inc")
                    field_dec = direction_b.number_input("Field declination east (°)", -360.0, 360.0, float(spin_Dec), key="grid_field_dec")
                    mag_inc = direction_a.number_input("Source magnetization inclination (°)", -90.0, 90.0, float(spin_Inc), key="grid_mag_inc")
                    mag_dec = direction_b.number_input("Source magnetization declination east (°)", -360.0, 360.0, float(spin_Dec), key="grid_mag_dec")
                    target_type = st.selectbox("Target", ["Pole (vertical)", "Equator (horizontal)"] , key="grid_target_type")
                    regularization = st.number_input("Tikhonov regularization λ", min_value=0.000001, value=0.05, format="%.6g", key="grid_rtp_lambda")
                    max_gain = st.number_input("Maximum allowed Fourier gain", min_value=0.1, value=20.0, format="%.4g", key="grid_rtp_gain")
                    if st.button("Calculate regularized direction transform", key="grid_rtp_run"):
                        try:
                            target_inc = 90.0 if target_type == "Pole (vertical)" else 0.0
                            result = stabilized_magnetic_transform_grid(
                                grid.values,
                                spacing_x_m=grid.x_m[1] - grid.x_m[0],
                                spacing_y_m=grid.y_m[1] - grid.y_m[0],
                                field_inclination_deg=field_inc,
                                field_declination_deg=field_dec,
                                magnetization_inclination_deg=mag_inc,
                                magnetization_declination_deg=mag_dec,
                                target_inclination_deg=target_inc,
                                target_declination_deg=field_dec,
                                regularization=regularization,
                                padding=grid_padding,
                                max_gain_limit=max_gain,
                            )
                            st.session_state.m61_magnetic_grid_transform = result
                            st.success(
                                f"Maximum gain {result.max_gain:.3g}; RMS gain {result.rms_gain:.3g}; "
                                f"{result.fraction_gain_above_one:.1%} of frequencies amplify above unity."
                            )
                            st.caption(" ".join(result.assumptions))
                            st.image(result.values, caption="Regularized magnetic transform", clamp=True)
                        except Exception as exc:
                            st.error(f"Magnetic direction transform failed: {exc}")

                st.markdown("#### Euler window sweep")
                euler_left, euler_right = st.columns(2)
                window_size = euler_left.number_input("Window width (odd cells)", min_value=3, value=15, step=2, key="grid_euler_window")
                stride = euler_right.number_input("Window stride (cells)", min_value=1, value=5, key="grid_euler_stride")
                indices_text = st.text_input("Structural indices (comma-separated)", "1, 2, 3", key="grid_euler_indices")
                condition_limit = st.number_input("Maximum Euler condition number", min_value=2.0, value=1e8, format="%.4g", key="grid_euler_condition")
                rms_limit = st.number_input("Maximum Euler RMS residual (0 disables)", min_value=0.0, value=0.0, format="%.6g", key="grid_euler_rms")
                if st.button("Run Euler window sweep", key="grid_euler_run"):
                    try:
                        derivatives = st.session_state.get("m61_grid_transform")
                        if derivatives is None:
                            derivatives = transform_grid(
                                grid.values,
                                spacing_x_m=grid.x_m[1] - grid.x_m[0],
                                spacing_y_m=grid.y_m[1] - grid.y_m[0],
                                padding=grid_padding,
                                units=grid.units,
                            )
                            st.session_state.m61_grid_transform = derivatives
                        indices = [float(item.strip()) for item in indices_text.split(",") if item.strip()]
                        estimates = euler_window_sweep(
                            grid.x_m, grid.y_m, grid.values,
                            derivatives.derivative_x, derivatives.derivative_y,
                            -derivatives.vertical_derivative_down,
                            structural_indices=indices,
                            window_size_samples=int(window_size), stride_samples=int(stride),
                            condition_limit=condition_limit,
                            max_rms_residual=rms_limit or None,
                            mask=~derivatives.edge_mask,
                        )
                        accepted = [item for item in estimates if item.accepted]
                        st.session_state.m61_euler_estimates = estimates
                        st.write(f"{len(accepted)} of {len(estimates)} windows passed the selected index, condition, and RMS filters.")
                        if accepted:
                            st.dataframe(pd.DataFrame([
                                {"x0_m": item.result.x0_m, "y0_m": item.result.y0_m,
                                 "depth_m": item.result.depth_below_observation_m,
                                 "structural_index": item.structural_index,
                                 "condition": item.result.condition_number,
                                 "rms": item.result.rms_residual,
                                 "window_x_m": item.center_x_m, "window_y_m": item.center_y_m}
                                for item in accepted
                            ]), use_container_width=True)
                        else:
                            st.info("No windows passed; review grid derivatives, window width, structural indices, and filters.")
                    except Exception as exc:
                        st.error(f"Euler window sweep failed: {exc}")

                st.markdown("#### Spectral depth estimate")
                depth_a, depth_b, depth_c = st.columns(3)
                depth_min = depth_a.number_input("Band minimum k (rad/m)", min_value=0.000001, value=0.005, format="%.6g", key="grid_depth_kmin")
                depth_max = depth_b.number_input("Band maximum k (rad/m)", min_value=0.000002, value=0.05, format="%.6g", key="grid_depth_kmax")
                depth_bins = depth_c.number_input("Radial bins", min_value=4, max_value=256, value=24, key="grid_depth_bins")
                if st.button("Estimate band-limited source depth", key="grid_depth_run"):
                    try:
                        depth = estimate_spectral_depth(
                            grid.values,
                            spacing_x_m=grid.x_m[1] - grid.x_m[0],
                            spacing_y_m=grid.y_m[1] - grid.y_m[0],
                            wavenumber_band_rad_per_m=(depth_min, depth_max),
                            radial_bin_count=int(depth_bins),
                        )
                        st.metric("Spectral depth", f"{depth.depth_m:.4g} m", f"± {depth.standard_error_m:.3g} m (linear-fit 1σ)")
                        st.caption(f"R²={depth.r_squared:.4f}; {depth.bin_count} bins. " + " ".join(depth.assumptions))
                    except Exception as exc:
                        st.error(f"Spectral depth estimate failed: {exc}")
            except Exception as exc:
                st.warning(f"Grid interpretation needs a complete, uniformly spaced rectangular XYZ grid: {exc}")

    with tab_spatial:
        layer = st.session_state.spatial_layer
        if layer is None:
            st.info("Load a local XYZ layer in the sidebar. CRS conversion requires explicit source and target CRS definitions.")
        else:
            st.caption(
                f"{layer.x.size} points · {layer.source_crs} → {layer.target_crs} · "
                f"{layer.units} · SHA-256 {layer.source_sha256}"
            )
            st.info("Local file layer only. No basemap or remote service is requested.")
            fig_layer, ax_layer = plt.subplots(figsize=(7, 5))
            points = ax_layer.scatter(layer.x, layer.y, c=layer.values, cmap="viridis", s=24)
            fig_layer.colorbar(points, ax=ax_layer, label=layer.units)
            ax_layer.set_xlabel(f"X ({layer.x_unit})")
            ax_layer.set_ylabel(f"Y ({layer.y_unit})")
            ax_layer.set_title(layer.name)
            ax_layer.set_aspect("equal", adjustable="datalim")
            ax_layer.grid(True, alpha=0.2)
            st.pyplot(fig_layer)
            plt.close(fig_layer)
            st.download_button(
                "Export transformed XYZ with CRS metadata",
                data=spatial_layer_csv(layer),
                file_name="KuffGravMag_Local_Spatial_Layer.csv",
                mime="text/csv",
                key="m71_export_spatial_layer",
            )

    with tab_3d:
        finite_bodies = [
            (index, body) for index, body in enumerate(st.session_state.bodies)
            if body.strike_mode == "finite"
        ]
        if not finite_bodies:
            st.info("Set a model body's strike mode to Finite Strike to view its 3D prism.")
        else:
            fig_3d = plt.figure(figsize=(7, 5.5))
            ax_3d = fig_3d.add_subplot(111, projection="3d")
            all_faces = []
            for index, body in finite_bodies:
                try:
                    faces = finite_prism_faces(body)
                except ValueError as exc:
                    st.warning(f"{body.name}: {exc}")
                    continue
                all_faces.extend(faces)
                ax_3d.add_collection3d(Poly3DCollection(
                    faces,
                    facecolor=body.color,
                    edgecolor="#26364a",
                    linewidth=1.1 if index == st.session_state.active_idx else 0.65,
                    alpha=0.78 if index == st.session_state.active_idx else 0.48,
                ))
                ax_3d.text(
                    float(np.mean(body.x[:-1])),
                    (float(body.y_min) + float(body.y_max)) / 2.0,
                    float(np.mean(body.z[:-1])),
                    body.name,
                    fontsize=8,
                )
            if all_faces:
                points = np.vstack(all_faces)
                minima = np.min(points, axis=0)
                maxima = np.max(points, axis=0)
                spans = np.maximum(maxima - minima, 1.0)
                padding = spans * 0.06
                x_limits = (minima[0] - padding[0], maxima[0] + padding[0])
                y_limits = (minima[1] - padding[1], maxima[1] + padding[1])
                z_lower = min(0.0, minima[2] - padding[2])
                z_upper = maxima[2] + padding[2]
                ax_3d.set_xlim(*x_limits)
                ax_3d.set_ylim(*y_limits)
                ax_3d.set_zlim(z_upper, z_lower)
                ax_3d.set_box_aspect((
                    x_limits[1] - x_limits[0],
                    y_limits[1] - y_limits[0],
                    z_upper - z_lower,
                ))
                ax_3d.set_xlabel("Profile distance X (m)")
                ax_3d.set_ylabel("Strike Y (m)")
                ax_3d.set_zlabel("Depth Z (m, positive down)")
                ax_3d.set_title("Finite-strike prism geometry")
                ax_3d.view_init(elev=23, azim=-58)
                ax_3d.grid(True)
            st.pyplot(fig_3d)
            plt.close(fig_3d)
            st.caption("The active body's 2D vertices and finite strike bounds are edited in the sidebar.")

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
project_data = project_to_dict(
    st.session_state.bodies,
    project_fields,
    project_observations,
    survey_profile=st.session_state.m6_profile,
    processing_steps=st.session_state.m6_processing_steps,
)
project_json = json.dumps(project_data, indent=2, allow_nan=False)
st.sidebar.download_button(
    "Save KuffGravMag Project (JSON)", data=project_json,
    file_name="KuffGravMag_Project.json", mime="application/json",
)
attachment_key = hashlib.sha256(
    b"".join(
        name.encode("utf-8") + hashlib.sha256(content).digest()
        for name, content in sorted(st.session_state.project_attachments.items())
    )
).hexdigest()
bundle_key = hashlib.sha256(
    project_json.encode("utf-8") + attachment_key.encode("ascii")
).hexdigest()
if st.session_state.get("portable_bundle_key") != bundle_key:
    try:
        st.session_state.portable_project_bundle = build_project_bundle(
            project_data, st.session_state.project_attachments
        )
        st.session_state.portable_bundle_key = bundle_key
    except Exception as exc:
        st.session_state.portable_project_bundle = None
        st.session_state.portable_bundle_error = str(exc)
if st.session_state.get("portable_project_bundle") is not None:
    st.sidebar.download_button(
        "Save portable project bundle (.kgmproj)",
        data=st.session_state.portable_project_bundle,
        file_name="KuffGravMag_Project.kgmproj",
        mime="application/vnd.kuffgravmag.project+zip",
    )
elif st.session_state.get("portable_bundle_error"):
    st.sidebar.warning(f"Portable bundle unavailable: {st.session_state.portable_bundle_error}")

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
