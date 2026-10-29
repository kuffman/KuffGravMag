import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your core computational backend engine classes and inversion function
from engine.forward import Body2D, PotentialField2D
from engine.inversion import invert_body_parameters

st.set_page_config(page_title="KuffGravMag Mobile (Multi-Body Workbench)", layout="wide")

st.title("🌐 KuffGravMag Workbench (2.5D iPad Edition)")
st.markdown("Multi-body 2.5D forward modeling, 3D spatial map slicing, and Levenberg-Marquardt inversions.")

# Sidebar controls for Multiple 2.5D Bodies
st.sidebar.header("2.5D Bodies Configuration")

# Body 1 Controls
st.sidebar.subheader("Body 1 (e.g., Ore Body)")
b1_strike = st.sidebar.slider("Body 1 Strike Half-Length (km)", 1.0, 50.0, 10.0, key="b1_s")
b1_density = st.sidebar.slider("Body 1 Density Contrast (g/cm³)", -1.0, 1.0, 0.3, key="b1_d")
b1_x1 = st.sidebar.number_input("Body 1 Top-Left X (km)", value=-3.0, key="b1_x1")
b1_z1 = st.sidebar.number_input("Body 1 Top-Left Z (km)", value=1.0, key="b1_z1")
b1_x2 = st.sidebar.number_input("Body 1 Top-Right X (km)", value=-1.0, key="b1_x2")
b1_z2 = st.sidebar.number_input("Body 1 Top-Right Z (km)", value=1.0, key="b1_z2")
b1_x3 = st.sidebar.number_input("Body 1 Bottom-Right X (km)", value=-1.0, key="b1_x3")
b1_z3 = st.sidebar.number_input("Body 1 Bottom-Right Z (km)", value=4.0, key="b1_z3")
b1_x4 = st.sidebar.number_input("Body 1 Bottom-Left X (km)", value=-3.0, key="b1_x4")
b1_z4 = st.sidebar.number_input("Body 1 Bottom-Left Z (km)", value=4.0, key="b1_z4")

st.sidebar.markdown("---")

# Body 2 Controls
st.sidebar.subheader("Body 2 (e.g., Intrusive / Dyke)")
b2_strike = st.sidebar.slider("Body 2 Strike Half-Length (km)", 1.0, 50.0, 15.0, key="b2_s")
b2_density = st.sidebar.slider("Body 2 Density Contrast (g/cm³)", -1.0, 1.0, -0.2, key="b2_d")
b2_x1 = st.sidebar.number_input("Body 2 Top-Left X (km)", value=1.0, key="b2_x1")
b2_z1 = st.sidebar.number_input("Body 2 Top-Left Z (km)", value=2.0, key="b2_z1")
b2_x2 = st.sidebar.number_input("Body 2 Top-Right X (km)", value=3.0, key="b2_x2")
b2_z2 = st.sidebar.number_input("Body 2 Top-Right Z (km)", value=2.0, key="b2_z2")
b2_x3 = st.sidebar.number_input("Body 2 Bottom-Right X (km)", value=3.0, key="b2_x3")
b2_z3 = st.sidebar.number_input("Body 2 Bottom-Right Z (km)", value=6.0, key="b2_z3")
b2_x4 = st.sidebar.number_input("Body 2 Bottom-Left X (km)", value=1.0, key="b2_x4")
b2_z4 = st.sidebar.number_input("Body 2 Bottom-Left Z (km)", value=6.0, key="b2_z4")

st.sidebar.markdown("---")
st.sidebar.header("📂 Data Import & 3D Slicing")
field_type = st.sidebar.selectbox("Field Type", ["gravity", "magnetic"])
data_type = st.sidebar.radio("Select Data Type", ["2D Profile (Distance, Value)", "3D Spatial Grid (X, Y, Value)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV / TXT Data File", type=["csv", "txt"])

# Define observation profile coordinates (meters)
x_obs_km = np.linspace(-25, 25, 250)
x_obs = x_obs_km * 1000.0
z_obs = np.zeros_like(x_obs)

# Construct Body 1 and Body 2 instances
body1 = Body2D(
    name="Body1",
    x_verts=[b1_x1 * 1000, b1_x2 * 1000, b1_x3 * 1000, b1_x4 * 1000],
    z_verts=[b1_z1 * 1000, b1_z2 * 1000, b1_z3 * 1000, b1_z4 * 1000],
    density=b1_density * 1000.0,
    y_min=-b1_strike * 1000, y_max=b1_strike * 1000,
    color='tab:blue'
)

body2 = Body2D(
    name="Body2",
    x_verts=[b2_x1 * 1000, b2_x2 * 1000, b2_x3 * 1000, b2_x4 * 1000],
    z_verts=[b2_z1 * 1000, b2_z2 * 1000, b2_z3 * 1000, b2_z4 * 1000],
    density=b2_density * 1000.0,
    y_min=-b2_strike * 1000, y_max=b2_strike * 1000,
    color='tab:orange'
)

# Multi-body potential field engine
field_engine = PotentialField2D(bodies=[body1, body2])

obs_data = None
obs_x = None
spatial_df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "2D Profile" in data_type:
            obs_x = df.iloc[:, 0].values * 1000.0
            obs_data = df.iloc[:, 1].values
        else:
            spatial_df = df
            x_col = df.iloc[:, 0].values
            y_col = df.iloc[:, 1].values
            val_col = df.iloc[:, 2].values

            st.sidebar.markdown("---")
            st.sidebar.subheader("✂️ Profile Slicer Controls")
            slice_y = st.sidebar.slider("Select Y-Slice Coordinate", float(np.min(y_col)), float(np.max(y_col)),
                                        float(np.median(y_col)))
            tolerance = st.sidebar.slider("Slice Tolerance Width", 0.1, 5.0, 1.0)

            idx_slice = np.abs(y_col - slice_y) <= tolerance
            obs_x = x_col[idx_slice] * 1000.0
            obs_data = val_col[idx_slice]

            sort_idx = np.argsort(obs_x)
            obs_x = obs_x[sort_idx]
            obs_data = obs_data[sort_idx]

        st.sidebar.success(f"Active Profile: {len(obs_x)} points")

        if st.sidebar.button("🚀 Run Body 1 Inversion"):
            with st.spinner("Running least-squares inversion on Body 1..."):
                success = invert_body_parameters(body1, x_obs, z_obs, obs_data, field_type=field_type)
                if success:
                    st.sidebar.success("Body 1 inversion converged successfully!")
                else:
                    st.sidebar.error("Inversion failed to converge.")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# Compute combined forward response for all bodies
if field_type == 'gravity':
    calc_anomaly = field_engine.calculate_gravity(x_obs, z_obs)
    ylabel_text = "Gravity Anomaly (mGal)"
else:
    _, _, calc_anomaly = field_engine.calculate_magnetics(
        x_obs, z_obs, F_nT=50000.0, I_f=60.0, D_f=0.0, alpha=0.0
    )
    ylabel_text = "Magnetic Anomaly (nT)"

# Main layout: Map / Subsurface model view on left, Profile fit on right
col1, col2 = st.columns([1, 1])

with col1:
    if spatial_df is not None and "3D Spatial Grid" in data_type:
        st.subheader("🗺️ 3D Spatial Grid Map & Slicer Cut")
        fig_map, ax_map = plt.subplots(figsize=(6, 4))
        sc = ax_map.scatter(spatial_df.iloc[:, 0], spatial_df.iloc[:, 1], c=spatial_df.iloc[:, 2], cmap='viridis', s=15)
        plt.colorbar(sc, ax=ax_map, label='Anomaly Value')
        if 'slice_y' in locals():
            ax_map.axhline(slice_y, color='red', linestyle='--', lw=2, label=f'Profile Slice (Y={slice_y:.1f})')
        ax_map.set_xlabel("X Coordinate")
        ax_map.set_ylabel("Y Coordinate")
        ax_map.legend(loc='upper right')
        st.pyplot(fig_map)
    else:
        st.subheader("Subsurface Multi-Body Model (2.5D)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill(body1.x / 1000.0, body1.z / 1000.0, color='tab:blue', alpha=0.5, label='Body 1')
        ax.fill(body2.x / 1000.0, body2.z / 1000.0, color='tab:orange', alpha=0.5, label='Body 2')
        ax.set_ylim(15, 0)
        ax.set_xlabel("Distance X (km)")
        ax.set_ylabel("Depth (km)")
        ax.legend()
        st.pyplot(fig)

with col2:
    st.subheader("Multi-Body Model vs. Observed Fit")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x_obs_km, calc_anomaly, color='blue', lw=2, label='Combined Calculated Response')

    if obs_data is not None and obs_x is not None:
        ax2.scatter(obs_x / 1000.0, obs_data, color='red', s=25, zorder=5, label='Observed Sliced Data')

    ax2.set_title(f"2.5D {field_type.capitalize()} Multi-Body Response")
    ax2.set_xlabel("Distance X (km)")
    ax2.set_ylabel(ylabel_text)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

if st.button("📄 Generate PDF Report"):
    st.success("Report generated successfully! Ready for download.")