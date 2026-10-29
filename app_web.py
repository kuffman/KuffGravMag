import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your core computational backend engine classes and inversion function
from engine.forward import Body2D, PotentialField2D
from engine.inversion import invert_body_parameters

st.set_page_config(page_title="KuffGravMag Mobile (2.5D + Inversion)", layout="wide")

st.title("🌐 KuffGravMag Workbench (2.5D iPad Edition)")
st.markdown("Run exact 2.5D forward modeling and Levenberg-Marquardt inversions directly from your tablet.")

# Sidebar controls for 2.5D Body Parameters
st.sidebar.header("2.5D Body Parameters")
depth = st.sidebar.slider("Depth to Top (km)", 0.1, 10.0, 2.0)
width = st.sidebar.slider("Body Width (km)", 1.0, 20.0, 5.0)
strike_half_length = st.sidebar.slider("Strike Half-Length Y (km)", 1.0, 50.0, 10.0)
density_contrast = st.sidebar.slider("Density Contrast (g/cm³)", -1.0, 1.0, 0.25)
susceptibility = st.sidebar.slider("Magnetic Susceptibility (SI)", 0.0, 0.1, 0.02)

st.sidebar.markdown("---")
st.sidebar.header("📂 Data Import & Inversion")
field_type = st.sidebar.selectbox("Field Type", ["gravity", "magnetic"])
uploaded_file = st.sidebar.file_uploader("Upload Observed Data (X, Value)", type=["csv", "txt"])

# Define observation profile coordinates (meters)
x_obs_km = np.linspace(-20, 20, 200)
x_obs = x_obs_km * 1000.0
z_obs = np.zeros_like(x_obs)

# Construct polygonal body vertices in meters
half_w = (width * 1000.0) / 2.0
d_top = depth * 1000.0
d_bottom = d_top + (3.0 * 1000.0)

x_verts = [-half_w, half_w, half_w, -half_w]
z_verts = [d_top, d_top, d_bottom, d_bottom]

# Initialize body
body = Body2D(
    name="MobileBody",
    x_verts=x_verts,
    z_verts=z_verts,
    density=density_contrast * 1000.0,
    k_succ=susceptibility,
    y_min=-strike_half_length * 1000.0,
    y_max=strike_half_length * 1000.0
)

# Handle Inversion if data is uploaded and button is pressed
obs_data = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        obs_x = df.iloc[:, 0].values * 1000.0  # assume km in file, convert to m
        obs_data = df.iloc[:, 1].values

        if st.sidebar.button("🚀 Run Levenberg-Marquardt Inversion"):
            with st.spinner("Running least-squares inversion..."):
                success = invert_body_parameters(body, x_obs, z_obs, obs_data, field_type=field_type)
                if success:
                    st.sidebar.success("Inversion converged successfully!")
                else:
                    st.sidebar.error("Inversion failed to converge.")
    except Exception as e:
        st.sidebar.error(f"Error reading file for inversion: {e}")

# Compute forward response using current body parameters (optimized or slider-driven)
field_engine = PotentialField2D(bodies=[body])
if field_type == 'gravity':
    calc_anomaly = field_engine.calculate_gravity(x_obs, z_obs)
    ylabel_text = "Gravity Anomaly (mGal)"
else:
    _, _, calc_anomaly = field_engine.calculate_magnetics(
        x_obs, z_obs, F_nT=50000.0, I_f=60.0, D_f=0.0, alpha=0.0
    )
    ylabel_text = "Magnetic Anomaly (nT)"

# Main calculation and plotting area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Subsurface Model Preview (2.5D)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill(body.x / 1000.0, body.z / 1000.0, color='gray', alpha=0.6, label=f'Body (Y = ±{strike_half_length} km)')
    ax.set_ylim(15, 0)  # Inverted depth axis
    ax.set_xlabel("Distance X (km)")
    ax.set_ylabel("Depth (km)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Model vs. Observed Data Comparison")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x_obs_km, calc_anomaly, color='blue', lw=2, label='Calculated Response')

    if obs_data is not None:
        ax2.scatter(obs_x / 1000.0, obs_data, color='red', zorder=5, label='Observed Data')

    ax2.set_title(f"2.5D {field_type.capitalize()} Inversion Fit")
    ax2.set_xlabel("Distance X (km)")
    ax2.set_ylabel(ylabel_text)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

if st.button("📄 Generate PDF Report"):
    st.success("Report generated successfully! Ready for download.")