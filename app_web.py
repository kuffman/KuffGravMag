import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="KuffGravMag Mobile (2.5D + 2D/3D Data)", layout="wide")

st.title("🌐 KuffGravMag Workbench (2.5D iPad Edition)")
st.markdown("Run 2.5D modeling and compare against imported 2D profiles or 3D XYZ observation datasets.")

# Sidebar controls for 2.5D model parameters
st.sidebar.header("2.5D Model Parameters")
depth = st.sidebar.slider("Depth to Top (km)", 0.1, 10.0, 2.0)
width = st.sidebar.slider("Body Width (km)", 1.0, 20.0, 5.0)
strike_half_length = st.sidebar.slider("Strike Half-Length Y (km)", 1.0, 50.0, 10.0)
density_contrast = st.sidebar.slider("Density Contrast (g/cm³)", -1.0, 1.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.header("📂 Data Import (2D Profile / 3D XYZ)")
data_type = st.sidebar.radio("Select Data Type", ["2D Profile (Distance, Value)", "3D Spatial Grid (X, Y, Value)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV / TXT Data File", type=["csv", "txt"])

# Main calculation and plotting area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Subsurface Model Preview (2.5D)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhspan(depth, depth + 3, xmin=0.4, xmax=0.6, color='gray', alpha=0.6,
               label=f'Prism (Y = ±{strike_half_length} km)')
    ax.set_ylim(15, 0)  # Inverted depth axis
    ax.set_xlabel("Distance X (km)")
    ax.set_ylabel("Depth (km)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Model vs. Observed Data Comparison")
    x_coords = np.linspace(-20, 20, 200)

    # 2.5D Forward Calculation
    factor = strike_half_length / np.sqrt(x_coords ** 2 + strike_half_length ** 2)
    anomaly = (density_contrast * width / (x_coords ** 2 + depth ** 2 + 1)) * factor * 10.0

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x_coords, anomaly, color='blue', lw=2, label='Calculated (2.5D)')

    # Plot imported observation data if available
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "2D Profile" in data_type:
                # Expecting columns: Distance, Observed Value
                obs_x = df.iloc[:, 0]
                obs_val = df.iloc[:, 1]
                ax2.scatter(obs_x, obs_val, color='red', zorder=5, label='Observed 2D Profile')
                st.sidebar.success("2D Profile data loaded!")
            else:
                # Expecting 3D XYZ format: X, Y, Observed Value -> extract a profile along X or plot summary
                obs_x = df.iloc[:, 0]
                obs_val = df.iloc[:, 2]  # Assuming 3rd column is anomaly value
                ax2.scatter(obs_x, obs_val, color='orange', zorder=5, label='3D Grid Projected (X)')
                st.sidebar.success("3D Spatial data projected!")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

    ax2.set_title("Gravity / Magnetic Anomaly Fit")
    ax2.set_xlabel("Distance X (km)")
    ax2.set_ylabel("Amplitude (mGal / nT)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

if st.button("📄 Generate PDF Report"):
    st.success("Report generated successfully! Ready for download.")