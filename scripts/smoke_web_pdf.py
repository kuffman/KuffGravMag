"""Render the web app and produce a small PDF report as an application smoke test."""

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from engine.forward import Body2D
from engine.processing import apply_gravity_reductions
from engine.project_bundle import load_project_bundle
from engine.report import generate_pdf_report
from engine.spatial import load_xyz_layer
from engine.survey import SurveyProfile


def smoke_web():
    app = AppTest.from_file(str(PROJECT_ROOT / "app_web.py"), default_timeout=60).run()
    if app.exception:
        messages = "\n".join(str(item.message) for item in app.exception)
        raise RuntimeError(f"Web app raised during its first render:\n{messages}")
    if not any("KuffGravMag" in str(item.value) for item in app.title):
        raise RuntimeError("The web app title did not render")

    profile = SurveyProfile.from_csv(
        b"""distance_m,elevation_m,gravity_mgal
0,100,10
100,110,11
""",
        metadata={
            "coordinate_reference": "local profile coordinates",
            "profile_azimuth_deg": 0.0,
            "elevation_datum": "local survey benchmark",
        },
        source_name="smoke-profile.csv",
    )
    app.session_state["m6_profile"] = profile
    app.session_state["m6_products"] = [
        apply_gravity_reductions(profile, datum_elevation_m=100.0)
    ]
    raw_layer = b"x,y,value\n500000,663204.567,1\n500100,663204.567,2\n500000,663304.567,3\n"
    layer = load_xyz_layer(
        raw_layer,
        source_crs="EPSG:32630",
        target_crs="EPSG:4326",
        units="mGal",
        name="synthetic-ghana-grid.csv",
    )
    attachment_name = "synthetic-grid.csv"
    app.session_state["project_attachments"] = {attachment_name: raw_layer}
    app.session_state["spatial_layer"] = layer
    workflow_steps = list(app.session_state["m71_source_steps"])
    workflow_steps.append({
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
    app.session_state["m71_source_steps"] = workflow_steps
    app.run()
    if app.exception:
        messages = "\n".join(str(item.message) for item in app.exception)
        raise RuntimeError(f"Web app raised while rendering processed survey data:\n{messages}")
    bundle = load_project_bundle(app.session_state["portable_project_bundle"])
    if bundle["attachments"] != {attachment_name: raw_layer}:
        raise RuntimeError("The web app portable project did not preserve its embedded XYZ layer")
    if not any(
        step.get("name") == "local_spatial_layer"
        for step in bundle["project"]["processing_steps"]
    ):
        names = [step.get("name") for step in bundle["project"]["processing_steps"]]
        raise RuntimeError(f"The web app project bundle lost local-layer provenance: {names}")


def smoke_pdf():
    body = Body2D(
        "Smoke test body",
        [-100.0, 100.0, 100.0, -100.0, -100.0],
        [200.0, 200.0, 400.0, 400.0, 200.0],
        density=500.0,
        k_succ=0.02,
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )
    with tempfile.TemporaryDirectory(prefix="kuffgravmag-pdf-smoke-") as directory:
        path = Path(directory) / "smoke-report.pdf"
        generate_pdf_report(
            str(path), "", [body], {"F": 48_000.0, "Inc": 55.0, "Dec": 0.0, "Alpha": 0.0}
        )
        data = path.read_bytes()
        if not data.startswith(b"%PDF-") or len(data) < 1000:
            raise RuntimeError("PDF report was missing or incomplete")


def main():
    smoke_web()
    smoke_pdf()
    print("Web app render and PDF report smoke passed")


if __name__ == "__main__":
    main()
