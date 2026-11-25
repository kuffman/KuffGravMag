"""Render the web app and produce a small PDF report as an application smoke test."""

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from engine.forward import Body2D
from engine.processing import apply_gravity_reductions
from engine.report import generate_pdf_report
from engine.survey import SurveyProfile


def smoke_web():
    app = AppTest.from_file(str(PROJECT_ROOT / "app_web.py"), default_timeout=60).run()
    if app.exception:
        messages = "\n".join(str(item.message) for item in app.exception)
        raise RuntimeError(f"Web app raised during its first render:\n{messages}")
    if not any("KuffGravMag" in str(item.value) for item in app.title):
        raise RuntimeError("The web app title did not render")
    lab_errors = [
        str(item.value) for item in app.error
        if "Synthetic experiment could not be generated" in str(item.value)
    ]
    if lab_errors:
        raise RuntimeError(f"The Academic Lab web panel failed to render: {lab_errors}")
    if not any(
        "Download seeded synthetic experiment CSV" in str(item.label)
        for item in app.get("download_button")
    ):
        raise RuntimeError("The Academic Lab seeded CSV download did not render")
    if not any(
        "Equation: Uniform sphere" in str(item.value)
        for item in app.caption
    ):
        raise RuntimeError("The Academic Lab did not show its analytical model equation")

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
    app.run()
    if app.exception:
        messages = "\n".join(str(item.message) for item in app.exception)
        raise RuntimeError(f"Web app raised while rendering processed survey data:\n{messages}")


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
