"""Generate and verify the deterministic KGM-M5 scientific validation report."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import tomllib

import numpy as np

from engine.forward import Body2D, PotentialField2D
from validation.analytical import (
    centered_rectangle_infinite_gravity_gz_mgal,
    dipole_magnetic_field_nT,
    point_mass_gravity_gz_mgal,
    profile_magnetization_a_m,
    polygon_infinite_strike_gravity_reference,
    polygon_infinite_strike_magnetic_reference_nT,
    rectangular_prism_gravity_gz_mgal,
    rectangular_prism_volume_integral,
)
from validation.harmonica_reference import (
    harmonica_rectangular_prism_gravity_gz_mgal,
    harmonica_rectangular_prism_magnetic_components_nT,
)
from validation.inversion_studies import (
    density_depth_tradeoff,
    geometry_identifiability_study,
    run_noise_recovery_ensemble,
)


def _body(*, density=500.0, susceptibility=0.04, remanence=1.7, remanent_inc=30.0,
          remanent_dec=65.0, strike_mode="finite", top=200.0, bottom=400.0):
    return Body2D(
        "validation prism",
        [-100.0, 100.0, 100.0, -100.0, -100.0],
        [top, top, bottom, bottom, top],
        density=density,
        k_succ=susceptibility,
        J_r=remanence,
        I_r=remanent_inc,
        D_r=remanent_dec,
        y_min=-300.0,
        y_max=500.0,
        strike_mode=strike_mode,
    )


def _benchmark_values() -> dict:
    x_bounds, y_bounds, z_bounds = (-100.0, 100.0), (-300.0, 500.0), (200.0, 400.0)
    x_obs = np.array([25.0, -60.0, 0.0])
    z_obs = np.array([0.0, 15.0, 199.999999])
    body = _body()
    engine = PotentialField2D([body])
    calculated_gravity = engine.calculate_gravity(x_obs, z_obs)
    analytical_gravity = np.array(
        [
            rectangular_prism_gravity_gz_mgal(
                x_bounds, y_bounds, z_bounds, body.density, (x, 0.0, z)
            )
            for x, z in zip(x_obs, z_obs)
        ]
    )
    scale_residuals = []
    for scale in (1e-3, 1.0, 1e3):
        scaled = _body()
        scaled.x *= scale
        scaled.z *= scale
        scaled.y_min *= scale
        scaled.y_max *= scale
        actual = PotentialField2D([scaled]).calculate_gravity(
            np.array([25.0 * scale]), np.array([0.0])
        )[0]
        expected = rectangular_prism_gravity_gz_mgal(
            (-100.0 * scale, 100.0 * scale),
            (-300.0 * scale, 500.0 * scale),
            (200.0 * scale, 400.0 * scale),
            body.density,
            (25.0 * scale, 0.0, 0.0),
        )
        scale_residuals.append(abs(actual - expected))

    field = (48_000.0, 35.0, 18.0, 25.0)
    magnetization = profile_magnetization_a_m(
        susceptibility=body.k_succ,
        remanent_intensity=body.J_r,
        remanent_inclination=body.I_r,
        remanent_declination=body.D_r,
        field_intensity_nT=field[0],
        field_inclination=field[1],
        field_declination=field[2],
        profile_azimuth=field[3],
    )
    calculated_magnetic = engine.calculate_magnetics_components(
        np.array([25.0]), np.array([0.0]), *field
    )
    volume_gravity, volume_magnetic = rectangular_prism_volume_integral(
        x_bounds,
        y_bounds,
        z_bounds,
        body.density,
        magnetization,
        (25.0, 0.0, 0.0),
        order=48,
    )
    max_volume_magnetic_residual = max(
        abs(float(component[0]) - float(reference))
        for component, reference in zip(calculated_magnetic[:3], volume_magnetic)
    )

    polygon_vertices = np.array(
        [(-140.0, 160.0), (50.0, 160.0), (50.0, 280.0), (120.0, 280.0),
         (120.0, 430.0), (-140.0, 430.0)]
    )
    polygon_observation = (260.0, 12.0)
    polygon_density = 520.0
    polygon_body = Body2D(
        "concave polygon benchmark",
        polygon_vertices[:, 0],
        polygon_vertices[:, 1],
        density=polygon_density,
        J_r=1.0,
        I_r=30.0,
        D_r=0.0,
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )
    polygon_engine = PotentialField2D([polygon_body])
    polygon_actual_gravity = float(
        polygon_engine.calculate_gravity(
            np.array([polygon_observation[0]]), np.array([polygon_observation[1]])
        )[0]
    )
    polygon_actual_magnetic = polygon_engine.calculate_magnetics_components(
        np.array([polygon_observation[0]]),
        np.array([polygon_observation[1]]),
        0.0,
        0.0,
        0.0,
        0.0,
    )
    polygon_expected_gravity = polygon_infinite_strike_gravity_reference(
        polygon_vertices, polygon_density, polygon_observation, order=64
    )
    polygon_expected_magnetic = polygon_infinite_strike_magnetic_reference_nT(
        polygon_vertices,
        (math.cos(math.radians(30.0)), math.sin(math.radians(30.0))),
        polygon_observation,
        order=64,
    )

    harmonica_gravity_residuals = []
    for x, z, actual in zip(x_obs[:2], z_obs[:2], calculated_gravity[:2]):
        reference = harmonica_rectangular_prism_gravity_gz_mgal(
            x_bounds,
            y_bounds,
            z_bounds,
            body.density,
            (x, 0.0, z),
            profile_azimuth=90.0,
        )
        harmonica_gravity_residuals.append(abs(float(actual) - reference))

    alpha = 90.0
    harmonica_magnetization = profile_magnetization_a_m(
        susceptibility=body.k_succ,
        remanent_intensity=body.J_r,
        remanent_inclination=body.I_r,
        remanent_declination=body.D_r,
        field_intensity_nT=field[0],
        field_inclination=field[1],
        field_declination=field[2],
        profile_azimuth=alpha,
    )
    alpha_field = (field[0], field[1], field[2], alpha)
    actual_magnetic = engine.calculate_magnetics_components(
        np.array([25.0]), np.array([0.0]), *alpha_field
    )
    harmonica_magnetic = harmonica_rectangular_prism_magnetic_components_nT(
        x_bounds,
        y_bounds,
        z_bounds,
        harmonica_magnetization,
        25.0,
        0.0,
        *alpha_field,
    )

    center = (0.0, 100.0, 300.0)
    observation = (20_000.0, 0.0, 0.0)
    volume = 200.0 * 800.0 * 200.0
    far_gravity_engine = PotentialField2D([body])
    far_gravity = float(
        far_gravity_engine.calculate_gravity(
            np.array([observation[0]]), np.array([observation[2]])
        )[0]
    )
    far_gravity_analytic = point_mass_gravity_gz_mgal(
        volume=volume, density=body.density, center=center, observation=observation
    )
    far_magnetic_body = _body(
        density=0.0,
        susceptibility=0.0,
        remanence=1.0,
        remanent_inc=0.0,
        remanent_dec=0.0,
    )
    far_magnetic_engine = PotentialField2D([far_magnetic_body])
    far_magnetic = np.array(
        [component[0] for component in far_magnetic_engine.calculate_magnetics_components(
            np.array([observation[0]]), np.array([observation[2]]), 0.0, 0.0, 0.0, 0.0
        )[:3]]
    )
    far_dipole = dipole_magnetic_field_nT(
        volume=volume, magnetization=(1.0, 0.0, 0.0), center=center, observation=observation
    )

    magnetic_body = _body(density=0.0, susceptibility=0.0, remanence=1.0,
                          remanent_inc=0.0, remanent_dec=0.0)
    near_field = PotentialField2D([magnetic_body]).calculate_magnetics_components(
        np.array([0.0]), np.array([200.0 - 1e-6]), 0.0, 0.0, 0.0, 0.0
    )
    return {
        "infinite_rectangle_gravity_mgal": centered_rectangle_infinite_gravity_gz_mgal(
            half_width=100.0, top=200.0, bottom=400.0, density=500.0
        ),
        "finite_prism_max_gravity_residual_mgal": float(
            np.max(np.abs(calculated_gravity - analytical_gravity))
        ),
        "finite_prism_volume_gravity_residual_mgal": abs(float(calculated_gravity[0]) - volume_gravity),
        "finite_prism_max_volume_magnetic_residual_nT": max_volume_magnetic_residual,
        "concave_polygon_gravity_residual_mgal": abs(
            polygon_actual_gravity - polygon_expected_gravity
        ),
        "concave_polygon_max_magnetic_residual_nT": max(
            abs(float(polygon_actual_magnetic[0][0]) - polygon_expected_magnetic[0]),
            abs(float(polygon_actual_magnetic[2][0]) - polygon_expected_magnetic[1]),
        ),
        "harmonica_max_gravity_residual_mgal": max(harmonica_gravity_residuals),
        "harmonica_max_magnetic_component_residual_nT": max(
            abs(float(component[0]) - float(reference))
            for component, reference in zip(actual_magnetic[:3], harmonica_magnetic[:3])
        ),
        "scale_max_gravity_residual_mgal": max(scale_residuals),
        "near_surface_all_components_finite": bool(
            all(np.all(np.isfinite(component)) for component in near_field)
        ),
        "far_field_gravity_relative_residual": abs(far_gravity - far_gravity_analytic)
        / abs(far_gravity_analytic),
        "far_field_magnetic_relative_residual": float(
            np.linalg.norm(far_magnetic - far_dipole) / np.linalg.norm(far_dipole)
        ),
    }


def render_validation_report() -> str:
    """Return byte-stable Markdown metrics, tolerances, and run provenance."""
    values = _benchmark_values()
    ensemble = run_noise_recovery_ensemble(seed=20261124, trials=32)
    tradeoff = density_depth_tradeoff()
    identifiability = geometry_identifiability_study()
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    records = ensemble["records"]
    project = tomllib.loads(
        Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    )
    lines = [
        "# KuffGravMag KGM-M5 scientific validation report",
        "",
        "Generated deterministically by `uv run --locked --group validation python -m validation.report --write docs/VALIDATION_REPORT.md`.",
        "The report includes no wall-clock timestamp; the seed, trial count, dependency pin, and lockfile digest identify the run.",
        "",
        "## Provenance",
        "",
        f"- Python requirement: `{project['project']['requires-python']}`",
        f"- NumPy: `{version('numpy')}`; SciPy: `{version('scipy')}`; Pandas: `{version('pandas')}`; Harmonica: `{version('harmonica')}`",
        f"- Harmonica is an optional validation dependency pinned to `{version('harmonica')}`; it is not a runtime application dependency.",
        f"- `uv.lock` SHA-256: `{lock_hash}`",
        f"- Inversion noise ensemble: seed `{ensemble['seed']}`, `{ensemble['trials']}` paired trials per level.",
        "",
        "## M5.1 analytical and independent references",
        "",
        f"- Centered infinite-strike rectangle analytical gravity: `{values['infinite_rectangle_gravity_mgal']:.9f} mGal`.",
        f"- Maximum finite-prism gravity residual against the independent Nagy corner sum: `{values['finite_prism_max_gravity_residual_mgal']:.3e} mGal`.",
        f"- Finite-prism gravity residual against independent 48-point tensor volume quadrature: `{values['finite_prism_volume_gravity_residual_mgal']:.3e} mGal`.",
        f"- Maximum finite-prism magnetic component residual against independent 48-point tensor volume quadrature: `{values['finite_prism_max_volume_magnetic_residual_nT']:.3e} nT`.",
        f"- Concave infinite-strike polygon residual against 64-point-per-axis area quadrature: gravity `{values['concave_polygon_gravity_residual_mgal']:.3e} mGal`; maximum magnetic component `{values['concave_polygon_max_magnetic_residual_nT']:.3e} nT`.",
        "",
        "## M5.2 Harmonica comparison",
        "",
        f"- Maximum downward gravity residual: `{values['harmonica_max_gravity_residual_mgal']:.3e} mGal`.",
        f"- Maximum `dX`, `dY`, or `dZ` residual: `{values['harmonica_max_magnetic_component_residual_nT']:.3e} nT`.",
        "- Comparison uses Harmonica 0.7.0 prism models. KuffGravMag local coordinates map to easting/northing/upward by a 90° eastward profile azimuth, reversed upward/downward signs, and transformed strike bounds and magnetization. Both packages report magnetic components in nT; gravity is compared in mGal with the downward component selected.",
        "",
        "## M5.3 numerical stress results",
        "",
        f"- Maximum gravity residual across 1 mm, 1 m, and 1 km geometry scales: `{values['scale_max_gravity_residual_mgal']:.3e} mGal`.",
        f"- Near-face magnetic components remain finite at a 1 µm separation: `{str(values['near_surface_all_components_finite']).lower()}`.",
        f"- At a 20 km observation offset, finite-prism gravity differs from its point-mass asymptote by `{values['far_field_gravity_relative_residual']:.3e}` relative; the magnetic field differs from its dipole asymptote by `{values['far_field_magnetic_relative_residual']:.3e}` relative.",
        "- The test suite also exercises winding reversal, malformed polygons, zero-width bodies, explicit singular magnetic vertices, and locked regression cases.",
        "",
        "## M5.4 inversion reliability",
        "",
        "Fixed-geometry density is recovered from synthetic profiles with paired standardized noise at three stated standard deviations. Uncertainty is the inversion engine's local linear 95% approximation, not geological uncertainty.",
        "",
        "| Noise σ (mGal) | RMSE density (kg/m³) | Mean local standard error (kg/m³) | 95% interval coverage | Identifiable fits |",
        "|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        coverage = "n/a" if record["coverage_95"] is None else f"{record['coverage_95']:.3f}"
        lines.append(
            f"| {record['sigma_mgal']:.2f} | {record['rmse_density_kg_m3']:.3f} | "
            f"{record['mean_standard_error_kg_m3']:.3f} | {coverage} | "
            f"{record['identifiable_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"A shallow `{tradeoff['shallow_density_kg_m3']:.0f} kg/m³` body with top depth `{tradeoff['shallow_top_m']:.0f} m` and a deeper `{tradeoff['deep_density_kg_m3']:.1f} kg/m³` body with top depth `{tradeoff['deep_top_m']:.0f} m` match one station to relative residual `{tradeoff['single_station_relative_residual']:.3e}`. Across `{tradeoff['profile_observation_count']}` profile stations their relative RMS difference is `{tradeoff['full_profile_relative_rms_difference']:.3f}`. This is a concrete density-depth tradeoff, not a unique geological interpretation.",
            f"A separate fit uses `{identifiability['observation_count']}` observations for `{identifiability['parameter_count']}` density/geometry parameters. Its optimizer converges (`{str(identifiability['success']).lower()}`), but the data Jacobian rank is `{identifiability['numerical_rank']}/{identifiability['parameter_count']}`; locally identifiable is `{str(identifiability['identifiable']).lower()}`, and standard errors are available is `{str(identifiability['uncertainty_available']).lower()}`.",
            "",
            "## Tolerances and limitations",
            "",
            "- Analytical rectangle/prism gravity checks use relative tolerances from `2e-11` to `5e-10`, with absolute tolerances scaled to the modeled geometry. Harmonica gravity comparison uses `rtol=2e-10`, `atol=2e-11 mGal`; magnetic component comparison uses `rtol=3e-8`, `atol=3e-8 nT`.",
            "- The local profile-to-global transform is axis aligned only at 90° profile-azimuth increments; the Harmonica regression intentionally uses 90° and does not claim validation at arbitrary azimuths.",
            "- Rectangular-prism references validate the finite 3D geometry supported by the engine. Spheres and infinite cylinders are not production body geometries and are not used as acceptance oracles. A 2D rectangular section is the supported dyke-style idealization.",
            "- Synthetic tests validate equations, conventions, numerical behavior, and the inversion workflow on controlled inputs. They do not validate field-data accuracy, remanence uncertainty, self-demagnetization, arbitrary 3D geology, or uniqueness of potential-field inversion.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", metavar="PATH", help="write the deterministic report")
    group.add_argument("--check", metavar="PATH", help="fail unless PATH matches the regenerated report")
    args = parser.parse_args(argv)
    report = render_validation_report()
    if args.write:
        Path(args.write).write_text(report, encoding="utf-8")
        return 0
    if args.check:
        target = Path(args.check)
        if not target.is_file():
            parser.error(f"report does not exist: {target}")
        if target.read_text(encoding="utf-8") != report:
            print(f"Validation report is stale: {target}")
            return 1
        print(f"Validation report matches: {target}")
        return 0
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
