"""Synthetic teaching models for gravity and magnetic potential fields.

Coordinates use a local profile frame: X is horizontal along profile, Z is
positive down, and the infinite-cylinder/dyke strike direction is Y. These
models are analytical teaching cases, not substitutes for the finite-prism
production model or field-data interpretation.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from engine.forward import Body2D, PotentialField2D


GRAVITATIONAL_CONSTANT = 6.67430e-11
VACUUM_PERMEABILITY = 4.0 * math.pi * 1.0e-7
SI_TO_MGAL = 1.0e5
TESLA_TO_NT = 1.0e9

MODEL_EQUATIONS = {
    "sphere_gravity": (
        "Uniform sphere point-mass field outside: gz=G*Δrho*(4*pi*R^3/3)*(zc-z)/r^3; "
        "inside: gz=(4*pi*G*Δrho/3)*(zc-z)."
    ),
    "cylinder_gravity": (
        "Infinite circular cylinder outside: gz=2*G*Δrho*pi*R^2*(zc-z)/r^2; "
        "inside: gz=2*pi*G*Δrho*(zc-z)."
    ),
    "dyke_gravity": (
        "Infinite-strike rectangular dyke: exact polygon boundary integral in the production 2-D engine."
    ),
    "sphere_magnetics": (
        "Uniform M=κ*F/μ0*uF+Jr*ur; outside B=μ0/(4*pi)*(3*r*(m·r)/|r|^5-m/|r|^3), "
        "inside B=2*μ0*M/3."
    ),
    "cylinder_magnetics": (
        "Uniform infinite circular cylinder outside H=(R^2/2)*(2*rperp*(Mperp·rperp)/rperp^4-Mperp/rperp^2), "
        "B=μ0*H; inside B=μ0*Mperp/2."
    ),
    "dyke_magnetics": (
        "Infinite-strike rectangular dyke: exact polygon boundary-charge field in the production 2-D engine; "
        "dT is the first-order projection onto the ambient-field direction."
    ),
}

MODEL_LIMITATIONS = (
    "Inputs are synthetic and use a flat local profile frame with Z positive down.",
    "Sphere and cylinder magnetic responses assume uniform magnetization and omit self-demagnetization coupling to susceptibility.",
    "An infinite cylinder or dyke has no end effects; magnetization parallel to its strike does not contribute in this 2-D model.",
    "Total-field anomaly is the first-order projection of the modeled field onto the declared ambient-field direction.",
    "Noise is independent, zero-mean Gaussian with the stated standard deviation; it is not a complete field-noise model.",
    "Fit uncertainties are local linear approximations under the selected geometry and noise assumptions; they are not geological uncertainty.",
)

SYNTHETIC_GHANA_CONTEXT = {
    "crs": "EPSG:32630",
    "anchor_easting_m": 500_000.0,
    "anchor_northing_m": 663_204.567,
    "approximate_longitude_deg": -3.0,
    "approximate_latitude_deg": 6.0,
    "measured_data": False,
    "description": "Coordinate anchor for a synthetic classroom exercise only; no field observations or geological claim.",
}


@dataclass(frozen=True)
class TeachingExperiment:
    """A reproducible synthetic profile and its generating metadata."""

    x_m: np.ndarray
    z_m: np.ndarray
    clean_response: np.ndarray
    noise: np.ndarray
    observed_response: np.ndarray
    response_units: str
    model_kind: str
    field_type: str
    seed: int
    noise_sigma: float
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class SphereInversionResult:
    """Density/depth fit with local rank and uncertainty diagnostics."""

    success: bool
    message: str
    density_kg_m3: float
    center_depth_m: float
    rms_mgal: float
    weighted_rms: float
    numerical_rank: int
    condition_number: float | None
    identifiable: bool
    standard_errors: Mapping[str, float] | None


def _finite_scalar(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _profile_vectors(x_m, z_m=None) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(x_m, dtype=np.float64)
        z = np.zeros_like(x) if z_m is None else np.asarray(z_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Profile coordinates must be numeric") from exc
    if x.ndim != 1 or x.size == 0 or z.shape != x.shape:
        raise ValueError("Profile X and Z must be aligned non-empty one-dimensional vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
        raise ValueError("Profile coordinates must be finite")
    return x, z


def _buried_circle(center_depth_m, radius_m) -> tuple[float, float]:
    depth = _finite_scalar(center_depth_m, "center_depth_m")
    radius = _finite_scalar(radius_m, "radius_m")
    if radius <= 0.0 or depth <= radius:
        raise ValueError("A circular teaching body must have positive radius and lie below the surface")
    return depth, radius


def sphere_gravity_mgal(
    x_m,
    z_m=None,
    *,
    center_x_m: float = 0.0,
    center_depth_m: float = 400.0,
    radius_m: float = 150.0,
    density_contrast_kg_m3: float = 500.0,
) -> np.ndarray:
    """Vertical positive-down attraction of a homogeneous buried sphere.

    The outside formula is the point-mass field with ``M=4 pi rho R^3/3``.
    Inside the sphere, the field uses the enclosed mass of a uniform sphere.
    """
    x, z = _profile_vectors(x_m, z_m)
    depth, radius = _buried_circle(center_depth_m, radius_m)
    center_x = _finite_scalar(center_x_m, "center_x_m")
    density = _finite_scalar(density_contrast_kg_m3, "density_contrast_kg_m3")
    dx = x - center_x
    dz = depth - z
    distance_squared = dx * dx + dz * dz
    distance = np.sqrt(distance_squared)
    volume = (4.0 / 3.0) * math.pi * radius**3
    mass = density * volume
    result = np.zeros_like(x)
    outside = distance >= radius
    safe = distance[outside]
    result[outside] = GRAVITATIONAL_CONSTANT * mass * dz[outside] / safe**3
    inside = ~outside
    result[inside] = (
        (4.0 / 3.0)
        * math.pi
        * GRAVITATIONAL_CONSTANT
        * density
        * dz[inside]
    )
    return result * SI_TO_MGAL


def cylinder_gravity_mgal(
    x_m,
    z_m=None,
    *,
    center_x_m: float = 0.0,
    center_depth_m: float = 400.0,
    radius_m: float = 150.0,
    density_contrast_kg_m3: float = 500.0,
) -> np.ndarray:
    """Vertical positive-down attraction of a homogeneous infinite cylinder."""
    x, z = _profile_vectors(x_m, z_m)
    depth, radius = _buried_circle(center_depth_m, radius_m)
    center_x = _finite_scalar(center_x_m, "center_x_m")
    density = _finite_scalar(density_contrast_kg_m3, "density_contrast_kg_m3")
    dx = x - center_x
    dz = depth - z
    distance_squared = dx * dx + dz * dz
    result = np.zeros_like(x)
    outside = distance_squared >= radius * radius
    area = math.pi * radius * radius
    result[outside] = (
        2.0
        * GRAVITATIONAL_CONSTANT
        * density
        * area
        * dz[outside]
        / distance_squared[outside]
    )
    inside = ~outside
    result[inside] = 2.0 * math.pi * GRAVITATIONAL_CONSTANT * density * dz[inside]
    return result * SI_TO_MGAL


def _magnetic_vectors(
    field_strength_nT,
    field_inclination_deg,
    field_declination_deg,
    profile_azimuth_deg,
    susceptibility_si,
    remanence_a_m,
    remanent_inclination_deg,
    remanent_declination_deg,
) -> tuple[np.ndarray, np.ndarray]:
    strength = _finite_scalar(field_strength_nT, "field_strength_nT")
    inclination = _finite_scalar(field_inclination_deg, "field_inclination_deg")
    declination = _finite_scalar(field_declination_deg, "field_declination_deg")
    azimuth = _finite_scalar(profile_azimuth_deg, "profile_azimuth_deg")
    susceptibility = _finite_scalar(susceptibility_si, "susceptibility_si")
    remanence = _finite_scalar(remanence_a_m, "remanence_a_m")
    rem_inc = _finite_scalar(remanent_inclination_deg, "remanent_inclination_deg")
    rem_dec = _finite_scalar(remanent_declination_deg, "remanent_declination_deg")
    if strength <= 0.0 or remanence < 0.0:
        raise ValueError("Ambient field must be positive and remanence magnitude cannot be negative")
    if not -90.0 <= inclination <= 90.0 or not -90.0 <= rem_inc <= 90.0:
        raise ValueError("Magnetic inclinations must lie between -90 and 90 degrees")

    radians = math.pi / 180.0
    field_i = inclination * radians
    field_d = (declination - azimuth) * radians
    rem_i = rem_inc * radians
    rem_d = (rem_dec - azimuth) * radians
    unit_field = np.array([
        math.cos(field_i) * math.cos(field_d),
        math.cos(field_i) * math.sin(field_d),
        math.sin(field_i),
    ])
    ambient_h = (strength * 1.0e-9 / VACUUM_PERMEABILITY) * unit_field
    remanent = remanence * np.array([
        math.cos(rem_i) * math.cos(rem_d),
        math.cos(rem_i) * math.sin(rem_d),
        math.sin(rem_i),
    ])
    return unit_field, susceptibility * ambient_h + remanent


def sphere_magnetic_components_nT(
    x_m,
    z_m=None,
    *,
    center_x_m: float = 0.0,
    center_depth_m: float = 400.0,
    radius_m: float = 150.0,
    field_strength_nT: float = 49_000.0,
    field_inclination_deg: float = 8.0,
    field_declination_deg: float = -2.0,
    profile_azimuth_deg: float = 90.0,
    susceptibility_si: float = 0.03,
    remanence_a_m: float = 0.0,
    remanent_inclination_deg: float = 0.0,
    remanent_declination_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly magnetized sphere response, including its analytic interior field."""
    x, z = _profile_vectors(x_m, z_m)
    depth, radius = _buried_circle(center_depth_m, radius_m)
    center_x = _finite_scalar(center_x_m, "center_x_m")
    unit_field, magnetization = _magnetic_vectors(
        field_strength_nT,
        field_inclination_deg,
        field_declination_deg,
        profile_azimuth_deg,
        susceptibility_si,
        remanence_a_m,
        remanent_inclination_deg,
        remanent_declination_deg,
    )
    r = np.column_stack((x - center_x, np.zeros_like(x), z - depth))
    distance = np.linalg.norm(r, axis=1)
    moment = (4.0 / 3.0) * math.pi * radius**3 * magnetization
    field_t = np.zeros_like(r)
    outside = distance >= radius
    if np.any(outside):
        ro = r[outside]
        r2 = np.einsum("ij,ij->i", ro, ro)
        moment_dot_r = ro @ moment
        field_t[outside] = (VACUUM_PERMEABILITY / (4.0 * math.pi)) * (
            3.0 * ro * (moment_dot_r / r2**2.5)[:, None]
            - moment[None, :] / r2[:, None] ** 1.5
        )
    field_t[~outside] = (2.0 / 3.0) * VACUUM_PERMEABILITY * magnetization
    components = field_t * TESLA_TO_NT
    d_total = components @ unit_field
    return components[:, 0], components[:, 1], components[:, 2], d_total


def cylinder_magnetic_components_nT(
    x_m,
    z_m=None,
    *,
    center_x_m: float = 0.0,
    center_depth_m: float = 400.0,
    radius_m: float = 150.0,
    field_strength_nT: float = 49_000.0,
    field_inclination_deg: float = 8.0,
    field_declination_deg: float = -2.0,
    profile_azimuth_deg: float = 90.0,
    susceptibility_si: float = 0.03,
    remanence_a_m: float = 0.0,
    remanent_inclination_deg: float = 0.0,
    remanent_declination_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact 2-D field of a uniformly magnetized infinite circular cylinder.

    The cylinder is infinite along Y. Its end effects and response to a
    magnetization parallel to Y are therefore absent from this model.
    """
    x, z = _profile_vectors(x_m, z_m)
    depth, radius = _buried_circle(center_depth_m, radius_m)
    center_x = _finite_scalar(center_x_m, "center_x_m")
    unit_field, magnetization = _magnetic_vectors(
        field_strength_nT,
        field_inclination_deg,
        field_declination_deg,
        profile_azimuth_deg,
        susceptibility_si,
        remanence_a_m,
        remanent_inclination_deg,
        remanent_declination_deg,
    )
    dx = x - center_x
    dz = z - depth
    r2 = dx * dx + dz * dz
    field_t = np.zeros((x.size, 3), dtype=np.float64)
    outside = r2 >= radius * radius
    mx, mz = magnetization[0], magnetization[2]
    dot = mx * dx + mz * dz
    safe_r2 = r2[outside]
    factor = (radius * radius / 2.0)
    field_t[outside, 0] = VACUUM_PERMEABILITY * factor * (
        2.0 * dx[outside] * dot[outside] / safe_r2**2 - mx / safe_r2
    )
    field_t[outside, 2] = VACUUM_PERMEABILITY * factor * (
        2.0 * dz[outside] * dot[outside] / safe_r2**2 - mz / safe_r2
    )
    field_t[~outside, 0] = VACUUM_PERMEABILITY * mx / 2.0
    field_t[~outside, 2] = VACUUM_PERMEABILITY * mz / 2.0
    components = field_t * TESLA_TO_NT
    return components[:, 0], components[:, 1], components[:, 2], components @ unit_field


def forward_teaching_model(
    model_kind: str,
    field_type: str,
    x_m,
    z_m=None,
    *,
    center_x_m: float = 0.0,
    center_depth_m: float = 400.0,
    radius_m: float = 150.0,
    density_contrast_kg_m3: float = 500.0,
    dyke_width_m: float = 150.0,
    dyke_top_m: float = 200.0,
    dyke_bottom_m: float = 600.0,
    field_strength_nT: float = 49_000.0,
    field_inclination_deg: float = 8.0,
    field_declination_deg: float = -2.0,
    profile_azimuth_deg: float = 90.0,
    susceptibility_si: float = 0.03,
    remanence_a_m: float = 0.0,
    remanent_inclination_deg: float = 0.0,
    remanent_declination_deg: float = 0.0,
) -> np.ndarray:
    """Evaluate sphere, infinite-cylinder, or vertical infinite-strike-dyke models."""
    model = str(model_kind).strip().lower()
    field = str(field_type).strip().lower()
    x, z = _profile_vectors(x_m, z_m)
    if model not in {"sphere", "cylinder", "dyke"}:
        raise ValueError("model_kind must be 'sphere', 'cylinder', or 'dyke'")
    if field not in {"gravity", "magnetics"}:
        raise ValueError("field_type must be 'gravity' or 'magnetics'")
    circle = {"center_x_m": center_x_m, "center_depth_m": center_depth_m, "radius_m": radius_m}
    magnetic = {
        "field_strength_nT": field_strength_nT,
        "field_inclination_deg": field_inclination_deg,
        "field_declination_deg": field_declination_deg,
        "profile_azimuth_deg": profile_azimuth_deg,
        "susceptibility_si": susceptibility_si,
        "remanence_a_m": remanence_a_m,
        "remanent_inclination_deg": remanent_inclination_deg,
        "remanent_declination_deg": remanent_declination_deg,
    }

    if model == "sphere":
        if field == "gravity":
            return sphere_gravity_mgal(
                x, z, density_contrast_kg_m3=density_contrast_kg_m3, **circle
            )
        return sphere_magnetic_components_nT(x, z, **circle, **magnetic)[3]
    if model == "cylinder":
        if field == "gravity":
            return cylinder_gravity_mgal(
                x, z, density_contrast_kg_m3=density_contrast_kg_m3, **circle
            )
        return cylinder_magnetic_components_nT(x, z, **circle, **magnetic)[3]

    width = _finite_scalar(dyke_width_m, "dyke_width_m")
    top = _finite_scalar(dyke_top_m, "dyke_top_m")
    bottom = _finite_scalar(dyke_bottom_m, "dyke_bottom_m")
    center_x = _finite_scalar(center_x_m, "center_x_m")
    if width <= 0.0 or top < 0.0 or bottom <= top:
        raise ValueError("A teaching dyke needs positive width and 0 <= top < bottom")
    body = Body2D(
        "Synthetic teaching dyke",
        [center_x - width / 2.0, center_x + width / 2.0,
         center_x + width / 2.0, center_x - width / 2.0],
        [top, top, bottom, bottom],
        density=_finite_scalar(density_contrast_kg_m3, "density_contrast_kg_m3"),
        k_succ=_finite_scalar(susceptibility_si, "susceptibility_si"),
        J_r=_finite_scalar(remanence_a_m, "remanence_a_m"),
        I_r=_finite_scalar(remanent_inclination_deg, "remanent_inclination_deg"),
        D_r=_finite_scalar(remanent_declination_deg, "remanent_declination_deg"),
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )
    engine = PotentialField2D([body])
    if field == "gravity":
        return engine.calculate_gravity(x, z)
    return engine.calculate_magnetics_components(
        x,
        z,
        _finite_scalar(field_strength_nT, "field_strength_nT"),
        _finite_scalar(field_inclination_deg, "field_inclination_deg"),
        _finite_scalar(field_declination_deg, "field_declination_deg"),
        _finite_scalar(profile_azimuth_deg, "profile_azimuth_deg"),
    )[3]


def sphere_density_for_equal_center_gravity(
    shallow_density_kg_m3: float,
    shallow_depth_m: float,
    deep_depth_m: float,
) -> float:
    """Return the sphere contrast matching one surface station at X=center."""
    density = _finite_scalar(shallow_density_kg_m3, "shallow_density_kg_m3")
    shallow = _finite_scalar(shallow_depth_m, "shallow_depth_m")
    deep = _finite_scalar(deep_depth_m, "deep_depth_m")
    if shallow <= 0.0 or deep <= 0.0:
        raise ValueError("Sphere centre depths must be positive")
    return density * (deep / shallow) ** 2


def _json_compatible(value):
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Experiment parameter {type(value).__name__} cannot be written as JSON")


def sphere_density_depth_tradeoff(
    x_m,
    *,
    radius_m: float = 150.0,
    shallow_density_kg_m3: float = 500.0,
    shallow_depth_m: float = 350.0,
    deep_depth_m: float = 550.0,
) -> dict[str, object]:
    """Make two sphere profiles equal at one station and compare the full profiles."""
    x, z = _profile_vectors(x_m)
    _, radius = _buried_circle(shallow_depth_m, radius_m)
    _buried_circle(deep_depth_m, radius_m)
    shallow_depth = _finite_scalar(shallow_depth_m, "shallow_depth_m")
    deep_depth = _finite_scalar(deep_depth_m, "deep_depth_m")
    shallow_density = _finite_scalar(shallow_density_kg_m3, "shallow_density_kg_m3")
    if deep_depth <= shallow_depth:
        raise ValueError("deep_depth_m must exceed shallow_depth_m")
    matched_x = float(x[int(np.argmin(np.abs(x)))])
    shallow_distance = math.hypot(matched_x, shallow_depth)
    deep_distance = math.hypot(matched_x, deep_depth)
    deep_density = (
        shallow_density
        * (shallow_depth / deep_depth)
        * (deep_distance / shallow_distance) ** 3
    )
    shallow = sphere_gravity_mgal(
        x, z, center_depth_m=shallow_depth_m, radius_m=radius,
        density_contrast_kg_m3=shallow_density_kg_m3,
    )
    deep = sphere_gravity_mgal(
        x, z, center_depth_m=deep_depth_m, radius_m=radius,
        density_contrast_kg_m3=deep_density,
    )
    center = int(np.argmin(np.abs(x)))
    scale = float(np.sqrt(np.mean(shallow**2)))
    relative_rms = 0.0 if scale == 0.0 else float(np.sqrt(np.mean((shallow - deep) ** 2)) / scale)
    return {
        "shallow_response_mgal": shallow,
        "deep_response_mgal": deep,
        "deep_density_kg_m3": deep_density,
        "matched_station_index": center,
        "single_station_difference_mgal": float(shallow[center] - deep[center]),
        "profile_relative_rms_difference": relative_rms,
    }


def simulate_teaching_experiment(
    model_kind: str,
    field_type: str,
    *,
    x_m,
    z_m=None,
    noise_sigma: float = 0.0,
    seed: int = 20261124,
    **model_parameters,
) -> TeachingExperiment:
    """Create a model profile with reproducible, independently seeded Gaussian noise."""
    x, z = _profile_vectors(x_m, z_m)
    x = x.copy()
    z = z.copy()
    sigma = _finite_scalar(noise_sigma, "noise_sigma")
    if sigma < 0.0:
        raise ValueError("noise_sigma cannot be negative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    clean = forward_teaching_model(model_kind, field_type, x, z, **model_parameters)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=x.shape)
    observed = clean + noise
    for values in (x, z, clean, noise, observed):
        values.setflags(write=False)
    units = "mGal" if str(field_type).strip().lower() == "gravity" else "nT"
    return TeachingExperiment(
        x_m=x,
        z_m=z,
        clean_response=clean,
        noise=noise,
        observed_response=observed,
        response_units=units,
        model_kind=str(model_kind).strip().lower(),
        field_type=str(field_type).strip().lower(),
        seed=seed,
        noise_sigma=sigma,
        parameters=_json_compatible(dict(model_parameters)),
    )


def export_teaching_experiment_csv(experiment: TeachingExperiment) -> bytes:
    """Export deterministic synthetic observations and full run provenance."""
    metadata = {
        "synthetic": True,
        "model_kind": experiment.model_kind,
        "field_type": experiment.field_type,
        "response_units": experiment.response_units,
        "seed": experiment.seed,
        "noise_sigma": experiment.noise_sigma,
        "coordinate_context": SYNTHETIC_GHANA_CONTEXT,
        "parameters": dict(experiment.parameters),
        "equation": MODEL_EQUATIONS[f"{experiment.model_kind}_{experiment.field_type}"],
        "limitations": list(MODEL_LIMITATIONS),
    }
    encoded_metadata = json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False)
    output = io.StringIO(newline="")
    output.write(f"# {encoded_metadata}\n")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("x_m", "z_m_positive_down", "clean_response", "noise", f"observed_{experiment.response_units}"))
    writer.writerows(
        zip(
            experiment.x_m.tolist(),
            experiment.z_m.tolist(),
            experiment.clean_response.tolist(),
            experiment.noise.tolist(),
            experiment.observed_response.tolist(),
        )
    )
    return output.getvalue().encode("utf-8")


def invert_sphere_gravity_density_depth(
    x_m,
    observed_mgal,
    *,
    z_m=None,
    radius_m: float = 150.0,
    observation_sigma_mgal: float | np.ndarray = 0.05,
    initial_density_kg_m3: float = 350.0,
    initial_depth_m: float = 500.0,
    density_bounds_kg_m3: tuple[float, float] = (-5000.0, 5000.0),
    depth_bounds_m: tuple[float, float] | None = None,
) -> SphereInversionResult:
    """Fit sphere density contrast and depth; expose rank and local uncertainty."""
    x, z = _profile_vectors(x_m, z_m)
    observed = np.asarray(observed_mgal, dtype=np.float64)
    if observed.shape != x.shape or not np.all(np.isfinite(observed)):
        raise ValueError("observed_mgal must be finite and aligned with profile coordinates")
    radius = _finite_scalar(radius_m, "radius_m")
    if radius <= 0.0:
        raise ValueError("radius_m must be positive")
    sigma = np.asarray(observation_sigma_mgal, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(x.shape, float(sigma), dtype=np.float64)
    if sigma.shape != x.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("observation_sigma_mgal must be positive and aligned with observations")
    try:
        density_lower, density_upper = map(float, density_bounds_kg_m3)
    except (TypeError, ValueError) as exc:
        raise ValueError("density_bounds_kg_m3 must be an increasing pair") from exc
    if not math.isfinite(density_lower) or not math.isfinite(density_upper) or density_lower >= density_upper:
        raise ValueError("density_bounds_kg_m3 must be finite and increasing")
    minimum_depth = radius + max(1.0e-6, radius * 1.0e-9)
    if depth_bounds_m is None:
        depth_lower, depth_upper = minimum_depth, max(10_000.0, radius * 100.0)
    else:
        try:
            depth_lower, depth_upper = map(float, depth_bounds_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth_bounds_m must be an increasing pair") from exc
        depth_lower = max(depth_lower, minimum_depth)
    if not math.isfinite(depth_lower) or not math.isfinite(depth_upper) or depth_lower >= depth_upper:
        raise ValueError("depth_bounds_m must be finite, increasing, and keep the sphere below the surface")

    initial = np.array([
        _finite_scalar(initial_density_kg_m3, "initial_density_kg_m3"),
        _finite_scalar(initial_depth_m, "initial_depth_m"),
    ])
    lower = np.array([density_lower, depth_lower])
    upper = np.array([density_upper, depth_upper])
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError("Initial density and depth must lie strictly inside their bounds")

    def predicted(parameters):
        return sphere_gravity_mgal(
            x,
            z,
            radius_m=radius,
            density_contrast_kg_m3=parameters[0],
            center_depth_m=parameters[1],
        )

    fit = least_squares(
        lambda parameters: (predicted(parameters) - observed) / sigma,
        initial,
        bounds=(lower, upper),
        x_scale=np.array([500.0, max(radius, 100.0)]),
        max_nfev=3000,
    )
    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    tolerance = (
        max(fit.jac.shape) * np.finfo(np.float64).eps * singular_values[0]
        if singular_values.size and singular_values[0] > 0.0 else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size >= 2 and singular_values[-1] > 0.0 else None
    )
    identifiable = rank == 2 and condition is not None and condition < 1.0e12
    standard_errors = None
    if identifiable and x.size > 2:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1.0e-12)
        standard_errors = {
            "density_kg_m3": float(math.sqrt(max(0.0, covariance[0, 0]))),
            "center_depth_m": float(math.sqrt(max(0.0, covariance[1, 1]))),
        }
    residual = predicted(fit.x) - observed
    return SphereInversionResult(
        success=bool(fit.success),
        message=str(fit.message),
        density_kg_m3=float(fit.x[0]),
        center_depth_m=float(fit.x[1]),
        rms_mgal=float(np.sqrt(np.mean(residual**2))),
        weighted_rms=float(np.sqrt(np.mean((residual / sigma) ** 2))),
        numerical_rank=rank,
        condition_number=condition,
        identifiable=identifiable,
        standard_errors=standard_errors,
    )
