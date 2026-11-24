# Gravity method and scientific follow-ups

## Coordinates, units, and assumptions

- `x` is horizontal profile distance in metres.
- `z` is positive downward in metres. The observation coordinates use the same datum; an observation above `z = 0` has negative `z_obs`.
- Each polygon has a uniform density contrast `Δρ` in `kg/m³` relative to its surroundings. Negative contrast is allowed.
- The 2D reference model is invariant along strike. The calculated component `g_z` is positive downward and is returned in mGal (`1 mGal = 10⁻⁵ m/s²`).
- Polygon winding is arbitrary. Polygons must be finite, non-self-intersecting, and enclose nonzero area. Repeated adjacent vertices are removed; an observation may lie on an edge or vertex.
- Contributions from separate bodies add linearly.

## Infinite-strike gravity equation

Let an observation be `(x₀, z₀)` and a source point in the cross-section be `(x, z)`. Integrating Newtonian gravity along an infinite strike direction gives the positive-down vertical response

```text
g_z(x₀, z₀) = 2 G Δρ ∬_A (z - z₀) / ((x - x₀)² + (z - z₀)²) dA
```

where `G = 6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻²` and `A` is the polygon cross-section. Multiplication by `10⁵` converts the SI acceleration to mGal.

For a rectangle centred on the observation's horizontal coordinate, with half-width `a` and depths `z₁` to `z₂` below the observation, integrating first across the width and then with respect to depth yields

```text
g_z = 2 G Δρ [ 2 z atan(a / z) + a ln((z² + a²) / L₀²) ] from z₁ to z₂
```

followed by the `10⁵` mGal conversion. For `a = 100 m`, `z₁ = 200 m`, `z₂ = 400 m`, and `Δρ = 500 kg/m³`, this is **+0.8870240147 mGal**.

## Polygon edge evaluation

Because

```text
2 (z - z₀) / r² = ∂ ln(r²) / ∂z
```

Green's theorem converts the area integral to a boundary integral. For a counterclockwise polygon in the numerical `(x, z)` coordinate plane,

```text
g_z = -G Δρ ∮ ln(r² / L₀²) dx
```

where `r² = (x - x₀)² + (z - z₀)²`. The arbitrary positive reference length `L₀` cancels for a closed polygon because `∮ dx = 0`; the implementation uses a body-scale value to keep the logarithm dimensionless. Reversing polygon order reverses both the signed area and edge direction, so the physical result is unchanged.

For an edge parameterized by distance `u` along its tangent and signed perpendicular offset `q`, an antiderivative of `ln((u² + q²) / L₀²)` is

```text
F(u, q) = u ln((u² + q²) / L₀²) - 2u + 2q atan(u / q)
```

with its continuous limiting value used when `q = 0` or the observation is on the edge. Each edge contributes its horizontal tangent component times `F(u₂, q) - F(u₁, q)`. The signed polygon area supplies the winding correction.

## Finite strike and remaining limitations

KGM-M3 replaces the prior midpoint correction with closed-form face integrals for the actual extruded finite polygonal prism. The equations, positive-down coordinate convention, full magnetic vector projection, legacy mode resolution, independent analytical and numerical references, tolerances, and performance limits are documented in [FINITE_STRIKE_METHOD.md](FINITE_STRIKE_METHOD.md). Finite strike supports asymmetric bounds, including bodies whose two ends lie on the same side of the profile. The project format stores an explicit Infinite Strike or Finite Strike mode; legacy `(0, 0)` bounds without a mode remain the Infinite Strike sentinel.

KGM-M2's infinite-strike magnetic edge-charge method and field-angle conventions are documented in [MAGNETIC_METHOD.md](MAGNETIC_METHOD.md). The desktop and web interfaces now expose field intensity, inclination, declination, profile azimuth, and remanent orientation. The inversion routines were not changed; the earlier review's desktop argument mismatch and optimizer-status issues remain open. Finite magnetic calculations assume uniform magnetization and no self-demagnetization.

## Tolerances and evidence

- The rectangle regression compares the edge kernel to the closed form at `1e-6` relative tolerance.
- A nonrectangular triangle compares the kernel to direct Gaussian quadrature of the area integral at `2e-6` relative tolerance. The source polygon is separated from the observation, so that independent quadrature has no singular point.
- Symmetry, linear superposition, vertical translation, and winding invariance use `1e-12` relative/absolute tolerances.
- These checks establish synthetic mathematical consistency for the 2D gravity reference and finite polygonal-prism field equations. They do not validate field data or inversion. Magnetic validation is recorded in [MAGNETIC_METHOD.md](MAGNETIC_METHOD.md) and [FINITE_STRIKE_METHOD.md](FINITE_STRIKE_METHOD.md).
- `.github/workflows/scientific-tests.yml` installs the locked Python test environment and runs the suite on pushes and pull requests.

## Next milestones

1. **KGM-M4 — inversion:** repair desktop argument binding, report optimizer convergence honestly, add physical bounds and geometry checks, and test recovery on synthetic data.
2. **Later — reproducible data flow:** document input schemas and coordinate metadata, and make generated reports session-safe.
