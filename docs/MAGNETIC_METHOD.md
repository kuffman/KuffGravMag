# KGM-M2 magnetic method and validation

## Scope

This note documents the corrected uniform-magnetization magnetic kernel for a polygonal cross-section that is invariant along strike. It returns the profile-horizontal component `dX`, the positive-down vertical component `dZ`, and the first-order total-field anomaly `dT`, all in nT. The magnetic kernel is independent of the gravity density contrast.

The validated calculation is the infinite-strike 2D model. The existing finite-strike midpoint multiplier is retained as an approximation and is not validated by these tests. This milestone does not change the inversion engine, add demagnetization, or validate field-data processing.

## Coordinates, angles, and units

- Geographic north, east, and vertical-down are the reference axes.
- Declination `D` is clockwise from geographic north; inclination `I` is positive below the horizontal. These are the conventions used by Kravchinsky et al. (2019).
- Profile azimuth `alpha` is clockwise from north toward the direction of increasing profile distance `x`. The cross-section depth coordinate `z` is positive down. Strike is the perpendicular horizontal direction `y`.
- Susceptibility `k_succ` is dimensionless SI. Ambient field strength `F_nT` is nT. Remanent intensity `J_r` and magnetization components are A/m.
- A body has uniform susceptibility and uniform remanent magnetization. Magnetic self-demagnetization and spatial variation in magnetization are not modelled.

For a unit vector with inclination `I` and declination `D`, its profile-plane components are

```text
u_x = cos(I) cos(D - alpha)
u_z = sin(I)
```

The omitted strike component is `cos(I) sin(D - alpha)`. A uniform infinite-strike body has no boundary magnetic charge from that parallel component, so it does not enter this 2D response.

The induced magnetization is `M_ind = k_succ H0 u_f`, where

```text
H0 = F_nT * 1e-9 / mu0       [A/m]
mu0 = 4 pi * 1e-7            [T m/A]
```

Remanence uses the same angle convention. The total profile-plane magnetization is therefore

```text
Mx = k_succ H0 cos(I_f) cos(D_f - alpha)
     + J_r cos(I_r) cos(D_r - alpha)
Mz = k_succ H0 sin(I_f) + J_r sin(I_r)
```

This vector addition handles low inclination, equatorial, oblique-profile, and opposing induced/remanent directions without calculating a resultant angle or dividing by a horizontal-field projection.

## Boundary integral and winding correction

For uniform `M = (Mx, Mz)`, the volume magnetic pole density is zero inside the polygon. The boundary pole density is `sigma_m = M dot n`, where `n` is the outward unit normal. Integrating the 3D dipole field along an infinite strike direction gives the profile-plane magnetic flux-density anomaly

```text
DeltaB = mu0 / (2 pi) * integral_boundary [ (M dot n) R / |R|^2 ] ds
R = observation - source
```

For each edge, let `t = (ux, uz)` be its unit tangent, `n_left = (-uz, ux)`, `L` its length, and `sA` the sign of the polygon's signed area (`+1` for positive winding and `-1` for negative winding). Then

```text
u1 = R1 dot t
u2 = u1 - L
q  = R1 dot n_left
integral R/|R|^2 ds = T t + N n_left
T = 0.5 ln((u1^2 + q^2) / (u2^2 + q^2))
N = atan2(sign(q) u1, |q|) - atan2(sign(q) u2, |q|), q != 0
N = 0, q = 0 (principal-value mean on an edge)
sigma_m ds = sA (Mx uz - Mz ux)
```

The implementation sums `sA (Mx uz - Mz ux) (T t + N n_left)` over the closed boundary and multiplies by `mu0/(2 pi) * 1e9` to return nT. The signed-area factor changes the outward normal with input winding, so reversing the vertex order leaves the physical field unchanged. A field evaluation at a polygon vertex is singular and raises `ValueError`; repeated adjacent vertices are cleaned and self-intersecting or zero-area polygons are rejected. At an observation lying in an edge interior, `q = 0` uses the principal-value mean for the normal component.

As an independent reference, the volume integral of the 3D dipole field can itself be integrated over the infinite strike coordinate analytically. For `R = (Rx, Rz)` in the cross-section,

```text
DeltaB = mu0 / (2 pi) * integral_area [
    2 (M dot R) R / |R|^4 - M / |R|^2
] dA
```

The triangle regression evaluates this area integral by tensor Gaussian quadrature after a Duffy mapping of each triangle. It does not use the boundary-edge implementation.

## Analytical rectangle reference

For a rectangle `[-a, a] x [z1, z2]` and a centered observation at `(0, 0)`, symmetry gives

```text
C = mu0/pi * [atan(a/z1) - atan(a/z2)]
dX = -C Mx
dZ = +C Mz
```

For `a = 100 m`, `z1 = 200 m`, and `z2 = 400 m`, `C = 87.4675783495768 nT per A/m`. Thus `Mx = 1 A/m, Mz = 0` gives `dX = -87.4675783496 nT`, while `Mx = 0, Mz = 1 A/m` gives `dZ = +87.4675783496 nT` under the positive-down convention.

At the reviewed baseline, the old horizontal-magnetization kernel returned `+87.4675783496 nT` for the rectangle's listed vertex order and the opposite sign when the order was reversed. The new result matches the analytical value for both windings. The old projection also used `cos(atan(tan(I)/cos(D-alpha))) cos(D-alpha)`. At `I = 60 deg` and `D-alpha = 60 deg`, that factor is `0.138675` instead of the correct `cos(I) cos(D-alpha) = 0.25`, a 44.5% underestimate of the profile-horizontal projection. In a separate direct projection regression at `(x,z) = (15,0) m` for `M = (0.8,-0.35) A/m`, the old `dT` was `30.5353 nT`; projecting its own `dX,dZ` components onto the declared ambient-field direction gives `38.6010 nT`.

## Total-field anomaly

The exact total-field change is

```text
DeltaT_exact = |B0 + DeltaB| - |B0|
```

For anomalies small relative to the ambient field, its first-order approximation is the dot product with the unit ambient-field direction. In profile coordinates the returned value is

```text
dT = dX cos(I_f) cos(D_f - alpha) + dZ sin(I_f)
```

This is a linearized total-field anomaly, not the exact norm difference. It matches the standard projection of the horizontal and vertical anomaly components onto the ambient field direction. The weak-anomaly approximation can differ from the exact norm difference at second order in `|DeltaB|/F`.

## Validation evidence and tolerances

`tests/test_magnetic_kernel.py` contains the following checks:

- Horizontal and vertical centered-rectangle analytical values, including both polygon windings: relative and absolute tolerance `1e-10` nT.
- A nonrectangular triangle compared to independent area integration of the 3D dipole field after strike integration: 72-point Gaussian quadrature per mapped coordinate; `rtol=2e-7`, `atol=2e-7` nT.
- Induced magnetization and `dT` for horizontal/equatorial, low-inclination off-profile, oblique-profile, and near-vertical field cases: `rtol=1e-10`, `atol=1e-10` nT.
- Remanent declination to profile-azimuth rotation, direct oblique total-field projection, horizontal symmetry for horizontal and vertical magnetization, multi-body superposition, susceptibility scaling, and zero magnetization.
- Invalid self-intersecting geometry, nonfinite observations, and a singular polygon-vertex observation.

The 16-case magnetic suite was run against the reviewed pre-fix kernel before implementation: **12 failed and 4 passed**, exposing the polarity/winding rectangle, both analytical components, remanent azimuth, the independent triangle, all five field-angle cases, the direct total-field projection, invalid geometry, and the singular vertex case. After correction, all 16 magnetic regression cases pass. The full gravity and magnetic suite is reported in `docs/BASELINE_AND_VALIDATION.md` after the final locked run.

## References

- Kravchinsky, V. A., Hnatyshin, D., Lysak, B., & Alemie, W. (2019). [Computation of Magnetic Anomalies Caused by Two-Dimensional Structures of Arbitrary Shape: Derivation and Matlab Implementation](https://doi.org/10.1029/2019GL082767). *Geophysical Research Letters*, 46. The paper defines susceptibility, inclination/declination, induced and remanent magnetization conventions, derives 2D infinite-strike anomalies, and projects horizontal/vertical components onto the ambient-field direction. [Author-hosted full text](https://sites.ualberta.ca/~vadim/Publications-Kravchinsky_files/2019-Kravchinsky%20et%20al%20-%20Computation%20of%20magnetic%20anomalies.pdf).
- The edge-charge and triangle-area formulas above are independently derived from the magnetic scalar-potential/dipole field and the infinite-strike integral; they provide the analytical and numerical test oracles rather than reusing the application's former edge equations.

## Application limitations

The current desktop and web interfaces still pass `D_f = 0` and `alpha = 0` to the magnetic model; the remanent declination parameter is also not exposed as an editable control. The engine now accepts these values and has nonzero-angle regression coverage, but users cannot yet configure the full orientation model through the interface. Finite-strike magnetic outputs still apply the old per-edge midpoint approximation. The gravity interface separately labels finite-strike gravity as approximate and the 2D infinite-strike reference as benchmarked.
