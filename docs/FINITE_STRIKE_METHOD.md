# KGM-M3 finite-strike polygonal prisms

## Scope and model

KGM-M3 replaces the per-edge midpoint strike multiplier with the field of the
actual three-dimensional body formed by extruding a 2D polygon between strike
coordinates `y_min` and `y_max`. The finite model evaluates closed-form face
integrals for the gravity and uniform-magnetization fields. The production
calculation does not approximate strike length with a midpoint factor or a
numerical volume grid.

Two body modes are explicit:

- **Infinite Strike** uses the validated 2D polygon kernels from KGM-M1 and
  KGM-M2. The saved strike bounds are ignored by the forward calculation.
- **Finite Strike** requires finite, ordered `y_min ≤ y_max` bounds. The
  profile lies at `y=0`; bounds may be asymmetric, both on one side of the
  profile, or otherwise offset. Equal bounds describe zero volume and return
  zero anomalies.

New projects save the mode independently from the bounds. For old project data
that has no mode, `(y_min, y_max) = (0, 0)` is interpreted as the historical
Infinite Strike sentinel; other finite bounds are interpreted as Finite Strike.
Once loaded and saved, the project records the resolved mode explicitly, so a
physical zero-width Finite Strike body remains unambiguous.

The cross-section is an arbitrary simple, non-self-intersecting polygon in the
`(x,z)` plane. It is extruded uniformly along `y`. Density contrast and
magnetization are constant within each body. The observation profile is at
`y=0`; observations on a finite magnetic prism edge or vertex are singular and
raise `ValueError`. Magnetic self-demagnetization, induced anisotropy,
magnetization variation within a body, and topographic clipping are not
modelled.

## Coordinates, field direction, and units

Coordinates form a right-handed local frame: `x` increases along the profile,
`y` is the perpendicular strike direction, and `z` is positive downward.
Profile azimuth `alpha` and field/remanent declinations are clockwise from
geographic north. Inclination is positive downward. With `delta = D_f -
alpha`, the unit ambient-field vector in profile coordinates is

```text
n_f = (cos(I_f) cos(delta), cos(I_f) sin(delta), sin(I_f))
```

The field magnitude `F_nT` is nT and `mu0 = 4 pi × 10^-7 T m/A`. Uniform
magnetization in A/m is

```text
H0 = F_nT × 10^-9 / mu0                         [A/m]
M = chi H0 n_f + Jr n_r                         [A/m]
n_r = (cos(I_r) cos(D_r-alpha),
       cos(I_r) sin(D_r-alpha), sin(I_r))
```

The magnetic output is `dX`, strike-parallel `dY`, positive-down `dZ`, and the
first-order total-field anomaly

```text
dT = dX cos(I_f) cos(delta)
   + dY cos(I_f) sin(delta)
   + dZ sin(I_f)                                  [nT]
```

`dT` is the linearized projection `n_f · dB`, not the exact norm difference
`|B0+dB|-|B0|`. The legacy `calculate_magnetics(...)` method remains available
and returns `(dX, dZ, dT)` for inversion and older callers. The new
`calculate_magnetics_components(...)` method returns `(dX, dY, dZ, dT)`.

## Finite-strike gravity derivation

For a body of uniform density contrast `Delta rho`, observation `r_o`, and
source point `r`, Newtonian attraction in positive-down coordinates is

```text
g_z(r_o) = G Delta rho ∫_V (z - z_o) / |r_o-r|^3 dV
```

By the divergence theorem, the `z` component is a sum over the planar prism
faces `f`:

```text
g_z(r_o) = -G Delta rho Σ_f n_f,z J_f
J_f       = ∫_f 1/|r_o-r| dS
```

Here `n_f` is the outward unit normal. For each face, project the observation
onto its plane and express a point on the face as `u`. With signed normal
distance `h`, `R = sqrt(|u|^2+h^2)`. A two-dimensional divergence identity is

```text
div_u [ u / (R + |h|) ] = 1/R
```

so `J_f` becomes a boundary integral. For an oriented face edge, let `t` be its
unit tangent, `m` its outward in-face unit normal, `q` the signed perpendicular
distance from the projected observation to the edge, and `s` the coordinate
along the edge. The edge contribution is `P(s2)-P(s1)`, where

```text
A = sqrt(q^2 + h^2),  b = |h|
P(s) = q asinh(s/A)
       - 2 sign(q) b atan( [|q|/(A+b)] s/[sqrt(s^2+A^2)+A] )
```

An edge with `q=0` contributes zero to this particular boundary field. The
face contributions sum to `J_f` in metres; multiplying by `G Delta rho`
produces m/s² and `10^5` converts to mGal.

## Finite-strike magnetic derivation

For uniform magnetization `M`, there is no volume magnetic charge. Each face
has magnetic pole density `sigma_f = M · n_f`. Outside the prism, the magnetic
flux-density anomaly in tesla is

```text
dB(r_o) = mu0/(4 pi) Σ_f sigma_f I_f
I_f      = ∫_f (r_o-r)/|r_o-r|^3 dS
```

Decompose `R = r_o-r` into its tangential and normal parts at the face. The
in-face vector integral reduces by the planar divergence theorem to edge line
integrals. With the same `t`, `m`, `q`, `h`, and `s` definitions as above,

```text
I_f = Σ_edges m [asinh(s2/sqrt(q^2+h^2))
                 - asinh(s1/sqrt(q^2+h^2))] - Omega_f n_f
Omega_f = ∫_f n_f · (r-r_o)/|r-r_o|^3 dS
```

`Omega_f` is the oriented solid angle of the polygon. The implementation sums
the stable triangle solid-angle `atan2` formula over a signed fan, so clockwise
or counter-clockwise input polygon winding produces the same outward prism
faces. Multiplication by `mu0/(4 pi)` and `10^9` converts tesla to nT. The
full vector is projected onto `n_f` for `dT`.

For infinite strike, the strike-parallel anomaly is identically zero and the
existing 2D boundary-charge calculation is used. For finite strike, all three
components are calculated. The asymmetric finite-y prism does not assume equal
strike extents on the two sides of the profile.

## Independent references and reproduced midpoint error

The finite test prism has cross-section `x=[-100,100] m`, `z=[200,400] m` and
strike bounds `y=[-300,500] m`, with observations on `y=0`. This is a genuinely
asymmetric, off-centre strike interval. At `(x,z)=(0,0) m` and density contrast
`500 kg/m³`, the old midpoint gravity result was **0.60848391 mGal**. The
independent 3D Gaussian volume integral and the rectangular-prism corner sum
both give **0.6990738981 mGal**; the midpoint model is low by **0.09058999
mGal (12.96%)**.

For a remanent magnetization `M=(0.8,-0.35,0.42) A/m`, the independent
32-point-per-axis volume integration gives `(dX,dY,dZ) = (-54.96475481,
12.92947538, 36.49399496) nT` at `(0,0,0) m`. The old midpoint result gave
`dX=-54.77665896 nT` and `dZ=39.33275916 nT`, and did not return `dY`; its
vertical component differs by **2.83876420 nT (7.78%)**. These values reproduce
the error caused by applying one midpoint factor per cross-section edge to an
asymmetric finite body.

Gravity is additionally checked at an off-axis observation against the exact
Nagy rectangular-prism corner sum and an independent 32-point-per-axis tensor
Gaussian volume integral. For the asymmetric box at the profile origin, the
closed-form engine returns `0.699073898114271 mGal`, the corner sum returns
`0.699073898114276 mGal`, and the independent volume integral returns
`0.699073898114272 mGal`. At the origin with the stated oblique magnetization,
the face-integral components are `(-54.964754812958, 12.929475380737,
36.493994956513) nT`; the independent volume values differ by at most
`7.4e-12 nT`. Additional checks cover symmetric and asymmetric bounds,
both polygon windings, oblique profiles, equatorial/low inclination, symmetry,
superposition, susceptibility scaling, zero magnetization, physical zero
width, legacy sentinel loading, and large-strike convergence to the infinite
2D kernels.

## Tolerances, performance, and limitations

Analytical gravity prism checks use relative tolerance `2e-11` and absolute
tolerance `2e-11 mGal`. The independent direct magnetic volume oracle uses
32-point Gaussian integration per Cartesian axis and component tolerances
`3e-8 nT` relative/absolute. Winding and symmetry comparisons use `1e-12`;
finite-to-infinite convergence at `±1,000,000 m` uses `1e-7` relative and
`1e-9 mGal` absolute tolerance for gravity, and `1e-7` relative / `2e-5 nT`
absolute tolerance for magnetics. Observed maximum component residuals at that
strike length are `4.8e-6 nT` (`dX`), `9.5e-6 nT` (`dY`), `3.3e-6 nT`
(`dZ`), and `1.8e-6 nT` (`dT`). These are regression tolerances, not claims
about field-data
accuracy.

The production solver evaluates each body face for each observation. Cost is
approximately proportional to `N_observations × N_faces × N_face_edges` and
uses only NumPy and the Python standard library. In the local locked
environment, a 250-observation profile with two finite bodies took about
`0.24 s` for gravity and `0.14 s` for magnetics; this is a single-machine
smoke benchmark, not a portable timing guarantee. Large profile sets or
highly vertex-dense polygons may require a future compiled/vectorized
implementation.
Exact face formulas can be ill-conditioned at a magnetic edge or vertex; those
locations are reported as singular. Gravity remains finite at polygon edges.
Finite magnetics assumes uniform magnetization and no demagnetization; no
inversion changes, exact topography, or spatially varying strike cross-section
are included in KGM-M3.

## References

- Cady, J. W. (1977). [Calculation of gravity and magnetic anomalies along
  profiles with end corrections and inverse solutions for density and
  magnetization](https://pubs.usgs.gov/of/1977/0463/report.pdf), USGS Open-File
  Report 77-463. Cady develops finite-strike end corrections for profile-based
  gravity and magnetic calculations and defines the positive-down profile
  coordinates used here.
- Nagy, D. (1966). [The gravitational attraction of a right rectangular
  prism](https://doi.org/10.1190/1.1439779). *Geophysics*, 31(2), 362–371.
  The closed rectangular-prism corner sum is the independent gravity oracle.
- Plouff, D. (1976). [Gravity and magnetic fields of polygonal prisms and
  application to magnetic terrain corrections](https://doi.org/10.1190/1.1440645).
  *Geophysics*, 41(4), 727–741. Develops exact fields for polygonal prisms.
- Kravchinsky, V. A., Hnatyshin, D., Lysak, B., & Alemie, W. (2019).
  [Computation of magnetic anomalies caused by two-dimensional structures of
  arbitrary shape](https://doi.org/10.1029/2019GL082767). Defines induced and
  remanent field-angle conventions and the first-order total-field projection.
