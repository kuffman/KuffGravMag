# KGM-M4 inversion method and limits

## Shared interface and result semantics

The desktop and web applications call `invert_body_parameters` with the same keyword-only interface. The target body, observation arrays, full body list, field type, and magnetic field settings are explicit, which prevents the former desktop call from binding `"gravity"` to `all_bodies`.

The function returns an `InversionResult`. `success` means the optimizer converged to a finite model. It does not mean the model is unique. `status`, the optimizer message, raw and weighted RMS residuals, evaluation counts, optimality, data-Jacobian rank, condition number, parameter sensitivity, and (when defensible) local standard errors are reported separately. A failed solve restores the starting body properties and geometry.

## Parameters and constrained geometry

Gravity fits the body's density contrast in `kg/m³`; magnetics fits susceptibility `k_succ` in SI. The defaults use bounds of `[-5000, 5000] kg/m³` and `[0, 1]` SI respectively. Callers can replace these through `parameter_bounds`.

Geometry is fixed unless `fit_geometry=True`. The optional geometry terms translate the cross-section in positive-down depth and scale its vertical relief while keeping x-coordinates, vertex order, strike bounds, and polygon topology fixed:

```text
z_i = z_min,reference + depth_offset + depth_scale * (z_i,reference - z_min,reference)
```

`depth_scale` remains positive. Feasible bounds keep every transformed vertex within `depth_bounds` (default 0–20,000 m), so the solve cannot move vertices above the selected depth range or reverse polygon topology. This version does not fit arbitrary individual vertex positions or body width.

## Objective, scaling, and noise

The solver minimizes a bounded trust-region objective with SciPy's `least_squares(method="trf")`:

```text
Phi(q) = ||(F(m(q)) - d) / sigma||² + lambda² ||q - q_ref||²
```

The dimensionless vector `q` scales density, susceptibility, depth shift, and depth scale before optimization. `observation_sigma` accepts one positive standard deviation or one value per observation, in mGal for gravity or nT for magnetics. The API also supports `linear`, `soft_l1`, and `huber` losses for noisy or outlier-prone data. `reference_model` supplies the regularization reference; if omitted, the starting model is used.

## Identifiability and uncertainty

Identifiability is assessed from the weighted **data-only** Jacobian, before regularization. Its numerical rank and condition number describe whether the observations locally constrain each fitted parameter. Regularization can stabilize an underdetermined solve, but it does not make the data identify a unique model.

Local standard errors are estimated only for full-rank linear-loss fits with positive residual degrees of freedom. They use a local linear approximation with the regularization term included in the Hessian. Robust-loss fits and rank-deficient systems do not receive standard errors. These values do not cover structural model error, remanent-field uncertainty, demagnetization, or alternate geological interpretations.

## Validation scope

Synthetic tests recover density and susceptibility from the existing forward kernels, preserve fixed bodies in a multi-body model, and exercise per-observation weights, regularization, optional constrained geometry, noisy observations, invalid geometry, optimizer exhaustion, and underdetermined systems. The earlier 50 gravity, magnetic, finite-strike, and project/report regressions remain unchanged.

The tests establish numerical behavior for the supplied synthetic models. They do not establish field-data accuracy, arbitrary geometry recovery, or uniqueness of potential-field inversion. Interpret converged but non-identifiable results as one fit among multiple plausible models.

## KGM-M5 noise and tradeoff study

KGM-M5 adds a fixed-seed, paired-noise density recovery ensemble at `0.01`,
`0.03`, and `0.1 mGal`, and records parameter bias, RMSE, local standard
errors, and empirical interval coverage. It also constructs a shallow and a
deeper density model that match one station but differ over a full profile.
The experiment settings, tolerances, limitations, and measured results are in
[SCIENTIFIC_VALIDATION.md](SCIENTIFIC_VALIDATION.md) and
[VALIDATION_REPORT.md](VALIDATION_REPORT.md).
