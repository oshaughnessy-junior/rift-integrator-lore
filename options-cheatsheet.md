# Options cheat-sheet (the flags that actually change behavior)

Driver: `integrate_likelihood_extrinsic_batchmode`. Line numbers vs branch
`rift_O4d_portfolio_freeze_tuning` (verify after rebases).

## Sampler selection
- `--sampler-method {adaptive_cartesian_gpu | AV | GMM | portfolio}` — default `adaptive_cartesian_gpu`.
- `--sampler-portfolio NAME` — REPEAT per member (`AV`,`GMM`,`adaptive_cartesian_gpu`,`NF`/nflow).
  action='append'. Comma form `AV,GMM` also split (recent).
- `--sampler-portfolio-args '<dict>'` — per-member kwargs (append, aligned to members).

## Adaptation
- `--no-adapt`, `--no-adapt-distance`, `--no-adapt-after-first`
- `--force-adapt-all` — adapt ALL params (use for high SNR / AV backstop).
- `--force-reset-all` — reset sampling each iteration.
- `--adapt-weight-exponent X` (tempering; ILE often 0.1, default 1.0), `--adapt-floor-level`.
  NOTE (AC): a zero histogram floor lets bins hit exactly-zero probability = absorbing =
  support-truncation lnZ bias; `ff0a04ba` (rift_O4d_mc_error_stabilization) clamps the AC hist
  floor to >=1e-2. See gotchas.md / design-history.md.

## Coordinates (see coordinates-and-degeneracies.md)
- `--inclination-cosine-sampler`, `--declination-cosine-sampler`
- `--internal-rotate-phase`, `--internal-sky-network-coordinates`[`-raw`]

## GMM (mcsamplerEnsemble)
- `--internal-gmm-adaptive-components`, `--internal-gmm-max-components N` (def 8)
- `--internal-gmm-sky-components N` (def 4), `--internal-gmm-phase-components N`
- `--internal-gmm-inflate F` (def 1.0), `--internal-gmm-defensive-frac F` (def 0), `--internal-gmm-correlate-all`

## Portfolio policy (mcsamplerPortfolio; see DESIGN_portfolio_freeze_policy.md)
- `--portfolio-varaha-never-freeze` (default on) / `--portfolio-varaha-can-freeze`
- `--portfolio-grace-iters N`, `--portfolio-revive-period N`, `--portfolio-freeze-wt X`
- `--portfolio-adaptive-alloc`, `--portfolio-quality-signal {global|credit|ness}`
- `--portfolio-weight-clip C`, `--portfolio-varaha-min-frac X`

## Likelihood path (see option-combos.md -- these only work in combination)
- `--vectorized --gpu --force-xpy` — the maintained NoLoop likelihood. `--force-xpy` is INERT without `--gpu`.
- `--interpolate-time True` — CUBIC Q_lm interpolation at fractional detector times instead of
  nearest sample bin. Requires the NoLoop combo above. Takes a truthy VALUE, not a bare switch.
  Removes a spurious extrinsic non-smoothness (time quantization) -> more robust convergence. USE IT.
- `--internal-use-lnL` — integrate lnL. Forced ON for the portfolio (which requires it).

## Budget / convergence
- `--n-max`, `--n-eff`, `--n-chunk`, `--vectorized --gpu`, `--force-xpy` (inert without `--gpu`)

## Extrinsic export (see gotchas.md)
- `--save-samples` (-S) — sparse sim_inspiral XML, **lnL only** (not the IS weight). `--save-P`
  (def 0.1! set 0 to keep all), `--save-deltalnL`.
- `--extrinsic-proposal-output PATH` — weight-correct per-group GMM breadcrumb (.npz). The RIGHT
  export for a weighted-posterior / shape check.
- `--fairdraw-extrinsic-output` — resamples ∝ weight; USELESS at low n_eff.
- `--calibration-export-posterior` — full fair-draw + cal nodes (calmarg).
