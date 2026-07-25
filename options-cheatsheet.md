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

## Budget / convergence
- `--n-max`, `--n-eff`, `--n-chunk`, `--vectorized --gpu`, `--force-xpy` (inert without `--gpu`)

## Extrinsic export (see gotchas.md)
- `--save-samples` (-S) — sparse sim_inspiral XML, **lnL only** (not the IS weight). `--save-P`
  (def 0.1! set 0 to keep all), `--save-deltalnL`.
- `--extrinsic-proposal-output PATH` — weight-correct per-group GMM breadcrumb (.npz). The RIGHT
  export for a weighted-posterior / shape check.
- `--fairdraw-extrinsic-output` — resamples ∝ weight; USELESS at low n_eff.
- `--calibration-export-posterior` — full fair-draw + cal nodes (calmarg).
