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

## [CIP] lnL overflow + auto-tempering (loud events)

Different driver, different flags: `bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py`.
Line numbers verified against branch `tdlike_paper2` (2026-07); `origin/rift_O4d` is byte-identical
in this file **except** for the `--lnL-protect-overflow` fix below (its own numbers are ~15 lines
lower, e.g. `my_exp` at `:2885`).

CIP passes the sampler `exp(lnL)`, and it AUTO-PICKS the tempering exponent from the peak of the
fitted surface (`:2900` AV branch / `:2902` GMM branch), then hands it over as `tempering_exp`
(`:3009`):

    my_exp = min(1, 0.8*ln(n_step)/max(Y))      # AV branch
    my_exp = min(1, 4.0*ln(n_step)/max(Y))      # GMM branch

With no shift and lnL~3300 at `n_step=1e5` that is `my_exp ~ 0.003` — adaptation effectively OFF.
Both loud-event problems (the `exp()` overflow and the tempering collapse) have the same one-knob
cure: shift the peak down to ~+100.

- **`--lnL-shift-prevent-overflow S`** (`:303`) — the working STATIC lever, and the one to reach for
  first. Set `S ~ lnLmax - 100`. It is subtracted from `Y` before every fit/`exp()`
  (the `Y = Y[indx_ok] - lnL_shift` lines, `:2183`-`:2476`), feeds the GMM `L_cutoff` (`:2947`), and
  is **added back to the output evidence** (`:3080`) — exact and reproducible by design. Verified in
  production 2026-07 on GW250114/U: lifted `my_exp` 0.0153 -> 0.504 (33x). Caveat from its own help
  text: do NOT set S so large that the peak shifted lnL approaches or crosses 0 — the fit relaxes to
  0 and you get garbage. Peak ~ +100 is the intended sweet spot. Works with AV, GMM and portfolio.
- **`--lnL-protect-overflow`** (`:304`) — the DYNAMIC sibling: "subtract lnLmax-100, add it back".
  **It was a silent NO-OP** across the whole O4b/O4c/O4d/master line, and `origin/rift_O4d` is still
  broken as of 2026-07. Fixed on `tdlike_paper2` (`:2144-2147`) by wiring it into the SAME
  `lnL_shift` machinery, combined via `max()` so the two flags coexist and well-behaved data
  (`max_lnL < 100`) stays byte-identical. Check your checkout before relying on it — see
  `gotchas.md`.
- **`--internal-temper-log`** — AV/GMM temper on log-scaled weights; help text: "reduce insane
  contrast and overfitting for high-amplitude cases". Plausible extra guard on a sharp peak, best
  combined with the shift. `(unverified)` — never run by us; suggested from the help text only.
- **Neither flag helps AV CONVERGE.** AV ignores `tempering_exp` entirely (`samplers.md`); for AV
  the shift only removes the overflow. It is GMM/AC that get their adaptation back.
- **`--n-max` is not the fix.** On a loud event the low n_eff is a tempering/overflow problem, not a
  sample-budget problem — raising `--n-max` alone burns cycles without moving n_eff.

## Extrinsic export (see gotchas.md)
- `--save-samples` (-S) — sparse sim_inspiral XML, **lnL only** (not the IS weight). `--save-P`
  (def 0.1! set 0 to keep all), `--save-deltalnL`.
- `--extrinsic-proposal-output PATH` — weight-correct per-group GMM breadcrumb (.npz). The RIGHT
  export for a weighted-posterior / shape check.
- `--fairdraw-extrinsic-output` — resamples ∝ weight; USELESS at low n_eff.
- `--calibration-export-posterior` — full fair-draw + cal nodes (calmarg).
- **`<output>_<indx>_integrator_status.json`** — not a flag, an ARTIFACT (PR #63, junior; merged 2026-08-11 as b4ddb3a0 on rift_O4d). Written
  beside the `.dat`/`.grid`/XML on EVERY run: `collapsed`, `collapse_reason`, lnL/sigma/neff/ntotal,
  plus whichever of `pareto_khat`, `n_ESS`, `n_live_final`, `n_empty_cycles`, `n_warm_seed(_rank)`,
  `n_replicas_pooled`/`_collapsed` apply. **This is the machine-readable integrator verdict** — the
  `.dat` schema is positional (CIP reads it by column) so the status could not be a new column.
  Under replica pooling lnL/sigma are the POOLED values while ESS/k-hat/live-set counts are the
  first run's; `n_replicas_*` and the per-replica `collapse_reason` carry the pooled picture.
- `--reject-collapsed-live-volume` — DROP an event whose live volume degenerated (no .dat/.grid/XML/
  sidecar at all) instead of exporting it. **Off by default deliberately**: dropping thins the
  posterior in an SNR-dependent way, which is the failure the fix removed. Checked twice — before
  replication (fast path) and again on the POOLED verdict, so a collapsed replica cannot slip past.
- `RIFT_AV_TRACE=1` — env var, not a flag. Per-cycle AV contraction trace; the diagnostic for any
  live-volume / contraction failure. See `gotchas.md`.
