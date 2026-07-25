# The samplers, one by one

Frame: all are importance samplers over the EXTRINSIC parameters (RA, dec, distance, inclination,
psi, phi_orb, and time via time-marginalization). Each draws points, evaluates the marginal
likelihood, and forms importance weights `w = L * prior / sampling_prior`. Quality is the effective
sample count `n_eff` (Kish). Intrinsic params are fixed per ILE evaluation (the grid point).

---

## AC — `adaptive_cartesian_gpu` (`mcsamplerGPU`)  [DRIVER DEFAULT]

Classic RIFT sampler: an independent adaptive histogram per sampled axis, tempered by the running
likelihood (`--adapt-weight-exponent`, `--adapt-floor-level`). Product of 1-D marginals — it CANNOT
represent cross-parameter correlation. On GPU via cupy. This is what you get if you pass no
`--sampler-method`. Solid general workhorse at low-to-moderate SNR; struggles when the extrinsic
posterior is strongly correlated/curved (high SNR) because the factorized histogram can't bend.

Key options: `--no-adapt`, `--no-adapt-distance`, `--no-adapt-after-first`, `--force-adapt-all`,
`--adapt-weight-exponent` (tempering exp; 1.0 default, ILE often uses 0.1), `--adapt-floor-level`.

---

## AV — `AV` (`mcsamplerAdaptiveVolume`) — "VARAHA"

A single axis-aligned **hypercube** over the *adaptive* dimensions that **contracts** toward the
support of the integrand. Very efficient and robust once contracted; the go-to for a well-localized
or high-SNR target because it concentrates draws into the shrunken box.

Critical behaviors / lore:
- **It contracts ONLY on a chunk where it is UPDATED** (`update_sampling_prior_selfish`). Before its
  first contraction its per-chunk n_ess is ~1. In a portfolio this made it get frozen at chunk 1 and
  never contract — see the freeze-policy fix (`design-history.md`).
- **Which dims it adapts is `self.adaptive`** (-> `indx_adaptive`, `mcsamplerAdaptiveVolume.py:247`).
  Non-adaptive dims get 1 bin. On a high-SNR event where every extrinsic dim is informative you want
  **`--force-adapt-all`** so the box contracts in all directions; otherwise the backstop is weak.
- **Coverage floor** `--sampler-warmstart-cover-frac` mixes uniform full-box points into the proposal
  (bias-safety when warm-started from a possibly-shifted seed; 0.5 default = safe, 0.05/0 = tight).
- **AV RESETS between `integrate()` calls** — there is no seedable/boundary-shifting AV yet, so
  burn-in / calmarg warm-start gives AV no speedup (correctness-safe only). (See DESIGN_adaptive_driver.)
- Standalone AV can stall at n_eff~1 on the hardest high-SNR best-fit points if run in correlated
  coordinates (use the rotations) or without full adaptation.

---

## GMM — `GMM` (`mcsamplerEnsemble`)

Full-covariance Gaussian mixture, fit each chunk to the importance-weighted elite cloud, **per
extrinsic group**. Default groups (`RIFT/calmarg/extrinsic_handoff.py` STANDARD_GROUPS, mirrored in
the sampler): `(ra,dec)`, `(distance,inclination)`, `(phi_orb,psi)`. A product of per-group GMMs — it
captures WITHIN-group correlation (curved dL-ι arc, sky ring) but not cross-group.

Key options:
- `--internal-gmm-adaptive-components` — choose each group's component count by BIC each chunk
  (flexible allocation), capped by `--internal-gmm-max-components` (default 8).
- `--internal-gmm-sky-components` (default 4, sized for a large sky RING; use 1-2 for a
  well-localized 3+ IFO / high-SNR peak), `--internal-gmm-phase-components`.
- `--internal-gmm-inflate` (covariance std multiplier, >1 widens vs the elite cloud),
  `--internal-gmm-defensive-frac` (broad box-covering component, Hesterberg defensive IS; opt-in,
  did NOT help on the SNR~82 benchmark).
- `--internal-gmm-correlate-all` — a SINGLE full-dimension GMM instead of the factored groups; CAN
  represent cross-group (sky-phase) correlation but needs ~(d+2) eff-samples/component, so on modest
  n_eff it goes near-singular and COLLAPSES more (observed strictly worse on the high-SNR study).

Failure mode: standalone GMM has been observed to NaN on chunk 1 on hard/high-SNR events (S250114ax,
both points). It works reliably only INSIDE the portfolio, where AV's broad density in `q_mix` bounds
the GMM importance weights and a NaN-weight guard stabilizes it.

---

## NF — normalizing flow (`mcsamplerNFlow`)  (unverified depth)

A normalizing-flow proposal; requires torch. Used as a portfolio member. Lore here is thin — fill in
when used. Note the plugin import can pull torch and fail in minimal containers (guarded in the
portfolio loader).

---

## portfolio (`mcsamplerPortfolio`)

A **mixture** of member samplers with the balance-heuristic density `q_mix = Σ_m frac_m * q_m`. Each
chunk it draws from members in proportion to `frac_m` and reweights by the true `q_mix`.

Why it matters:
- **Unbiased for ANY member weights** — because every sample is weighted by the full `q_mix`, a
  member can only ever cost draws, never bias lnZ. This is the theoretical basis for making AV
  freeze-exempt and for adaptive draw allocation.
- **Designed to STABILIZE**: AV as a robust backstop that prevents a GMM runaway/collapse; GMM to
  catch fine correlated structure at high SNR where the AV box is a crude fit. If the portfolio still
  behaves like a seed lottery, suspect the members are mis-configured (AV under-adapted, correlated
  coordinates) rather than the mixture being fundamentally unstable.
- Members: `--sampler-method portfolio --sampler-portfolio AV --sampler-portfolio GMM` (repeat).
  Inside the portfolio, member names dispatch at driver ~1227-1236 (`AV`,`GMM`,
  `adaptive_cartesian_gpu`, else `known_pipelines` incl. nflow).

Policies (all in `DESIGN_portfolio_freeze_policy.md`): VARAHA never-freeze (default on), plateau
revive (default OFF, halved n_eff in the gate), adaptive draw allocation (opt-in, quality signals
global/credit/ness), weight clipping (opt-in, adaptation-fit-input ONLY — never the estimator).
