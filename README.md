# RIFT Integrator Lore

Working knowledge of RIFT's Monte-Carlo **integrators** (the extrinsic samplers used by
`integrate_likelihood_extrinsic_batchmode` / ILE): what each one is, which options matter, which
sampler/coordinates to use for which event, the design choices behind them, and the traps.

**Why this exists (and why it is NOT in the code repo):** this is cross-cutting, slowly-changing
lore about *design intent and use* — it should survive code-branch churn and be transferable across
clones/machines/agents. It is deliberately separate from `research-projects-RIT`. Deep,
code-versioned design notes live *with* the code as `RIFT/integrators/DESIGN_*.md` and
`RIFT/calmarg/DESIGN_*.md`; this repo links to them rather than duplicating.

Maintained by Richard O'Shaughnessy's group + Claude agents. Correct anything you verify; mark
unverified claims `(unverified)`. Cite `file:line` against a named branch when you can.

## The sampler family (name -> class -> file)

Plugin registry (`MonteCarloMarginalizeCode/Code/setup.py:74-77`, group `RIFT.integrator_plugins`):

| plugin | `--sampler-method` string | class | file | one-liner |
|--------|---------------------------|-------|------|-----------|
| **AC** | `adaptive_cartesian_gpu` | `mcsamplerGPU.MCSampler` | `mcsamplerGPU.py` | DEFAULT. Classic per-axis adaptive-Cartesian histogram, on GPU. |
| **AV** | `AV` | `mcsamplerAdaptiveVolume.MCSampler` | `mcsamplerAdaptiveVolume.py` | VARAHA: a contracting axis-aligned hypercube over the adaptive dims. |
| **GMM** | `GMM` | `mcsamplerEnsemble.MCSampler` | `mcsamplerEnsemble.py` | Full-covariance Gaussian mixture per extrinsic group. |
| **NF** | (portfolio member only) | `mcsamplerNFlow.MCSampler` | `mcsamplerNFlow.py` | Normalizing-flow proposal (needs torch). |
| — | `portfolio` | `mcsamplerPortfolio.MCSampler` | `mcsamplerPortfolio.py` | Balance-heuristic MIXTURE of the above members. |

Also present: `mcsampler.py` (classic CPU base), `mcsampler_generic.py` (shared base). Driver
dispatch: `integrate_likelihood_extrinsic_batchmode` ~lines 1169-1244.

## GOLDEN RULES (read before running)

1. **`adaptive_cartesian_gpu` (the driver DEFAULT) is NOT AV and NOT the portfolio.** It is the
   classic Cartesian GPU sampler (`mcsamplerGPU`). To get AV or the portfolio you MUST pass
   `--sampler-method` explicitly. Do not call `adaptive_cartesian_gpu` "AV".
2. **Select portfolio members with REPEATED flags:** `--sampler-method portfolio --sampler-portfolio AV
   --sampler-portfolio GMM`. `--sampler-portfolio` is `action='append'`; the comma form `AV,GMM` is a
   recent convenience split, not the canonical form.
3. **High-SNR / sharp extrinsic posterior:** run in DECORRELATED coordinates with FULL adaptation —
   `--force-adapt-all --internal-rotate-phase --internal-sky-network-coordinates`. Omitting these
   leaves the sky-ring and phase-polarization degeneracies correlated and turns GMM fitting into a
   seed lottery. See `coordinates-and-degeneracies.md`.
4. **n_eff is GPU-non-deterministic even at fixed `--seed`** (float reduction order). Never judge
   reliability from <~8 draws. See `gotchas.md`.
5. **Valid option COMBINATIONS are the user's burden BY DESIGN.** Many RIFT options are de facto
   defaults meant to be used together, but the wiring that would enforce that was never added (it
   would be spaghetti after a decade of accretion). An incomplete combo usually degrades silently
   rather than failing. Read `option-combos.md` before composing a command line. Two you almost
   always want: `--vectorized --gpu --force-xpy` (the maintained NoLoop likelihood) and
   `--interpolate-time True` (cubic Q_lm time interpolation; needs NoLoop; removes a spurious
   extrinsic non-smoothness and makes convergence more robust).
6. **The portfolio estimate is unbiased for ANY member weights** (balance-heuristic `q_mix`), so a
   member can never bias lnZ — only cost draws. This is why AV can be freeze-exempt. See `samplers.md`.

## Contents

- `samplers.md` — per-sampler: what it is, key options, when to use, failure modes.
- `coordinates-and-degeneracies.md` — extrinsic coordinate transforms + the degeneracies they tame.
- `option-combos.md` — **valid option COMBINATIONS** (NoLoop, cubic time interp, lnL/return_lnI). Read first.
- `options-cheatsheet.md` — the CLI flags that actually change behavior, grouped.
- `use-cases.md` — decision guide: given an event, which sampler + config.
- `gotchas.md` — the traps, with the symptom that reveals each.
- `design-history.md` — why the integrators are built the way they are.

## In-code deep dives (canonical, versioned with the code)

- `RIFT/integrators/DESIGN_portfolio_freeze_policy.md` — portfolio freeze policy, adaptive draw
  allocation, weight clipping, the S250114ax high-SNR study, the n_eff-lottery / pool-copies result.
- `RIFT/calmarg/DESIGN_extrinsic_handoff.md` — extrinsic-proposal breadcrumb (GMM handoff).
- `RIFT/calmarg/DESIGN_adaptive_driver.md`, `DESIGN_calmarg_in_loop.md` — calibration marginalization.
