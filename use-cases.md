# Use-case decision guide

Given an event/point, which sampler + config. Heuristics from the O4d portfolio study; verify per case.

## By SNR / posterior sharpness

- **Low–moderate SNR, broad extrinsic posterior:** default `adaptive_cartesian_gpu` (AC) is fine and
  cheap. Or `AV` for a bit more concentration.
- **High SNR, sharp/correlated posterior (e.g. the best-fit point of a loud event):** this is the
  hard regime.
  - Use DECORRELATED coordinates + FULL adaptation: `--force-adapt-all --internal-rotate-phase
    --internal-sky-network-coordinates`.
  - Prefer `AV` (contracts to the sharp support) as the backbone; the `portfolio AV+GMM` adds GMM to
    catch curved within-group structure while AV backstops runaway.
  - Expect n_eff to be a LOTTERY per single run — plan to POOL many copies (below).
- **Strongly-correlated target where a factored proposal stalls:** the portfolio (correlated members)
  can beat AC/AV. NOTE `--internal-gmm-correlate-all` (single full-dim GMM) is usually WORSE, not
  better — it needs too many eff-samples/component. Fix the coordinates instead.

## Standalone vs portfolio

- **Standalone GMM:** avoid on hard/high-SNR events — observed to NaN on chunk 1. Use GMM only inside
  the portfolio.
- **Standalone AV:** great when well-localized; can stall at n_eff~1 in correlated coords / partial
  adaptation.
- **Portfolio AV+GMM:** the intended stabilizer for high SNR (AV backstop + GMM detail). Select
  members with repeated `--sampler-portfolio` flags. It replicates standalone-AV lnZ unbiased on
  typical events; on an AV-optimal event it costs a modest tail slowdown for carrying GMM.

## When n_eff is a lottery (pool copies)

On a hard point, a single run — any config — is not a posterior. Recipe:
1. Run ~2-3x as many independent copies as the number of "landed" (n_eff >= ~15) copies you need.
2. Pool WEIGHTED BY RELIABILITY (n_eff), or simply drop n_eff<5 copies. Naive equal-weight pooling is
   biased by the collapsed copies.
3. Diagnostic: the reliability-weighted "effective #copies" (Kish over the copies' n_eff) tells you
   how many the pool really rests on.
Harness: `MonteCarloMarginalizeCode/Code/test/integrators/bench_onsource_ensemble.sh` +
`compare_extrinsic_breadcrumbs.py` (reads `--extrinsic-proposal-output` breadcrumbs).

## Sanity: is the config even running what you think?

`grep "PORTFOLIO setup"` (portfolio) or `grep "ILE: "` in the ILE log to confirm the sampler. A run
that finishes in ~60s with no `.dat` is usually the wrong sampler / a config that bailed.

## VALIDATED high-SNR recipe (the n_eff lottery fix)

On a very sharp high-SNR peak, AV contracts onto the peak or the wrong spot ~50/50, so a large
fraction of independent runs collapse to n_eff~1 (a bimodal LOTTERY, robust across every proposal/cap
tried). Two levers, in order of impact:

1. **`--sampler-warmstart-retry-neff 5` (L0 auto-rescue) -- the big one.** If a pass finishes below
   the threshold, re-seed AV from the run's OWN highest-L samples (the peak it did find) and re-run.
   Same-problem reuse, cannot bias, frame-safe. On the SNR~80 study this took the landed fraction
   from 4/9 to **8/9** (chronic n_eff~1 collapsers landed at 25-33). Cost: a rescued run does 2
   integration passes. Works for standalone AV or the AV+GMM portfolio.
2. **`--force-adapt-all --internal-rotate-phase` (frame-matched seed).** Improves the LANDERS
   (n_eff ~50 vs ~30, fuller sky-ring exploration) but does NOT change the collapse rate. Use with a
   phase-rotated warm seed (see coordinates-and-degeneracies.md) or it poisons the proposal.

Full recipe: portfolio AV+GMM (cap8, `--internal-gmm-adaptive-components`) + `--force-adapt-all`
+ `--internal-rotate-phase` (phase-frame-matched warm seed) + `--sampler-warmstart-retry-neff 5`.
Even then, at modest n_eff, pool a few landed copies for a publication-grade posterior.

## Methodology note: chase a ROBUST setup, not a characterization of bad ones

When an integrator setup shows instability (e.g. a bimodal n_eff lottery), the useful move is to find
a configuration that is ROBUST and then **backtest it down to lower SNR to confirm it is still
useful** there -- not to spend the budget characterizing exactly how a bad setup fails. Performance
of a bad setup is worth a bounded look, no more.

Always include `--interpolate-time True` (with the NoLoop combo) in any convergence study: without it
the extrinsic surface carries a time-quantization non-smoothness that is a discretization artifact,
and you will be tuning the sampler against an artifact.
