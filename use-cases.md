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
