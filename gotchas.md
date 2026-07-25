# Gotchas (symptom -> cause)

- **`adaptive_cartesian_gpu` is NOT AV / NOT the portfolio.** It is the DEFAULT and it is
  `mcsamplerGPU`. Symptom: you passed `--internal-gmm-*` / expected AV but the log says
  `ILE: adaptive_cartesian_gpu` and the run bails fast with no `.dat`. Fix: pass `--sampler-method`
  explicitly.
- **`--sampler-portfolio AV,GMM` silently became one bogus member** (pre comma-split). Use REPEATED
  flags `--sampler-portfolio AV --sampler-portfolio GMM`.
- **n_eff is GPU-non-deterministic at fixed `--seed`** (float reduction order). Symptom: same seed,
  different n_eff (e.g. 7.0 vs 1.0). Take >=8 draws before concluding anything about reliability.
  Beware 3-sample "it's stable" survivorship bias.
- **Low n_eff on a high-SNR point == extrinsic MODE COLLAPSE**, not just noise: the recovered
  extrinsic posterior collapses to a single degenerate blob (1 GMM mode/group; lost sky ring / dL-i
  arc / phase-pol). Landed runs show 3-4 modes/group. Mode count (or n_eff) detects it. The fix is
  coordinates + adaptation + pooling, not a bigger GMM cap.
- **Standalone GMM NaNs on chunk 1** on hard/high-SNR events. Use it inside the portfolio.
- **`--save-samples` XML carries lnL only**, NOT the importance weight (no joint prior / sampling
  prior). You CANNOT reweight it by likelihood for a weighted-posterior/shape check. Use
  `--extrinsic-proposal-output` (full log-weight, per-group GMM breadcrumb) instead.
- **`--save-P` defaults to 0.1** (prunes the bottom 10% of probability); on a peaked cloud that can
  drop nearly everything. Set `--save-P 0` for a raw export.
- **Portfolio `--save-samples`/`--extrinsic-proposal-output` gave 0-row/empty exports** for peaked
  runs until PR #35 (junior; oshaughnessy-junior/research-projects-RIT): the `_rvs` cleanup cumsummed
  LOG-weights and any `-inf` poisoned it -> 0 rows. Fix = linear-weight cumsum + `-inf` guard. If you
  see "zero-size array to reduction cupy_max", you are on pre-#35 code.
- **`--force-xpy` is inert without `--gpu`**; the pure-CPU path fails on float128 — run `--gpu` inside
  a cupy container. (See group memory `rift-gpu-container-invocation-gotchas`.)
- **AV resets between `integrate()` calls** — no seedable AV yet, so burn-in / calmarg warm-start
  gives AV no speedup (correctness-only). It CAN warm-start GMM/portfolio (model reuse).
- **`--fairdraw-extrinsic-output` is useless at low n_eff** (resamples ∝ weight -> copies of one
  point). For a posterior at low n_eff you must pool many copies.
