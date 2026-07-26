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

- **Coordinate-transform flags need a FRAME-MATCHED warm-start seed.** `--internal-rotate-phase`
  (sampler's phi_orb/psi hold phase_p=phi+psi, phase_m=phi-psi in [0,4pi]) and
  `--internal-sky-network-coordinates` (H1-L1 network sky frame) change what the sampler's parameter
  slots MEAN. But `--sampler-warmstart-samples` maps the seed to the sampler by COLUMN NAME and does
  NOT transform values (`bootstrap_from_samples`, driver ~2714). So a seed in PHYSICAL coordinates
  (e.g. a cherry-picked-pilot `--save-samples` export, which the driver un-rotates on output) is
  poured straight into the rotated/network slots -> the warm live region lands in the wrong place ->
  the run COLLAPSES (observed: a high-SNR point that landed n_eff~36 with a physical seed + no
  rotation collapsed to n_eff~1 for ALL seeds when the rotations were added with the same physical
  seed). Symptom: adding the coordinate flags makes n_eff uniformly WORSE, not better.
  Fix: transform the seed into the sampling frame first. Phase is trivial:
  `seed_phi_orb = mod(phi+psi, 4pi)`, `seed_psi = mod(phi-psi, 4pi)`. Sky-network needs the driver's
  `my_rotation` on (ra,dec). Or generate the pilot with the SAME rotation flags. `--force-adapt-all`
  is frame-preserving (only sets which dims AV contracts), so it is safe with a physical seed.

- **`opts.sampler_method` is CLOBBERED to 'GMM' during portfolio setup.** In the driver's portfolio
  member loop, the GMM branch does `opts.sampler_method='GMM'` (to force GMM-specific arg-parsing).
  So for an AV+GMM portfolio, EVERY downstream `opts.sampler_method == 'portfolio'` check is False,
  and `== 'GMM'` is True. This silently (a) made the portfolio-only setup block dead code (the GMM
  branch happens to cover the gmm_adaptive forwarding, so no visible harm) and (b) broke the L0
  auto-rescue gate for the portfolio. Detect a portfolio via `opts.sampler_portfolio` (the member
  list, which is NOT clobbered), never via `opts.sampler_method`, in any code after sampler setup.
