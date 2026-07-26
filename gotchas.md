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

- **`opts.sampler_method` was CLOBBERED to 'GMM' during portfolio setup (FIXED upstream; still true
  on older checkouts).** The driver's portfolio member loop used to set `opts.sampler_method='GMM'` (to
  force GMM-specific arg-parsing for a GMM member). For an AV+GMM portfolio that made EVERY downstream
  `opts.sampler_method == 'portfolio'` test False and `== 'GMM'` True: the portfolio-only setup block
  became dead code, the L0 auto-rescue gate never fired for a portfolio, and `--internal-use-lnL` took
  GMM's branch (getting `return_lnI`, which the portfolio does not consume). Now replaced by a
  non-destructive `use_gmm_member` flag + `use_gmm_args = (method=='GMM') or use_gmm_member`.
  NOTE: the portfolio still ran only because its setup branch force-sets `internal_use_lnL=True` --
  that forcing is INTENTIONAL design (see option-combos.md), not a lucky accident; RIFT deliberately
  leaves valid option combinations to the user rather than wiring them in.
  LESSON that outlives the fix: **never branch on a mutable `opts.*` field that setup code rewrites** —
  and if you must know whether a portfolio is active, use `opts.sampler_portfolio` (the member list) or
  the local `use_portfolio` flag, which are not rewritten. If your checkout still has the clobber,
  every `sampler_method=='portfolio'` check after sampler setup is silently dead.
