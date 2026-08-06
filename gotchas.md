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

- **COLD portfolio start crashed on a sharp high-SNR peak: "boolean index did not match indexed
  array" (FIXED; pre-existing in rift_O4d and earlier).** `mcsamplerEnsemble.update_sampling_prior`
  dropped NaN-weight samples by REASSIGNING `ln_weights` -- but `ln_weights` is built ONCE, before the
  loop over dim_groups. The first group containing a NaN shrank it (e.g. 10000 -> 8686); every later
  group rebuilt `temp_samples` at full length and reused the stale shorter weights, so `GMM.update`/
  `GMM.fit` raised IndexError. Only reachable when the weights actually contain NaN -- i.e. a
  degenerate/cold pass -- so WARM runs never saw it, while every cold AV+GMM portfolio start on a very
  sharp peak died around chunk 8 and wrote NO output. Fix: filter into a loop-LOCAL name.
  **Generalized lesson (same class as the `sampler_method` clobber above): never mutate state that is
  shared across loop iterations / read later -- filter into a local. Two separate silent failures on
  this codebase came from exactly this pattern.**

- **`FAILED ANALYSIS` used to print only the exception message, not the traceback.** For faults deep
  in the sampler stack the message alone ("boolean index did not match...", "index out of range") is
  useless, and in a batch job this handler is often the ONLY record. It now prints the traceback too.
  If you are debugging on an older checkout, add `traceback.print_exc()` there first -- it converts a
  day of code archaeology into one run.

- **The L0 auto-rescue used to skip the case it was built for.** `mcsamplerPortfolio`/AV return
  `(None,None,None,None)` from their "terminate early" branch when the live volume never finds finite
  in-volume samples -- exactly the cold, ultra-sharp peak scenario. The rescue gate required
  `neff is not None`, so it never fired there. Now `neff=None` counts as below-threshold (the pass
  still populated `_rvs`, so the peak-seed is available).

- **AC support-truncation bias: lnZ biased LOW while every within-run error estimate reads fine.**
  In AC (`mcsamplerGPU.py`), an adapted 1-D histogram bin with exactly ZERO probability is an
  ABSORBING state — no future draw can land in it, so it can never be re-weighted, and the sampled
  support shrinks irreversibly. lnZ is biased low by exactly the excluded probability mass, and the
  bias is INVISIBLE to every within-run diagnostic: on a mild 2D Gaussian test, naive weight
  variance, block scatter, and khat all read ~0.02 while the true bias was **-0.32 nats**.
  Two traps in diagnosing it:
  - **Per-bin marginal comparisons CANNOT see it** — the drawn marginals match the claimed pdf
    exactly (the sampler is perfectly self-consistent on its truncated support).
  - The one-line detector that DOES work: draw fresh samples from the final adapted proposal and
    check `E[prior/p_s] == 1`. A value < 1 IS the supported prior-volume fraction.
  Root fix: commit `ff0a04ba` on branch `rift_O4d_mc_error_stabilization`
  (`~/RIFT_develUWM/src/research-projects-RIT`); mechanism + floor-value lore in
  `design-history.md` ("AC support-truncation bias"). Related knobs: `--adapt-floor-level`,
  `--adapt-weight-exponent` (`options-cheatsheet.md`).

- **The shape-recovery merge gate DEADLOCKS under its own multiprocessing pool.** Symptom: the run
  goes quiet forever -- elapsed time grows, CPU time does NOT (check
  `cut -d' ' -f14,15 /proc/<pid>/stat` twice, or `ps -o etime=,time=`), the log stops growing, and
  one worker sits on a large frozen CPU total while the others idle at ~0. Observed at `--jobs 16`
  (hung ~44/96 cases) and again at `--jobs 4` (hung mid-run). It is NOT branch-specific: both the
  base and the candidate arm hung on separate attempts, and the same code completes all 24 portfolio
  targets cleanly with **`--jobs 1`**.
  Fix: run the gate with `--jobs 1` (slower but reliable), or add a per-case timeout. Do NOT conclude
  a code regression from a gate hang until you have reproduced it single-process.
  Related trap: the *parent* process shows ~0 CPU even when healthy (it delegates to the pool), so
  parent CPU is not a liveness signal -- measure the WORKERS. And do not use log growth either: the
  gate's python is not line-buffered when redirected, so a healthy run can look silent.

- **[CIP] `zero-size array to reduction operation maximum` on a LOUD event = GMM overflowed, use AV.**
  Verified in production 2026-07 on real LIGO data (GW250114, lnL up to ~3307). Full chain: CIP hands
  the sampler `exp(lnL)`, not lnL -> `exp(3300)` is float64 `+inf` -> GMM (`mcsamplerEnsemble`)
  `_initialize` raises `ValueError: probabilities contain NaN` -> every `dat_logL` is non-finite ->
  CIP itself dies at `lnLmax = np.max(dat_logL[np.isfinite(dat_logL)])`
  (`bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py:3148`, branch `tdlike_paper2`;
  `origin/rift_O4d` byte-identical here). **The traceback you see is several frames from the cause** --
  it looks like a data/format problem in CIP and is actually the sampler. Diagnostic: if CIP is dying
  on an all-NaN reduction, print `max(lnL)` from `all.net` first.
  **Fix: `--sampler-method AV`.** Measured on the same real `all.net`: AV n_eff 33-72 vs GMM 1-4, and
  AV recovered the published detector-frame Mc (31.34 vs published ~31.3). AV needs no lnL shift, uses
  log-space weights, and runs happily on CPU.
  **This is the THIRD time GMM has bitten this group on a peaked target** (the other two: standalone
  GMM NaN-ing on chunk 1 at high SNR, above; and the earlier pp-correction round). Treat "GMM is
  unreliable on sharp/loud likelihoods" as a standing rule, not a per-event surprise.
  Overflow protection (below) stops the CRASH but does NOT rescue GMM's quality -- its ESS still
  collapses to ~1. See `samplers.md`, `recommended-configs.md` Group E.

- **[CIP] `--lnL-protect-overflow` is a SILENT NO-OP on the whole O4b/O4c/O4d/master line.** It is
  declared in argparse with a perfectly good docstring ("Before fitting, subtract lnLmax - 100. Add
  this quantity back at the end.") at
  `bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py:304`, and `opts.lnL_protect_overflow`
  is then **never referenced anywhere in the body** -- verified by grep on both `tdlike_paper2` and
  `origin/rift_O4d` (still broken there as of 2026-07). It was implemented only on the old O4a line
  and was dropped in the post-O4a refactors. Symptom: you add the flag to a loud-event CIP run and
  absolutely nothing changes, including the printed tempering exponent.
  **Workaround that works TODAY on any checkout: `--lnL-shift-prevent-overflow (lnLmax - 100)`** --
  same mechanism, static number, already wired (see `options-cheatsheet.md`).
  Fixed on `tdlike_paper2` (commit `1b96f488`) at `:2144-2147` as the *dynamic sibling* of the static
  flag -- `lnL_shift = max(lnL_shift, max_lnL - 100)` feeding the EXISTING `lnL_shift` plumbing, so it
  subtracts before every fit and is added back to the evidence, and the two flags compose via `max()`.
  Generalizable lesson: **a declared-but-unreferenced argparse option is invisible to every test and
  to every user** -- when a flag "does nothing", grep for the `opts.` name before theorising. This
  codebase has now produced two of these; a lint pass for unreferenced `add_argument` dest names would
  pay for itself.

- **[CIP] CIP grabs cupy whenever it is IMPORTABLE, and the GMM-family guard has no runtime probe.**
  Deployment trap on a heterogeneous pool. RIFT's integrator modules are INCONSISTENT about this,
  which is why the failure is confusing (branch `tdlike_paper2`; identical on `origin/rift_O4d`):
  - `mcsamplerAdaptiveVolume.py` (`:22`-`:54`), `mcsamplerGPU.py` (`:35`), `mcsamplerPortfolio.py`
    (`:33`), `mcsamplerNFlow.py` (`:78`) wrap the import in a **bare `except:`** AND execute a
    runtime probe `junk_to_check_installed = cupy.array(5)  # this will fail if GPU not installed
    correctly`. These degrade to numpy correctly on a broken/old driver.
  - `mcsamplerEnsemble.py` (GMM, `:10`-`:17`), `gaussian_mixture_model.py` (`:18`-`:25`) and
    `MonteCarloEnsemble.py` (`:15`-`:22`) catch **`except ImportError:` only and have NO probe**. On
    a node where cupy imports but the driver is too old, they set `cupy_ok = True` and the job dies
    later with a runtime `cudaErrorInsufficientDriver`, far from the import.
  **CIP does not need a GPU at all.** Our fix (verified in production 2026-07, CIT pool): run the
  container's stock CIP on ANY singularity CPU slot -- drop `request_GPUs`/`require_gpus`, set
  `requirements = (HAS_SINGULARITY =?= TRUE)`, and put a tiny shim directory first on `PYTHONPATH`
  inside the container whose `cupy.py` raises `ImportError`, forcing numpy for every module
  regardless of which guard it uses. Proof: a real CIP job landed on a CPU execute node, logged
  `no cupy (mcsamplerAV)`, n_eff 28.85, correct posterior.
  The naive alternative -- constraining CIP to GPU slots so cupy works -- is worse: it makes the
  cheap, serial CIP node compete for the same scarce GPU slots as ILE and simply starves the DAG.
  Two neighbouring deployment rules learned the same week:
  - **Every RIFT `.sub` that runs a GPU container needs the SAME GPU landing constraint as `ILE.sub`.**
    An inconsistent one lands the container on a node it cannot use.
  - **If your submit files select a container image by GPU capability tier, BUILD EVERY TIER.** We
    shipped a `Capability < 9.0` ceiling in `require_gpus` because the `cc90-120` (CUDA 12.8,
    Hopper/Blackwell) image had never been built. Nothing failed -- the DAGs just sat IDLE for ~3 days
    while ~50 free cap-8/12 slots were excluded. Building the second image and dropping the ceiling
    took "would match if drained" from 24 to ~55 slots. **A missing image is a throughput bug that
    presents as no error at all.** (Corollary hard gate: the two images must produce identical
    likelihoods -- build them from the same source commit and diff a deterministic probe.)

- **A RIFT DAG reporting "1 failed node / N futile" and EXITING WITH STATUS 0 does NOT prove it
  converged -- it may be a CRASHED convergence test laundered into success.** *(Corrected
  2026-07-29; an earlier version of this entry asserted the optimistic reading. Do not trust it.)*
  The iteration loop is stopped by `ABORT-DAG-ON <convergence_test_node> 1 RETURN 0`, i.e.
  "exit 1 == converged, stop the DAG, report success". `convergence_test_samples` returns 1 on
  purpose when converged -- but it also returns 1 when it **crashes**, and DAGMan cannot tell the
  difference: a genuine convergence and an ImportError are **bit-identical from the DAG's side**
  (same "1 failed" line, same futile count, same `EXITING WITH STATUS 0`).
  Verified 2026-07 on four real-data DAGs: the test died on `from RIFT.misc import
  hyperpipeline_io` because a `universe=local, getenv=*` node inherited a `PYTHONPATH` pointing at
  an OLDER RIFT checkout without that module. The loop stopped at iteration 2 with
  `--iteration-threshold 5` (which alone should have made an early stop impossible), the test
  never ran at all, and the "final" posteriors were mid-run snapshots -- the hand-run KL was
  **36.2 against a 0.02 threshold**.
  **How to actually check:** the newest `iteration_*_test/logs/test-*.err` must be **EMPTY** and
  the matching `.out` must contain a KL number. Empty `.err` + a KL verdict = real convergence.
  Non-empty `.err` = a crash wearing convergence's clothes. The presence of
  `posterior_samples-*.dat` proves **nothing** -- it is written every iteration regardless.
  Cross-check the stopping iteration against `--iteration-threshold`; an early stop is a red flag.
  **Two traps that follow:**
  (a) When `ABORT-DAG-ON ... RETURN 0` fires, **DAGMan writes no rescue file.** The newest rescue
  on disk is stale, so a bare resubmit silently re-runs every node completed since it (for us:
  ~576 expensive ILE/CIP nodes). Reconstruct the true DONE set from `dagman.out` union the prior
  rescue and hand-build the rescue before resubmitting.
  (b) The root cause generalises: **every `getenv=*` local node inherits the submitting shell's
  `PYTHONPATH`**, so any stale RIFT on that path silently version-skews submit-host nodes
  (this same skew separately broke `util_ParameterPuffball`). Wrap such nodes to prepend the
  intended `Code/` directory, or point `RIFT_CODE` at a stable checkout before launching.

- **`gmm_dict` is a live, mutated object -- never store or reuse a reference to it.**
  It looks like an inert grouping spec (`{(0,1,2): None, (3,): None}`), and production always
  supplies one. But `mcsamplerEnsemble.setup()` passes the caller's dict straight to
  `monte_carlo.integrator`, which stores it **without copying** (`MonteCarloEnsemble.py:110`) and
  then writes trained models into it (`self.gmm_dict[dim_group] = model`, `:403`). So after one
  integration the caller's "spec" holds that run's fitted proposal.
  Anything that caches setup arguments in order to replay them later (a reset between sequential
  points, a checkpoint, a member rebuild in a portfolio) therefore hands the **next** point the
  **previous** point's trained GMM unless it snapshots the dict. Snapshotting on store is only half
  the fix: the replay must pass a **fresh copy each time**, or the rebuilt integrator trains into
  the snapshot and the leak simply returns one point later.
  **Severity, measured (2026-08, truth-known 2-D displaced-target pair):** with the trained
  proposal leaking across a reset, `n_eff` fell 2036/1583/823 -> 1700/487/196 while `|lnZ bias|`
  stayed <= 0.010 in **both** arms. A stale GMM proposal is merely a *bad* proposal, so only
  efficiency suffered here. Contrast the **AV** grid leak, which removes support and genuinely
  biases (n_eff 1, lnZ bias -59.7).
  **CORRECTION (2026-08-06):** an earlier version of this entry explained the mild severity by
  saying the GMM member "still covers the support". That is WRONG -- see the entry below on the
  1e-300 score floor. In the default configuration the GMM has no defensive component, and its
  apparent far-field density is a numerical floor, not coverage. The correct reading of the
  measurement is that the far region was never *sampled* in those runs, not that it was covered.
  **Consequence for testing:** no statistical assertion can detect the GMM aliasing leak -- every
  n_eff/JS/bias check passes. Assert on the object directly (`all(v is None for v in
  integrator.gmm_dict.values())` after the reset), and record separately that training actually
  happened first, or the check passes vacuously. Likewise, a configuration test that omits
  `gmm_dict` exercises a different branch of `setup()` and cannot see this class of defect at all;
  grouping, `n_comp` and `gmm_adapt` all survive the aliasing bug intact.

- **You cannot `deepcopy` a RIFT GMM model -- and the failure is silent if you wrote a fallback.**
  A `gaussian_mixture_model.gmm` (and `estimator`) holds a module reference (`xpy`, numpy or cupy)
  and bound functions, so `copy.deepcopy(model)` raises `TypeError: cannot pickle 'module' object`.
  Any "clone it, fall back to the original on failure" helper therefore takes the fallback on
  **every real model**, leaving it shared -- the isolation you thought you added is inert, and
  nothing tells you.
  **What works:** `copy.copy(model)` (a shallow copy never pickles, so the module ref is simply
  shared -- which is fine, it is stateless), then clone the mutable attributes yourself: `means`,
  `weights`, `p_nk`, `covariances`, `adapt`. Verify by mutating the clone and asserting the
  original's `means` and `tempering_coeff` are unchanged -- an identity check (`clone is not orig`)
  passes even when the attribute state is still shared.
  **Where this bites:** anything that snapshots a seeded model to restore later. With
  `--extrinsic-proposal-breadcrumb --extrinsic-proposal-adapt`, seeded groups keep re-fitting and
  `gmm.update()` mutates in place, so a snapshot-by-reference drifts with the run and replays the
  previous point's adaptation into the next one. With adapt OFF (the default) `_train` skips seeded
  groups and nothing mutates, so the bug is invisible until someone enables the flag.

- **A reused sampler carries its contracted live volume into the NEXT point (`--n-events-to-analyze > 1`).**
  `mcsamplerPortfolio.integrate_log` does not call `self.setup()`, so member state -- in particular
  the AV/VARAHA live volume, which has contracted around the point just finished -- survives into
  the next integral. If the next point's support lies outside that box, the estimate is biased LOW
  with a healthy-looking n_eff and no error. This needs no warm-start option: it bites any job that
  analyses several points with one sampler.
  **Measured** (truth-known 2-D pair of displaced targets, one reused portfolio): with no reset,
  point 2 gave n_eff **0-10** and lnZ bias **+0.08 / -35.9 / -449.9** across three seeds; with the
  reset, n_eff 823-2036 and |bias| <= 0.008.
  **Fixed** in `mcsamplerPortfolio.clear_warm_state()` (merged into `rift_O4d` 2026-08-06, PR #45),
  which nulls `_warm`/`_warm_applied` on every member and re-runs each member's `setup()` with its
  ORIGINAL arguments replayed from a snapshot. The driver calls it unconditionally at the top of
  every point after the first. Note the ordering that makes it safe: reset FIRST, then install any
  intentional seed for the new point.
  **Contrast with the GMM proposal leak** in the entry above: that one costs efficiency only,
  because the AV member still covers the support. This one REMOVES support, so it biases. When
  triaging a suspected state leak, ask first which kind it is -- it decides whether any statistical
  check could have caught it.

- **You cannot build a lnZ-bias alarm out of "mass the warm member could not have drawn".**
  Tempting idea: in a portfolio, flag samples where the warm-started member's density is exactly
  zero (`escaped_mass = sum_{q_m(x)=0} w / sum w`), since a misplaced seed should let another
  member find mass outside the seeded box. Measured over 1440 runs (20 target seeds per cell,
  d=4 and d=6, seed displacement 0-4), it does not work as a correctness gate, for a structural
  reason worth remembering:
  **escaped weight is observable only when another member covers the complement of the warm
  member's support -- which is exactly the configuration in which the balance heuristic already
  keeps lnZ unbiased.** You can see the escape only when the escape is harmless. The statistic is
  a lower bound on mismatch, never an upper bound.
  Concretely: (a) in `[AV warm, GMM cold]` the first-chunk statistic IS sharp (floor ~1e-5, TPR
  1.00 at displacement 2, d=4) -- but |lnZ bias| never exceeded 0.31 nat there anyway, so it flags
  efficiency loss, not a wrong answer; (b) in the DEFAULT portfolio it reads exactly 0.000 in
  320/320 runs, because `mcsamplerEnsemble` also implements `bootstrap_from_samples`, so
  `portfolio.bootstrap_from_samples` seeds every member and nothing is drawn outside the seeded
  volume; (c) in all-AV -- the configuration that IS biased, median -22.6 nats at displacement 4 --
  the best variant only reaches TPR 0.6-0.85 once the median bias is already -2 to -4 nats, and at
  d=6 the runs it misses are the worse ones.
  **The cumulative form has no floor at all**: on a CORRECTLY placed seed it reads median 0.51
  (d=4) / 0.80 (d=6), because a well-placed VARAHA member contracts to a likelihood contour that
  legitimately excludes most of the posterior weight. Only the first-chunk form has a usable floor.
  **Method note:** the cumulative statistic scored AUC 1.000 at every displacement in the
  all-seeded arm until a FIXED-BUDGET control was added, whereupon it collapsed to 0.47 -- it was
  measuring run length, and `1000/n_eff` scored 1.000 in the same cells. Always run the
  matched-budget control before believing a separation.
  Keep the number as an off-path warm-start QUALITY monitor if you like (threshold
  `escaped_mass_early > 1e-2`, 0/40 false positives on matched seeds), but do not gate lnZ on it
  and do not use it to trigger a rerun.


- **`gmm.score()` floors at 1e-300 -- that is a log(0) guard, NOT coverage.**
  It is tempting to argue that a GMM member gives a portfolio full support because a Gaussian
  mixture has nonzero density everywhere. In this code it does not, twice over:
  (a) the tails underflow -- a fixed-component fit to a tight cloud returns *exactly* the 1e-300
  floor at the far corner of the [-5,5]^d box for every d >= 4 (measured d=2 2.9e-273, d>=4
  1.0e-300); and (b) `score()` clamps with `maximum(scores, 1e-300)`
  (`gaussian_mixture_model.py:621`), so you can never distinguish "genuinely tiny" from
  "underflowed to zero". A sample drawn where q is the floor carries importance weight
  `L*p/q ~ 1e300`: that does not support the estimate, it destroys it.
  **The only real guarantee is the defensive component** (`add_defensive_component`, Hesterberg
  1995), which puts a broad box-covering Gaussian at weight `gmm_defensive_frac`. With it, the
  same far corner reads 1.3e-04 (d=2) and 7.6e-10 (d=6) -- small but usable.
  **Trap:** until 2026-08 `add_defensive_component` was called ONLY from `fit_gmm_adaptive`, while
  `gmm_adaptive` defaults to off -- so the documented default `gmm_defensive_frac=0.05` was inert
  on the default path. Any code reading `gmm_defensive_frac > 0` as a guarantee was wrong by
  default. Check the fitted model (`model.defensive_frac`), never the config value.
