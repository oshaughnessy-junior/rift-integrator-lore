# Design history & rationale

## Balance-heuristic mixture (portfolio) -> unbiased for any weights
The portfolio reweights every sample by the full mixture density `q_mix = Σ_m frac_m q_m`
(multiple-importance-sampling balance heuristic). Consequence: the lnZ estimate is unbiased for ANY
member allocation `frac_m`. This unlocks (a) making AV freeze-exempt (a continuously-updated member
can only cost draws, not bias), and (b) adaptive draw allocation by a quality signal.

## The VARAHA freeze bug and fix
AV (VARAHA) contracts its volume ONLY when updated (`_selfish`); before contracting its n_ess ~1, so
the n_ess reweighting froze it at chunk 1 and it never contracted -> portfolio rode a stalling GMM.
Grace/revive can't fix it (weight feedback: AV needs to contract to earn weight, needs weight to be
updated). Fix: `portfolio_varaha_never_freeze=True` (default) — AV updates every chunk past its
breakpoint, exactly like standalone AV. Plateau-revive was tried and defaulted OFF (halved n_eff in
the shape gate).

## Draw allocation quality signals
Kish n_ess is scale-invariant (blind to integral mass), so it can't rank members by contribution.
Alternatives: `global` = marginal pooled-n_eff gain; `credit` = MIS credit assignment
(frac_m q_m / q_mix summed with weights). Adaptive allocation is opt-in; hard to beat plain AV unless
the problem is strongly correlated.

## Weight clipping (Ionides truncated IS) — adaptation-only
Clipping heavy importance weights can stabilize the PROPOSAL FIT, but clipping the ESTIMATOR biases
lnZ, and clipping the n_ess report inflates Kish n_ess. So clipping is applied ONLY to the
proposal-fit input (opt-in), never the estimator or the reported n_ess.

## AC support-truncation bias (zero histogram bins are absorbing)
The failure (see `gotchas.md` for the symptom-side entry): in AC
(`MonteCarloMarginalizeCode/Code/RIFT/integrators/mcsamplerGPU.py`), once an adapted 1-D histogram
bin reaches exactly zero probability it is an absorbing state — it can never be re-drawn, so it can
never earn weight back, and the sampled support only ever shrinks. The estimator is then unbiased
*on the truncated support* but misses the excluded mass entirely, so lnZ is biased low and every
within-run error estimate (naive weight variance, block scatter, khat) stays tiny: ~0.02 reported
vs a -0.32-nat true bias on a mild 2D Gaussian. Detector: fresh draws from the final adapted
proposal must satisfy `E[prior/p_s] == 1`; the shortfall equals the lost prior-volume fraction.

Fix (commit `ff0a04ba`, branch `rift_O4d_mc_error_stabilization` of
`~/RIFT_develUWM/src/research-projects-RIT`), three parts:

1. **`HIST_FLOOR_LEVEL_MIN = 1e-2` clamp in `compute_hist`.** Any nonzero floor converts the
   invisible truncation *bias* into visible *variance* (the bin can be re-drawn, so lost mass shows
   up as occasional large weights instead of silently vanishing). Value matters: `1e-3` still left a
   -0.06-nat heavy-tailed residual; `1e-2` was clean.
2. **`integrate_log` adaptation weights were wrong.** They were `lnL + max(maxlnL, 200)` — ignoring
   both `tempering_exp` (`--adapt-weight-exponent`) and the `1/p_s` importance correction. With
   near-flat weights, each histogram just replays the previous proposal's *sampling noise*: a
   neutral multiplicative random walk that progressively collapses the proposal onto a comb of
   surviving bins (this is what feeds mechanism 1). Fixed to
   `exp(tempering_exp*lnL + ln p - ln p_s)`, so each histogram estimates the FIXED target
   `L^beta * prior` instead of compounding the previous proposal.
3. **`integrate_log`'s `n_adapt` freeze test double-multiplied by `n`** and therefore never froze
   adaptation.

Transferable lesson: a proposal-update rule must never be able to assign exactly zero probability
to a region the prior supports, and adaptation weights must target a fixed distribution — anything
that reweights relative to the *current* proposal without the `1/p_s` correction is a random walk.

## High-SNR study conclusion (S250114ax best-fit point)
n_eff is a bimodal LOTTERY for every GMM cap (survivorship bias if judged on few seeds). Low n_eff is
an extrinsic-degeneracy collapse. `correlate-all` is worse (fitting variance > correlation benefit).
The robust recipe is pooling many copies weighted by reliability. OPEN QUESTION being tested: does
`--force-adapt-all --internal-rotate-phase --internal-sky-network-coordinates` (full adaptation +
decorrelated coordinates) drop the collapse rate — i.e. is the lottery a coordinate/adaptation
artifact rather than intrinsic portfolio instability? The portfolio is *designed* to stabilize this
(AV backstop + GMM detail), so a persistent lottery points at member mis-configuration.

Full running record: `RIFT/integrators/DESIGN_portfolio_freeze_policy.md` (in the code repo).
