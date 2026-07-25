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

## High-SNR study conclusion (S250114ax best-fit point)
n_eff is a bimodal LOTTERY for every GMM cap (survivorship bias if judged on few seeds). Low n_eff is
an extrinsic-degeneracy collapse. `correlate-all` is worse (fitting variance > correlation benefit).
The robust recipe is pooling many copies weighted by reliability. OPEN QUESTION being tested: does
`--force-adapt-all --internal-rotate-phase --internal-sky-network-coordinates` (full adaptation +
decorrelated coordinates) drop the collapse rate — i.e. is the lottery a coordinate/adaptation
artifact rather than intrinsic portfolio instability? The portfolio is *designed* to stabilize this
(AV backstop + GMM detail), so a persistent lottery points at member mis-configuration.

Full running record: `RIFT/integrators/DESIGN_portfolio_freeze_policy.md` (in the code repo).
