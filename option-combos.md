# Valid option COMBINATIONS (read this before composing a command line)

**The single most important thing to understand about RIFT's options.**

RIFT has accumulated many options over a decade. A large number of them are *de facto* defaults —
settings that are intended to be used, and intended to be used **together** — but the logic that
would wire those dependencies automatically was never added, because doing so across ten years of
accreted options would be spaghetti. **This is deliberate: the burden of knowing valid, current
option combinations sits with the USER, not the code.** The code will usually not warn you, and an
invalid or incomplete combination frequently produces a silently-worse (not failing) result.

That is precisely why this lore repo exists. When you learn a combination, record it here.

## The combinations

### NoLoop likelihood (the maintained one)
    --vectorized --gpu --force-xpy
Selects the maintained **NoLoop** likelihood path. Notes:
- `--force-xpy` is **inert without `--gpu`** (a longstanding trap).
- The pure-CPU path fails on float128; run `--gpu` inside a cupy container.

### Cubic time interpolation (USE THIS)
    --vectorized --gpu --force-xpy   --interpolate-time True
`--interpolate-time` evaluates Q_lm at **fractional** detector times by cubic interpolation instead
of snapping to the nearest sample bin. It **requires the NoLoop likelihood** (the combo above);
without it the flag does nothing. Takes a truthy VALUE (`True`/`1`/`yes`), not a bare switch.

Why it matters for the integrator: nearest-bin time quantization injects a **superfluous
non-smoothness into the extrinsic likelihood surface**. The samplers then have to chase structure
that is a discretization artifact rather than physics. Turning on cubic interpolation removes it and
makes convergence more robust. Default is off for backward compatibility — turn it on.

### lnL mode
- `--internal-use-lnL` — likelihood returns lnL and the integrator integrates lnL. Only valid for
  samplers in `ok_lnL_methods`.
- **Portfolio forces it on** (`opts.internal_use_lnL=True` in the portfolio setup branch): the
  portfolio only implements the lnL scenario, and `mcsamplerPortfolio.integrate()` RAISES
  "must integrate lnL" if `use_lnL` is not set. This forcing is intentional, not accidental.
- **`return_lnI` is REQUIRED for a STANDALONE GMM in lnL mode.** It is consumed only by
  `mcsamplerEnsemble.integrate()` (the GMM driver's own integration loop), where it controls whether
  the integrand is treated as lnI. The driver pairs it with `use_lnL` for `--sampler-method GMM`.
- **For a PORTFOLIO carrying a GMM member, `return_lnI` is inert** — the portfolio never calls
  `member.integrate()`; it drives members via `draw_simplified()` / `update_sampling_prior()` and
  does the integration itself in `integrate_log()`. So the GMM member never reaches the code that
  reads `return_lnI`. (Verified: `mcsamplerPortfolio` has zero references to it, and the CI test
  below is bit-identical with and without it on the portfolio path.)

### Portfolio selection
    --sampler-method portfolio --sampler-portfolio AV --sampler-portfolio GMM
Repeat the flag per member (`action='append'`). The driver default `adaptive_cartesian_gpu` is the
classic Cartesian GPU sampler — NOT AV, NOT the portfolio.

### High-SNR robustness (see use-cases.md)
    --force-adapt-all --internal-rotate-phase --sampler-warmstart-retry-neff 5
Plus a **phase-frame-matched** warm seed if warm-starting (see coordinates-and-degeneracies.md), or
run cold, which sidesteps seed frame-matching entirely.

## The CI test that covers lnL vs non-lnL

    python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test [--use-lnL]

The raw integrator test: compares the default mcsampler, adaptive-Cartesian, GMM (mcsamplerEnsemble)
and NF on a synthetic target, in both lnL and non-lnL mode (`--use-lnL` sets `return_lnI` too). Run
BOTH modes when touching anything in the lnL / `use_lnL` / `return_lnI` wiring. Compare the
`GMM/default` line across arms; note NF flow-training output is nondeterministic, so diff the GMM
numbers, not the whole log.

Known pre-existing quirk: in lnL mode the **variance** estimates come back `nan` for the
adaptive-Cartesian and GMM arms (`[0.0288, nan, nan, 0.0188]`) while the integral ratio is fine
(GMM/default ~0.9967). Present on both `rift_O4d` and later branches — not a regression.
