# Recommended configuration GROUPS

Copy-paste starting points. Each group is a SET -- the pieces were validated together and several
of them do nothing (or harm) in isolation. Read `option-combos.md` first: RIFT deliberately leaves
valid option combinations to the user.

Status key: **[default]** already on; **[opt-in, validated]** default-off, cleared the merge gate AND
the flag-on probe; **[opt-in, unproven]** cleared both gates but its benefit is not statistically
established; **[situational]** use only for the stated symptom.

---

## Group A -- typical event, single run (the baseline you should start from)

    --sampler-method portfolio --sampler-portfolio AV --sampler-portfolio GMM \
    --vectorized --gpu --force-xpy --interpolate-time True \
    --inclination-cosine-sampler --declination-cosine-sampler

- portfolio AV+GMM: AV is the robust workhorse, GMM catches curved within-group structure.
  VARAHA never-freeze is **[default]** and is what keeps AV from being starved of UPDATES.
- `--vectorized --gpu --force-xpy` selects the maintained NoLoop likelihood. `--force-xpy` is INERT
  without `--gpu`.
- `--interpolate-time True` **[opt-in, validated]**: cubic Q_lm time interpolation. Removes a
  time-quantization non-smoothness that is a discretization artifact, so the sampler stops chasing
  non-physics. Requires the NoLoop combo above. Cheap; turn it on.

## Group B -- sharply peaked / high-amplitude target (add to Group A)

    --force-adapt-all \
    --internal-rotate-phase \
    --sampler-warmstart-retry-neff 5

- `--force-adapt-all`: AV adapts ALL dimensions. At high SNR every extrinsic dimension is
  informative; without this the live volume contracts weakly. Frame-preserving, so it is safe with
  any warm seed.
- `--internal-rotate-phase`: samples phi+psi / phi-psi, decorrelating the phase-polarization
  degeneracy. **Improves the runs that land (n_eff ~50 vs ~30) but does NOT change how often they
  land.** If warm-starting, the seed MUST be transformed into the same frame (see
  `coordinates-and-degeneracies.md`) or it poisons the proposal -- measured 0/9 usable with a
  physical seed. Cold start sidesteps this entirely.
- `--sampler-warmstart-retry-neff 5` **[opt-in, validated]**: L0 auto-rescue. If a pass ends below
  the threshold (or terminates degenerately), re-seed from the peak THAT pass found and re-run.
  Same-problem reuse, cannot bias. **This is the single biggest reliability lever measured: usable
  runs 4/9 -> 8/9.** Costs a second pass on rescued runs.

## Group C -- mixture-degeneration guard (add to B when the target is sharp)

    --portfolio-varaha-min-frac 0.25
    # optionally also: --portfolio-varaha-max-frac 0.75  --internal-gmm-max-components 3

**[opt-in, unproven]** -- cleared the merge gate (0 blocking regressions) and the flag-on probe
(0 opt-in regressions), so it is SAFE; but the real-event benefit is p~0.08 at n=9, i.e. suggestive,
not established. No default was changed.

Symptom it addresses: the draw allocation collapses onto the momentarily-most-efficient member
(measured AV share -> 0.0099), q_mix loses its broad component, and any region the peaked member
misses is effectively unsampled -> lnZ biased LOW while n_eff looks HIGH. The floor keeps a broad
backstop in q_mix. The cap blocks the mirrored runaway (share -> 1); it is principled but added
nothing measurable over the floor alone (p=0.48).

Measured, 9 copies, matched seeds: sd(lnZ) 5.04 -> 3.01 (floor) / 3.08 (band); worst deviation from
own median 10.4 -> 6.1 / 7.2 nats. Every confidently-wrong copy in the study occurred with an
UNCONSTRAINED share; none with a constrained one.

Note `--internal-gmm-max-components 3` does nothing useful ALONE (sd 5.43 vs 5.96 baseline) and
alone produced the worst case recorded (n_eff 124, lnZ 10 nats low). Only use it WITH the share
constraint.

## Group D -- multiple instances of one sampler [situational]

    --sampler-portfolio AV --sampler-portfolio AV --sampler-portfolio GMM

Supported: the member list is not deduped, so this builds two independent AV instances, and
`--sampler-portfolio-args` aligns positionally for per-member configuration. The VARAHA share
constraint applies to the COMBINED share, so Group C composes with it. Reserved for hard cases;
not a default, and not yet benchmarked.

---

## How to JUDGE a result (this is not optional at high SNR)

1. **Never select or weight by n_eff.** It measures weight CONCENTRATION, not COVERAGE. Measured:
   the highest-n_eff copy in an ensemble (58) was 11 nats off; elsewhere n_eff 124 was 10 nats off.
   An n_eff-weighted pool is actively dragged toward the confidently-wrong copy, because that copy
   carries the largest weight.
2. **Use cross-copy CONSENSUS**, e.g. the median over copies that landed. The outlier is detectable
   only by disagreeing with the pool.
3. **No configuration makes a single run trustworthy on a sharp target** -- the best still deviates
   ~6 nats from its own median. Run several copies. Copies are needed both to FIND a good draw and
   to DETECT a bad one that looks good.
4. For a weight-correct posterior/shape check use `--extrinsic-proposal-output`, never the sparse
   sample export (it carries lnL only, not the importance weight).

## Before proposing any of this as a DEFAULT

It must clear BOTH tiers: the shape-recovery merge gate (`--jobs 1`; see `gotchas.md` for the pool
deadlock) and, because these are opt-in, the flag-on probe. A default-path gate run is bitwise
identical for opt-in code and proves nothing about it.
