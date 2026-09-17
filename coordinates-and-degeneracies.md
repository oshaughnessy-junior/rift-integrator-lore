# Extrinsic coordinates & degeneracies

The extrinsic posterior has characteristic degeneracies. Sampling in RAW coordinates leaves them
correlated, which forces the proposal to bend across axes it can't (AC factored histogram, GMM
factored groups) and makes fits seed-sensitive. Choosing coordinates that DECORRELATE the
degeneracy is often worth more than any sampler tuning.

## The three degeneracy groups (also the default GMM groups)

- **(RA, dec) — sky.** With only 2 IFOs the sky localization is a **ring**, not a point (multimodal).
  High SNR makes the ring THIN but not gone. GMM needs several sky components (default 4) to cover it.
- **(distance, inclination) — the dL-ι degeneracy.** A curved arc (louder ⇄ nearer/face-on). Needs a
  proposal that can bend within the group.
- **(phi_orb, psi) — phase-polarization.** Strongly degenerate, especially with 2 IFOs; typically the
  LEAST-constrained extrinsic direction (broad even in good runs).

## Coordinate options (driver flags)

- `--inclination-cosine-sampler`, `--declination-cosine-sampler` — sample cos(ι), cos(dec) so the
  measure is uniform on the sphere / isotropic in inclination. Almost always on.
- `--internal-rotate-phase` — sample `phase_p = phi+psi` and `phase_m = phi-psi` (each 0..4pi, prior
  2x larger) instead of (phi, psi). **Decorrelates the phase-polarization degeneracy** — directly
  targets the loosest group. Turn ON for high SNR / whenever the (phi,psi) group is broad/unstable.
- `--internal-sky-network-coordinates` — rotate the sky frame to align with the first two IFOs, so
  the ring becomes (roughly) one well-behaved coordinate + one along-ring coordinate. **Tames the
  2-IFO sky ring.** `--internal-sky-network-coordinates-raw` uses the IFOs in the given order without
  reordering.

## Recipe for a high-SNR / sharp target

    --force-adapt-all --internal-rotate-phase --internal-sky-network-coordinates \
    --inclination-cosine-sampler --declination-cosine-sampler

Rationale: high SNR sharpens every extrinsic dimension, so (a) AV must adapt ALL of them to contract
usefully (`--force-adapt-all`), and (b) the phase-pol and sky degeneracies must be rotated into
near-independent coordinates or the GMM has to fit correlated structure it factorizes away — the
observed symptom of skipping this is a bimodal n_eff LOTTERY (half the seeds collapse to n_eff~1 with
a single degenerate extrinsic mode). See DESIGN_portfolio_freeze_policy.md (high-SNR study).

## Symptom -> coordinate fix

| symptom | likely fix |
|---------|-----------|
| n_eff collapses on ~half of seeds (lottery) at high SNR | add `--force-adapt-all` + the two rotations |
| (phi,psi) group broad / unstable across copies | `--internal-rotate-phase` |
| sky posterior won't converge, 2 IFOs | `--internal-sky-network-coordinates` |
| AV stalls at n_eff~1 though warm-started | check it is adapting all dims (`--force-adapt-all`) |

## What the n_eff lottery actually needs (measured on a synthetic target, 2026-09-17)

The lottery above was only ever seen on real high-SNR events, where several degeneracies are present
at once. A synthetic zero-strain H1 fixture with an analytic supplementary factor separates them,
because the factor's marginal is known in closed form and its shape is chosen, not inherited.

Factor `A cos(phi_orb) + B cos(iota)`, exact `ln Z = ln I0(A) + ln(sinh(B)/B)`, GMM, 8 seeds each:

| case | worst sigma | least n_eff | collapse? |
|---|---|---|---|
| A=0.75, B=0 | 0.0160 | 750 | no |
| A=8, B=0 (sharp phi_orb peak, no inclination term) | 0.0349 | 267 | **no** |
| A=0.75, B=3 | 0.0235 | 381 | no |
| A=8, B=2 | 0.0732 | **14.5** | **yes, 2 of 8** |

So a sharp peak in ONE circular parameter is not sufficient. `A=8, B=0` and `A=8, B=2` share the
same phi_orb peakedness and differ only by the inclination term, and only the second lotteries. The
dL-inclination arc is doing the work, which matches the "correlated structure it factorizes away"
rationale above, and does not add to it.

SCOPE, because it is easy to over-read: the fixture is H1-only, so neither the sky ring nor the
2-IFO phase-polarization structure exists in it. This reproduces ONE of the three degeneracies the
high-SNR study names, not the production failure mode. The evidence is correct in every case above,
including the collapsing one. It is n_eff that lotteries, not ln Z.

Do not fix such a lane by adding `--force-adapt-all` or the rotations when a closed-form answer is
what the lane measures: `--internal-rotate-phase` changes which coordinates a supplementary factor
is handed, so the closed form no longer describes the integral being computed.
