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
