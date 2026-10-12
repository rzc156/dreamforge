# Results (running log)

All numbers below come from `runs/` artifacts in this repo and are regenerated
by `python -m dreamforge.report --run runs/main`.

## Setup

* Policy: `lerobot/smolvla_libero` (SmolVLA-450M), unchanged at hand-off.
* Suite: LIBERO-Spatial (10 tasks), episodes capped at the LIBERO protocol
  (280 steps; a 50-step chunk may overshoot to ≤330), seeds fixed.
* World model: latent dynamics (3.75M params, `aso` checkpoint) + progress head.
* CEM: horizon 50, population 64, 8 elites, 6 iterations, actions clipped to
  dataset min/max; search seeded with the base policy's own chunk.

## W0 — pre-flight: the world model must respond to actions

Before trusting imagination, `dreamforge diagnose` measures action response:

| world model | score std over random plans | action-swap Δprogress | latent swap norm |
|---|---|---|---|
| `base` (1.39M) | 0.0003 | 0.0003 | ≈0 |
| `aso` (3.75M)  | 0.0215 | 0.0150 | large |

The unregularised model ignores the action entirely (a reproduction of the
"action vacancy" phenomenon); CEM cannot optimize it. All runs below use `aso`.

## W1 — imagined corrections improve imagined progress

`imagine` (CEM on 48 failure states from 8 failed episodes):

* mean imagined progress: **0.480 → 0.516** (mean gain **+0.041**, sd 0.025)
* CEM history rises monotonically within its 6 iterations

## W2 — imagined corrections have real value in the simulator

`verify`: each saved state is restored exactly (`MjSimState` round-trip) and
replayed with three arms, then the unchanged policy resumes.
`none` reproduces the original episode outcome (validated: identical episode
length on replay), which anchors the comparison.

| arm | n states | success | 95% Wilson CI |
|---|---|---|---|
| none | 48 | 7 | 14.6% [7.2, 27.2] |
| **imagined** | 48 | **16** | **33.3% [21.7, 47.5]** |
| random | 48 | 3 | 6.2% [2.1, 16.8] |

* **imagined vs none: +18.8pp**, 11 states improve / 2 worsen, exact McNemar
  **p = 0.0225**
* **imagined vs random: +27.1pp**, 13 improve / 0 worsen, **p = 0.0002**

Two things stand out:

1. *Imagination is not just perturbation.* Random action sequences of the same
   length are worse than doing nothing (6.2% vs 14.6%); the model-optimised
   corrections are far better than both.
2. *Imagined progress is a weak predictor of real value* (point correlation
   +0.12 between WM gain and real delta; high-gain half +20.8pp vs low-gain half
   +16.7pp). This extends the previous study's conclusion — an offline world
   model score does not proxy decision quality — from *selection* to
   *correction*.

Breakdown by where in the episode the correction is applied:

| decision step | n | imagined | none | delta |
|---|---|---|---|---|
| start (step 0) | 8 | 75% | 25% | +50pp |
| early (≤100) | 16 | 50% | 25% | +25pp |
| late (>100) | 24 | 8% | 4% | +4pp |

Corrections matter most early — late-stage states are often already
unrecoverable, which also bounds how much a world model can be expected to fix.

> Caveat: these 48 states come from 8 failed episodes, so states within an
> episode are correlated and the intervals are optimistic. The numbers are
> being extended to the full mining run in the next pass.

## W3 — training on imagined corrections (in progress)

Datasets built from the same corrections:

| filter | rule | samples |
|---|---|---|
| `all` | every imagined correction (no simulator) | 48 |
| `gained` | world-model score gain ≥ 0.05 (no simulator) | 20 |
| `lifted` | real replay flipped the outcome (uses simulator) | 11 |

Fine-tuning (SmolVLA, flow-matching loss, lr 5e-6, batch 16, 50% expert replay)
and paired closed-loop evaluation are running; results will be appended here.
