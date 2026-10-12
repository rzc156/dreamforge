# Results

Everything below is regenerated from the artifacts in `runs/main` by
`python -m dreamforge.report --run runs/main` (see `runs/main/report.md`).

## Setup

* Policy: `lerobot/smolvla_libero` (SmolVLA-450M), unchanged at hand-off.
* Suite: LIBERO-Spatial (10 tasks); episodes capped at the LIBERO protocol
  (280 steps, a 50-step chunk may overshoot to ≤330); fixed seeds.
* Mining: 150 episodes (10 tasks × 15) → **60 failed episodes → 360 failure
  states**. Base policy on the standard 10-episode protocol: **61.0%**
  (61/100).
* World model: latent dynamics + progress head, 3.75M params, `aso`
  checkpoint (see W0). CEM horizon 50, population 64, 8 elites, 6 iterations,
  trust region 0.5σ per action dimension around the policy's own chunk.

## W0 — pre-flight: the world model must respond to actions

`python -m dreamforge.diagnose --mine runs/main/mine --wms base aso`:

| world model | score std over random plans | action-swap Δprogress | |z0| |
|---|---|---|---|
| `base` (1.39M) | 0.0003 | 0.0003 | 33 |
| `aso` (3.75M) | 0.0215 | 0.0150 | 14 |

The unregularised model ignores the action almost exactly (a reproduction of
the *action vacancy* phenomenon); CEM cannot optimise it. All runs use `aso`.
Takeaway, shipped as a tool: **measure action response before trusting a world
model for imagination.**

## W1 — imagination raises imagined progress

360 failure states, CEM seeded with the base chunk, 0.5σ trust region:

* mean imagined progress: **0.480 → 0.500** (mean gain **+0.020**, sd 0.021)
* CEM's best score improves monotonically over its 6 iterations

The gain is small — consistent with a policy that is already competent.

## W2 — what imagined corrections are worth in the simulator

Each saved state is restored exactly (`MjSimState` round-trip; the `none` arm
reproduces the original episode length, which anchors the comparison) and
replayed with three arms before the unchanged policy resumes.

| arm | n | success | 95% Wilson CI |
|---|---|---|---|
| none | 360 | 34 | 9.4% [6.8, 12.9] |
| imagined | 360 | 49 | **13.6% [10.5, 17.5]** |
| random (same per-dim budget) | 360 | 53 | **14.7% [11.4, 18.8]** |

* **imagined vs none: +4.2pp**, 29 states improve / 14 worsen, exact McNemar
  **p = 0.032** — corrections do have real causal value.
* **imagined vs random: −1.1pp**, 15 improve / 19 worsen, **p = 0.61** — the
  world model's *direction* buys nothing over an equal-sized random
  perturbation. The point estimate even favours random.

Episode-clustered view (states within an episode are correlated, so the
state-level p-value is optimistic):

* 60 episodes with paired states; mean per-episode delta **+4.2pp**;
  **21 better / 30 equal / 9 worse** episodes.

Where the correction helps:

| decision step | n | imagined | none | delta |
|---|---|---|---|---|
| start (step 0) | 60 | 32% | 17% | **+15pp** |
| early (≤100) | 120 | 23% | 18% | +6pp |
| late (>100) | 180 | 1% | 2% | −1pp |

Late-stage failures are essentially unrecoverable by a 16-step correction,
which bounds what any world model can be expected to fix.

## W2b — targeted corrections (only where correction is possible)

The step breakdown above says the effect lives at episode *starts*, so training
on all 360 corrections dilutes the useful signal with late corrections that
change nothing. We therefore build a second dataset with
`--max-decision-step 50` (start + first decision point, 120 samples) and run
the same fine-tuning + paired evaluation (`ft_early`). This is a data-targeting
test, not a new method.

### An unconstrained-budget sweep is left as future work

The pilot pass (48 states from 8 episodes, *unconstrained* CEM plans clipped
only to the dataset range) reported imagined 33.3% vs none 14.6% vs random
6.2% (p = 0.0225 / 0.0002): large model-guided deviations helped while large
*random* deviations actively hurt, which would mean the model's direction only
matters when big deviations are allowed. Re-running that budget on all 360
states is impractical here: with large corrections the hand-off rollouts run
to the episode cap, making each trial ~20-40 s (vs ~1.5 s), i.e. 6-12 h of
GPU for the full sweep. It is the first item in `TODO` above any larger claim.

## W3 — fine-tuning on imagined corrections

| policy | steps | closed-loop success (100 eps, paired) |
|---|---|---|
| base `smolvla_libero` | — | **61.0%** [51.2, 70.0] |
| `ft_all` (all 360 imagined corrections + 50% replay) | 3000 | 57.0% [47.2, 66.3] |
| `ft_replay` (control: replay only) | 3000 | 58.0% [48.2, 67.2] |
| `ft_early` (managed corrections from steps ≤ 50, 120 samples) | 3000 | _pending_ |

* base vs `ft_all`: +4.0pp, exact McNemar p = 0.48 (n.s.)
* base vs `ft_replay`: +3.0pp, p = 0.65 (n.s.)
* **`ft_all` vs `ft_replay`: −1.0pp, 12 win / 13 loss, p = 1.0**

Fine-tuning perturbs the policy slightly (both runs lose ~3-4pp, not
significant); the imagined corrections add **no measurable benefit** over a
replay-only control trained with the same budget. Training-side evidence for
the correction data:

* synthetic (correction) loss falls from 3.85 to ~0.55 over 3000 steps,
  replay loss stays at ~0.04 — the policy *does* learn the corrections
* but what it learns is not worth more than random equal-budget perturbations
  (W2), which is why closed-loop success does not move.

Dataset filters built from the same corrections (in `runs/main`):
`all` 360 / `gained` (WM score) / `lifted` (simulator-verified).

## Bottom line

For a strong VLA on LIBERO-Spatial, neither *selecting* among imagined
candidates (previous work) nor *training on* imagined corrections (this work)
turns world-model imagination into actionable value:

1. Corrections help only slightly, and **only as much as a random perturbation
   of the same magnitude** — the model's direction is not where the value is.
2. Any benefit concentrates at episode starts (+15pp) and vanishes late.
3. Offline imagined progress is a weak predictor of real gain (corr +0.09).
4. Fine-tuning on the corrections matches, but does not beat, a replay-only
   control.

## Caveats

* One suite, one base policy, one fine-tuning run per condition; all claims
  are bounded by the reported intervals, and the state-level tests are paired
  but clustered (episode-level summary included).
* The world model is small (3.75M params) and trained on expert frames; larger
  video world models are out of scope here.
* The `0.5σ` trust region is a design choice; the budget sweep (W2b) exists to
  make the sensitivity to that choice explicit.
