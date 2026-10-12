# Results

Everything below is regenerated from the artifacts in `runs/main` by
`python -m dreamforge.report --run runs/main` (see `runs/main/report.md`).
A curated, committed copy of the key artifacts lives in [`results/`](results/).

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

`python -m dreamforge.diagnose --mine runs/main/mine --wms base aso full`
(12 states; artifact: `runs/main/wm_diagnosis.json`):

| world model | score std over random plans | action-swap Δprogress | latent swap norm | mean |z0| |
|---|---|---|---|---|---|
| `base` (1.39M) | 0.0003 | 0.0002 | 0.41 | 34 |
| `aso` (3.75M) | **0.0163** | **0.0101** | 17.9 | 11 |
| `full` (3.75M) | 0.0000 | 0.0003 | 14.6 | 2961 |

`base` ignores the action almost exactly (a reproduction of the *action
vacancy* phenomenon); CEM cannot optimise it. `full` has collapsed latents
(|z0| ~ 3000) and is unusable. All runs use `aso`, whose action response is
~50× larger. Takeaway, shipped as a tool: **measure action response before
trusting a world model for imagination.**

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

| policy | data | closed-loop success (100 eps, paired) |
|---|---|---|
| base `smolvla_libero` | — | **61.0%** [51.2, 70.0] |
| `ft_all` | all 360 imagined corrections + 50% replay | 57.0% [47.2, 66.3] |
| `ft_replay` | control: replay only, same budget | 58.0% [48.2, 67.2] |
| `ft_early` | corrections from steps ≤ 50 only (120 samples) | **45.0%** [35.6, 54.8] |

Paired comparisons (exact McNemar, 100 shared episodes):

* base vs `ft_all`: +4.0pp, p = 0.48 (n.s.) · base vs `ft_replay`: +3.0pp, p = 0.65 (n.s.)
* **`ft_all` vs `ft_replay`: −1.0pp, p = 1.0** — imagined data adds nothing
* base vs `ft_early`: **+16.0pp, p = 0.0037** · `ft_early` vs `ft_replay`: **−13.0pp, p = 0.041**
* `ft_all` vs `ft_early`: +12.0pp, p = 0.012

Training-side evidence:

* synthetic (correction) loss falls 3.85 → ~0.55 over 3000 steps, replay loss
  stays ~0.04 — the policy *does* learn the corrections
* but concentrating the corrections where the step breakdown says they help most
  (episode starts) makes the policy **substantially worse**.

### Why targeted corrections backfire (the informative part)

The step breakdown is a *test-time* statement: at a failing start state a
16-step correction rescues the episode ~15pp more often. It is **not** a
statement that those states are failures. Start states are the task's initial
state, and the base policy already succeeds from them in most episodes (61% of
all episodes). Imitating a perturbation there overwrites behaviour that is
mostly correct — and since every episode begins at such a state, the damage is
broad.

Formally, the mining unit is the *episode*, but the training unit is the
*state*: correcting every decision point of every failed episode includes many
states that are not reliably failures. The dose-response is visible in the
table: no correction data (`ft_replay` 58%) ≈ all corrections (`ft_all` 57%) >
corrections concentrated where they "help" (`ft_early` 45%).

This is the strongest evidence in the run for the paper's central claim, moved
from selection to training: **a signal that looks useful at the decision point
(imagined progress, +15pp rescue rate) is not useful as supervision**, because
offline value at a state does not tell you what the policy should be taught at
that state.

Dataset filters built from the same corrections (in `runs/main`):
`all` 360 / `gained` (WM score) / `lifted` (simulator-verified).

## Bottom line

For a strong VLA on LIBERO-Spatial, world-model imagination does not convert
into actionable value — neither as *selection* among candidates (previous work)
nor as *training supervision* (this work):

1. Imagined corrections do change outcomes (+4.2pp, p = 0.032) — but **only as
   much as a random perturbation of the same magnitude** (−1.1pp vs random,
   p = 0.61). The model's direction is not where the value is.
2. Improbably, corrections **hurt** when concentrated where they look most
   useful: training on early-step corrections drops closed-loop success from
   58% to 45% (p = 0.041). Offline value at a state does not tell you what to
   teach at that state.
3. Offline imagined progress is a weak predictor of real gain (corr +0.09).
4. Fine-tuning on all corrections matches, but does not beat, a replay-only
   control (p = 1.0).

The practical reading: before paying for a world model in a control loop,
(a) measure its action response, (b) measure the paired value of its output
against an equal-budget random perturbation, and (c) never assume a
decision-point metric is a training signal.

## Caveats

* One suite, one base policy, one training seed per condition; all claims are
  bounded by the reported intervals. The `ft_early` drop is significant
  (p = 0.041) but from a single seed and would benefit from replication.
* The world model is small (3.75M params) and trained on expert frames; larger
  video world models are out of scope here.
* The `0.5σ` trust region is a design choice; the unconstrained-budget sweep
  that would test its sensitivity is left as future work (W2b).
