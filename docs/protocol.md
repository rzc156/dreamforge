# Experimental protocol

## Question

Given a pretrained VLA that already solves ~60-70% of LIBERO-Spatial, and a
latent world model that is *not* strong enough to select actions at test time
(previous study), can world-model imagination still produce **training data**
that makes the policy better?

Three nested claims are tested in order, so a failure at any level is
still an interpretable result:

| level | claim | measurement | artifact |
|---|---|---|---|
| L1 | CEM finds plans with higher imagined progress | `wm_best_score > wm_init_score` | `imagine/summary.json` |
| L2 | imagined plans have real value | success after replay: `imagined` vs `none` vs `random` | `verify/*.jsonl` |
| L3 | fine-tuning on imagined corrections helps | closed-loop success of fine-tuned vs base policy, paired by (task, seed) | `eval_*.json` |

## Mining

* Base policy: `lerobot/smolvla_libero` (SmolVLA-450M), evaluated exactly as in
  the previous study (chunk = 50, two cameras, 8-D state).
* Suite: `libero_spatial` (10 tasks). Episodes: N per task, seeds fixed.
* At every decision point (chunk boundary) we store: the observation the policy
  saw (both cameras), the executed chunk in **dataset raw action space**, the
  flattened MuJoCo state, and the episode outcome.
* `sim_state` allows exact replay later: `MjSimState.from_flattened` +
  `sim.set_state` + `sim.forward()`.

## Imagination (CEM)

* Latent world model: encoder 96×96 agent view → z (128-d); MLP transition on
  `(z, a)`; progress head `z → p ∈ [0,1]` trained on normalized episode time.
* Search: CEM in action space, horizon 50 (matching the policy chunk), seeded
  with the base policy's own chunk, actions clipped to dataset min/max.
* Score: mean progress over the last 3 imagined steps.
* Saved per state: best plan, best/initial score, imagined progress curve,
  CEM history, plus a persisted uniform-random plan of the same length (the
  `random` control is therefore identical across reruns).

## Verification

Arms (same saved state, same hand-off policy, same step budget):

* `none` — hand back immediately; should reproduce the original failure.
* `imagined` — execute the first `exec_horizon=16` steps of the CEM plan, then
  hand back.
* `random` — execute a uniform-random plan of the same length, then hand back.

Details that matter for validity:

* The env's internal `timestep` is set to the mined step index, so the
  remaining episode budget matches the original episode.
* The hand-off policy is the unchanged base policy; torch RNG is re-seeded
  per trial so arms are comparable.
* `none` vs `imagined` is a paired comparison over identical states; the
  analysis uses exact McNemar on the discordant pairs.

## Dataset & fine-tuning

* Samples are `(observation at failure state → imagined corrective chunk)`.
* Filters:
  * `all` — every correction (pure imagination, no simulator);
  * `gained` — WM's own score says the plan is better (`Δ ≥ 0.05`, no simulator);
  * `lifted` — the real replay says the plan changed the outcome (uses the
    simulator, i.e. the expensive upper bound).
* Fine-tuning: SmolVLA, flow-matching loss, batch 16-32, lr 5e-6, mixed with
  expert replay from `lerobot/libero`. Control run: replay only, same budget.
* Evaluation: paired closed-loop on the same suite/tasks/seeds; Wilson CIs and
  exact McNemar; per-task breakdown.

## Caveats

* Single suite, single base policy, single seed per configuration in the first
  pass; numbers are reported with intervals, and claims are limited to what the
  intervals support.
* The world model is small (1.4-3.8M parameters) and was trained on expert
  frames; it is not a video model.
* `lifted` filtering uses the simulator, so it is not a deployable recipe by
  itself — it bounds how much of the signal imagination *could* carry.
