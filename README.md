# dreamforge

**Turn world-model imagination into corrective training data for robot policies.**

A single-GPU pipeline that asks a sharper question than "can a world model pick
good actions at test time?":

> Test-time imagination (selecting among the policy's own samples) has near-zero
> headroom once the policy is strong. What if we move imagination to **training
> time** — use the world model to synthesize *corrective* action labels at the
> policy's own failure states, and fine-tune on them?

```
mine     roll out a base VLA (SmolVLA) on LIBERO, record every decision point:
         observation + executed chunk + exact simulator state + episode outcome
imagine  at failure states, run CEM over a latent world model to search a
         50-step action plan that maximizes imagined task progress
verify   replay each saved simulator state and measure what the plan is worth
         in the real simulator: none vs imagined vs random, then hand back to
         the unchanged policy
build    export (observation -> imagined correction) samples; optionally keep
         only corrections that verifiably lifted the outcome
train    fine-tune the policy on imagined corrections + expert replay
eval     paired closed-loop evaluation with Wilson intervals and exact McNemar
```

The repo is built to give **honest** answers: every imagined correction can be
replayed in the real simulator, so the pipeline measures the value of
imagination instead of assuming it.

## Why this design

This project continues a previous controlled study of *test-time* action
selection for flow-matching VLAs, which found that on a strong policy only ~4%
of decision points contain a real choice, and that offline world-model scores do
not predict decision quality. That negative result leaves an obvious opening:
if imagination cannot *select*, can it *teach*?  The failure-state correction
protocol above is a minimal, falsifiable version of that question:

1. **WM-side signal** — does CEM find plans with higher imagined progress?
2. **Real-side value** — executed in the simulator, do those plans change the
   episode outcome (`imagined` vs `none`)?  Is that better than any random
   plan of the same length (`imagined` vs `random`)?
3. **Training-side value** — does fine-tuning on the corrections improve the
   policy, and which filter (all / WM-score / replay-verified) is needed?

## Install

```bash
# The heavy dependencies (lerobot, libero, robosuite, jax for pi0.5) live in
# an existing robotics conda env; this package itself only needs numpy+torch.
pip install -e . --no-deps
pytest tests -q
```

## Quickstart

```bash
RUN=runs/main
P="python -m dreamforge"

$P mine     --task libero_spatial --episodes 5 --out $RUN/mine
$P imagine  --mine $RUN/mine --out $RUN/imagine --wm base --context failure
$P verify   --mine $RUN/mine --imagine $RUN/imagine --out $RUN/verify
$P build    --mine $RUN/mine --imagine $RUN/imagine --out $RUN/build --filter lifted \
            --verify-dir $RUN/verify
$P train    --samples $RUN/build --out $RUN/ft_synth --steps 3000 --replay-ratio 0.5
$P eval     --policy-path $RUN/ft_synth --task libero_spatial --episodes 10 \
            --out $RUN/eval_ft.json
$P report   --run $RUN
```

`runs/main/report.md` collects the whole story: imagined scores, verify arms
with confidence intervals and exact McNemar tests, and policy comparison.

## Layout

```
src/dreamforge/
  wm.py        latent world model + progress head (small, CPU/GPU friendly)
  cem.py       CEM planner over imagined futures
  envs.py      LIBERO env helpers (state save/restore, raw stepping)
  policy.py    SmolVLA loading + observation/action pipelines
  mining.py    stage 1: failure-context decision points
  imagine.py   stage 2: imagined corrective plans
  verify.py    stage 3: real-simulator value of imagined plans
  dataset.py   stage 4: fine-tuning samples (all / gained / lifted filters)
  finetune.py  stage 5: SmolVLA fine-tune on corrections + replay
  evaluate.py  paired closed-loop evaluation
  report.py    markdown report with paired statistics
  stats.py     Wilson / exact McNemar helpers
```

## Status

Working pipeline, run on LIBERO-Spatial with SmolVLA-450M and a 3.75M-parameter
latent world model on a single RTX 5080. Headline numbers so far (see
[`docs/results.md`](docs/results.md) for the full log and caveats):

* base policy on the 10-episode protocol: **61.0%** (61/100 episodes)
* `imagine` (CEM, `aso` world model): imagined progress 0.480 → 0.516
  (mean gain +0.041 over 48 failure states in the pilot pass)
* `verify` (paired replay, pilot pass, n = 48 states):

  | arm | success | 95% CI |
  |---|---|---|
  | none | 14.6% | [7.2, 27.2] |
  | **imagined** | **33.3%** | [21.7, 47.5] |
  | random | 6.2% | [2.1, 16.8] |

  imagined vs none **+18.8pp, p = 0.0225**; imagined vs random **+27.1pp,
  p = 0.0002** (exact McNemar, paired on identical saved states).
* fine-tuning on imagined corrections and the paired policy evaluation are
  running; results are appended to `docs/results.md`.

## Diagnostics first

Imagination is only meaningful if the world model responds to actions. The
repo ships a pre-flight check:

```bash
python -m dreamforge diagnose --mine runs/main/mine --wms base aso
```

On this stack the unregularised checkpoint scores `std=0.0003` across random
plans (it ignores actions — the "action vacancy" phenomenon), while the
action-sensitivity checkpoint scores `std=0.0215`. All results use the latter.

## Citation / attribution

The world model, CEM interface and evaluation protocol adapt code from the
author's previous research project (`near-zero-headroom`, manuscript under
review), reused with permission.

## License

MIT
