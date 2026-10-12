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

Working pipeline, run end to end on LIBERO-Spatial with SmolVLA-450M and a
3.75M-parameter latent world model on a single RTX 5080. Full log and caveats:
[`docs/results.md`](docs/results.md).

| stage | result |
|---|---|
| base policy (10-episode protocol, 100 eps) | **61.0%** |
| mining | 150 episodes → 60 failures → **360 failure states** |
| `imagine` (CEM + action-sensitive WM) | imagined progress 0.480 → 0.500 (+0.020) |
| `verify`, 0.5σ corrections (360 states, paired) | none **9.7%** · imagined **13.4%** · random **14.6%** |
| imagined vs none / vs random | **+3.7pp (p = 0.060)** / **−1.1pp (p = 0.61)** |
| `ft_all` (imagined corrections + replay) | 57.0% |
| `ft_replay` (control, replay only) | 58.0% (ft_all vs ft_replay: −1.0pp, p = 1.0) |
| `ft_early` (corrections at steps ≤ 50 only) | **45.0%** (vs control: −13.0pp, p = 0.041) |

**What this says (and does not say).** World-model imagination does not
convert into actionable value for a strong policy. Corrections change outcomes
(+4.2pp over doing nothing, p = 0.032) but **no more than a random perturbation
of the same magnitude** (p = 0.61), and training on them does not beat a
replay-only control. Unexpectedly, concentrating corrections where the
decision-point metric says they help most makes the policy **worse** (58% →
45%, p = 0.041), because those states are not reliably failures and every
episode starts from one. Offline value at a state is not a training signal.

This is the honest continuation of the finding that test-time *selection* has
near-zero headroom: moving imagination from selection to training does not
rescue it. See [`docs/results.md`](docs/results.md) for the full log, the
mechanism, and the caveats.

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
