# Contributing

Thanks for looking at `dreamforge`. This is research code, but it is kept to a
reproducibility standard: if a number appears in a document, it should be
regenerable from committed artifacts by a documented command.

## Development setup

```bash
pip install -e . --no-deps     # the package itself needs only numpy + torch
pip install pytest
pytest tests -q                # 10 tests, CPU only, no simulator
```

The full pipeline additionally needs a LeRobot + LIBERO + robosuite + MuJoCo
environment, a policy checkpoint and a world-model checkpoint; see
[`docs/protocol.md`](docs/protocol.md) and [`configs/default.yaml`](configs/default.yaml)
for the expected layout.

## What is easy to contribute

- **New statistics or diagnostics** — pure functions in `stats.py` /
  `diagnose.py` with tests. Keep them dependency-free.
- **New figures or dashboard panels** — `figures.py` / `dashboard.py`. The
  dashboard must stay self-contained (no CDN, no JavaScript).
- **A new suite or base policy** — the env layer is small (`envs.py`); add the
  suite's step cap to `_LIBERO_MAX_STEPS`.
- **Checks that would have caught a silent bug** — see the fixes in
  [`CHANGELOG.md`](CHANGELOG.md) for the class of bug we care about (OOD
  actions, double-normalized inputs, mismatched tensor shapes).

## Guidelines

1. **Keep claims measurable.** Any new metric must be computed from committed
   artifacts, and comparisons must be paired where the design allows it.
2. **Control the budget.** If you add a model-guided number, compare it against
   a random baseline of the same magnitude.
3. **Prefer exact tests.** `stats.py` avoids approximations (exact McNemar,
   Wilson) — keep it that way.
4. **Do not commit large artifacts.** `runs/`, `*.npz`, `*.pt`, `*.safetensors`
   are gitignored; put small, curated evidence in `results/`.

## Pull requests

- Run `pytest tests -q` and `python -m dreamforge.report --run <run>` locally.
- Describe *what the change lets someone claim*, not just what it changes.
- One logical change per PR; update `CHANGELOG.md` under `Unreleased`.
