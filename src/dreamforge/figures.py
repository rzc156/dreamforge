"""Figures for a dreamforge run (verify arms, imagined progress, policy eval).

Publication-style defaults: Okabe-Ito colourblind-safe palette, no chartjunk,
vector output.  Produces ``report_verify.png/pdf`` and ``report_eval.png/pdf``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stats import mcnemar_exact, paired_table, wilson_ci

__all__ = ["fig_verify", "fig_eval", "make_figures"]

# Okabe-Ito colourblind-safe palette
_C = {"none": "#999999", "imagined": "#0072B2", "random": "#D55E00",
      "base": "#999999", "ft_all": "#0072B2", "ft_replay": "#D55E00",
      "ft_lifted": "#009E73", "ft_gained": "#CC79A7"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
    })
    return plt


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _arm_outcomes(run: Path) -> dict[str, dict[tuple[str, int], bool]]:
    out = {}
    for arm in ("none", "imagined", "random"):
        recs = _load_jsonl(run / "verify" / f"{arm}.jsonl")
        if recs:
            out[arm] = {(r["file"], int(r["decision"])): bool(r["success"]) for r in recs}
    return out


def fig_verify(run_dir: str, out_stem: str | None = None):
    plt = _style()
    run = Path(run_dir)
    arms = _arm_outcomes(run)
    if not arms:
        raise FileNotFoundError(f"no verify records under {run}/verify")

    # imagined progress curves
    curves = []
    gains = []
    for f in sorted((run / "imagine").glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        curves.append(np.asarray(d["progress_curve"]))
        gains.append(np.asarray(d["wm_best_score"]) - np.asarray(d["wm_init_score"]))
    curves = np.concatenate(curves, axis=0) if curves else np.zeros((0, 1))
    gains = np.concatenate(gains) if gains else np.zeros(0)

    ncols = 3 if curves.size else 2
    fig, axes = plt.subplots(1, ncols, figsize=(3.3 * ncols, 2.9))
    ax = axes[0]
    names = [a for a in ("none", "imagined", "random") if a in arms]
    rates = [np.mean(list(arms[a].values())) for a in names]
    err = []
    for a in names:
        v = list(arms[a].values())
        lo, hi = wilson_ci(int(sum(v)), len(v))
        err.append([rates[names.index(a)] - lo, hi - rates[names.index(a)]])
    err = np.asarray(err).T
    ax.bar(names, rates, color=[_C[a] for a in names], width=0.6)
    ax.errorbar(names, rates, yerr=err, fmt="none", ecolor="k", capsize=3, lw=1)
    for i, a in enumerate(names):
        ax.text(i, min(1.0, rates[i] + err[1][i] + 0.03), f"n={len(arms[a])}",
                ha="center", fontsize=8)
    if "imagined" in arms and "none" in arms:
        t = paired_table(arms["imagined"], arms["none"])
        p = mcnemar_exact(t.a_only, t.b_only)
        ax.set_title(f"real value of corrections\nimagined vs none p={p:.3f}")
    ax.set_ylabel("episode success after hand-off")
    ax.set_ylim(0, 1.05)

    if curves.size:
        ax = axes[1]
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0)
        x = np.arange(len(mean))
        ax.fill_between(x, mean - sd, mean + sd, color=_C["imagined"], alpha=0.2, lw=0)
        ax.plot(x, mean, color=_C["imagined"], lw=1.8)
        ax.set_xlabel("imagined step")
        ax.set_ylabel("imagined progress")
        ax.set_title(f"CEM improves imagined progress\nmean gain {gains.mean():+.3f}")
        ax.set_ylim(0, 1)

    # scatter: WM gain vs real delta
    idx = 2 if curves.size else 1
    if idx < len(axes) and "imagined" in arms and "none" in arms:
        ax = axes[idx]
        g, dlt = [], []
        for k in set(arms["imagined"]) & set(arms["none"]):
            rec = next(r for r in _load_jsonl(run / "verify" / "imagined.jsonl")
                       if (r["file"], int(r["decision"])) == k)
            g.append(rec["wm_best_score"] - rec["wm_init_score"])
            dlt.append(float(arms["imagined"][k]) - float(arms["none"][k]))
        g, dlt = np.asarray(g), np.asarray(dlt)
        rng = np.random.default_rng(0)
        ax.scatter(g, dlt + rng.normal(0, 0.02, len(dlt)), s=14, alpha=0.6,
                   color=_C["imagined"], edgecolor="none")
        if len(g) > 2 and g.std() > 0 and dlt.std() > 0:
            ax.set_title(f"model score vs real value\nr = {np.corrcoef(g, dlt)[0, 1]:+.2f}")
        ax.set_xlabel("world-model gain")
        ax.set_ylabel("real outcome delta")
        ax.set_yticks([-1, 0, 1])

    stem = out_stem or str(run / "report_verify")
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    return stem + ".png"


def fig_eval(run_dir: str, out_stem: str | None = None):
    plt = _style()
    run = Path(run_dir)
    evals = sorted(run.glob("eval_*.json"))
    if not evals:
        return None
    outcomes = {}
    for e in evals:
        d = json.loads(e.read_text())
        name = Path(d["policy_path"]).name
        outcomes[name] = {(ep["task"], ep["episode"]): bool(ep["success"]) for ep in d["episodes"]}

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    names = list(outcomes)
    rates, err = [], []
    for n in names:
        v = list(outcomes[n].values())
        r = np.mean(v)
        lo, hi = wilson_ci(int(sum(v)), len(v))
        rates.append(r)
        err.append([r - lo, hi - r])
    err = np.asarray(err).T
    ax.bar(names, rates, color=[_C.get(n, "#0072B2") for n in names], width=0.6)
    ax.errorbar(names, rates, yerr=err, fmt="none", ecolor="k", capsize=3, lw=1)
    for i, n in enumerate(names):
        ax.text(i, min(1.02, rates[i] + err[1][i] + 0.03), f"n={len(outcomes[n])}",
                ha="center", fontsize=8)
    ax.set_ylabel("closed-loop success")
    ax.set_ylim(0, 1.05)
    ax.set_title("policy evaluation (paired seeds)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    stem = out_stem or str(run / "report_eval")
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    return stem + ".png"


def make_figures(run_dir: str) -> list[str]:
    made = []
    try:
        made.append(fig_verify(run_dir))
    except FileNotFoundError as exc:
        print("verify figure skipped:", exc)
    p = fig_eval(run_dir)
    if p:
        made.append(p)
    return made


def main() -> None:
    p = argparse.ArgumentParser(description="render run figures")
    p.add_argument("--run", required=True)
    args = p.parse_args()
    for f in make_figures(args.run):
        print("wrote", f)


if __name__ == "__main__":
    main()
