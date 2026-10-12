"""Report -- aggregate verify / eval artifacts into markdown + paired stats.

Usage:
    python -m dreamforge.report --run runs/main
Writes runs/main/report.md with:
  - imagined-score diagnostics (WM init vs best score)
  - verify table: none / imagined / random success with Wilson CIs and
    exact McNemar tests (imagined vs none, imagined vs random)
  - policy eval table when eval_*.json files exist (paired McNemar)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stats import mcnemar_exact, paired_table, proportion_ci_95

__all__ = ["make_report"]


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def make_report(run_dir: str, out_file: str | None = None) -> str:
    run = Path(run_dir)
    lines: list[str] = ["# dreamforge run report", ""]

    # ---------- imagine diagnostics ----------
    im_dir = run / "imagine"
    if (im_dir / "summary.json").exists():
        s = json.loads((im_dir / "summary.json").read_text())
        lines += ["## Imagined corrections (world-model side)", "",
                  f"- world model: `{s.get('wm')}`  context: `{s.get('context')}`",
                  f"- states imagined: **{s.get('n_states', 0)}**  ({s.get('minutes', 0)} min)", ""]
        if s.get("episodes"):
            lines += ["| episode file | states | WM score init -> best |", "|---|---|---|"]
            for e in s["episodes"]:
                lines.append(f"| {e['file']} | {e['states']} | {e['mean_init']:.3f} -> {e['mean_best']:.3f} |")
            lines.append("")

    # ---------- verify ----------
    v_dir = run / "verify"
    v_dirs = ([v_dir] if v_dir.exists() else []) + [p for p in sorted(run.glob("verify_*")) if p.is_dir()]
    if len(v_dirs) > 1:
        lines += ["## Budget comparison (same mined states, different correction budgets)", "",
                  "| verify dir | n | none | imagined | random | imagined-none | imagined-random |",
                  "|---|---|---|---|---|---|---|"]
        for vd in v_dirs:
            arms_d: dict[str, dict[tuple[str, int], bool]] = {}
            for arm in ("none", "imagined", "random"):
                recs = _load_jsonl(vd / f"{arm}.jsonl")
                if recs:
                    arms_d[arm] = {(r["file"], int(r["decision"])): bool(r["success"]) for r in recs}
            if "imagined" not in arms_d:
                continue
            n = len(arms_d["imagined"])
            def rate(a):
                return f"{100 * np.mean(list(arms_d[a].values())):.1f}%" if a in arms_d else "—"
            def vs(a, b):
                if a in arms_d and b in arms_d:
                    t = paired_table(arms_d[a], arms_d[b])
                    return f"{100 * t.delta():+.1f}pp (p={mcnemar_exact(t.a_only, t.b_only):.3f})"
                return "—"
            lines.append(f"| {vd.name} | {n} | {rate('none')} | {rate('imagined')} | {rate('random')} | "
                         f"{vs('imagined', 'none')} | {vs('imagined', 'random')} |")
        lines += ["", "_Budgets: `verify` = 0.5σ per-dimension trust region; `verify_big` = "
                  "unconstrained CEM (dataset-range clipping)._", ""]

    if v_dir.exists():
        per_arm: dict[str, dict[tuple[str, int], bool]] = {}
        rows: dict[tuple[str, int], dict] = {}
        for arm in ("none", "imagined", "random"):
            recs = _load_jsonl(v_dir / f"{arm}.jsonl")
            if recs:
                per_arm[arm] = {(r["file"], int(r["decision"])): bool(r["success"]) for r in recs}
                for r in recs:
                    rows.setdefault((r["file"], int(r["decision"])), {})[arm] = r
        if per_arm:
            lines += ["## Verify: do imagined corrections change outcomes?", "",
                      "| arm | n | success | 95% CI |", "|---|---|---|---|"]
            for arm, d in per_arm.items():
                k = sum(d.values())
                lines.append(f"| {arm} | {len(d)} | {k} | {proportion_ci_95(k, len(d))} |")
            lines.append("")
            for a, b in (("imagined", "none"), ("imagined", "random")):
                if a in per_arm and b in per_arm:
                    tab = paired_table(per_arm[a], per_arm[b], key="pair")
                    p = mcnemar_exact(tab.a_only, tab.b_only)
                    lines.append(f"- **{a} vs {b}**: delta = {100 * tab.delta():+.1f}pp "
                                 f"({tab.a_only} improve / {tab.b_only} worsen, "
                                 f"{tab.n_pairs} paired states), exact McNemar p = {p:.4f}")
            lines.append("")

            # --- episode-clustered view (states within an episode are correlated) ---
            if "imagined" in per_arm and "none" in per_arm:
                by_ep: dict[str, list[tuple[bool, bool]]] = {}
                for key, r in rows.items():
                    if "imagined" not in r or "none" not in r:
                        continue
                    by_ep.setdefault(r["imagined"]["file"], []).append(
                        (bool(r["imagined"]["success"]), bool(r["none"]["success"])))
                if by_ep:
                    diffs = []
                    better = worse = equal = 0
                    for v in by_ep.values():
                        d = np.mean([a for a, _ in v]) - np.mean([b for _, b in v])
                        diffs.append(d)
                        better += d > 0
                        worse += d < 0
                        equal += d == 0
                    lines += ["### Episode-clustered view (guard against pseudo-replication)", "",
                              f"- episodes with paired states: **{len(by_ep)}**",
                              f"- mean per-episode delta (imagined - none): **{100 * np.mean(diffs):+.1f}pp**",
                              f"- episodes better / equal / worse: **{better} / {equal} / {worse}**",
                              ""]

            # --- does the world model's own score predict real value? ---
            pairs = [(r["imagined"]["wm_best_score"] - r["imagined"]["wm_init_score"],
                      float(r["imagined"]["success"]) - float(r["none"]["success"]))
                     for r in rows.values() if "imagined" in r and "none" in r]
            if len(pairs) >= 5:
                gains = np.asarray([p[0] for p in pairs])
                deltas = np.asarray([p[1] for p in pairs])
                lines += ["### Does imagined progress predict real value?", "",
                          f"- paired states with both arms: **{len(pairs)}**",
                          f"- mean WM gain (best - init): {gains.mean():+.3f} "
                          f"(sd {gains.std():.3f})",
                          f"- mean real delta (imagined - none): {deltas.mean():+.3f}",
                          f"- correlation(WM gain, real delta): {np.corrcoef(gains, deltas)[0, 1]:+.3f}"]
                if (deltas != 0).any():
                    hi = gains >= np.median(gains)
                    lines.append(f"- real delta for high-WM-gain states: {deltas[hi].mean():+.3f} "
                                 f"(n={int(hi.sum())}), low: {deltas[~hi].mean():+.3f} "
                                 f"(n={int((~hi).sum())})")
                lines.append("")

            # --- breakdown by decision step (is late-stage correction easier?) ---
            if "imagined" in per_arm and "none" in per_arm:
                buckets: dict[str, list[tuple[bool, bool]]] = {}
                for key, r in rows.items():
                    if "imagined" not in r or "none" not in r:
                        continue
                    step = int(r["imagined"].get("decision_step", 0))
                    b = "start" if step == 0 else ("early" if step <= 100 else "late")
                    buckets.setdefault(b, []).append((bool(r["imagined"]["success"]), bool(r["none"]["success"])))
                if buckets:
                    lines += ["### By decision step", "",
                              "| bucket | n | imagined | none | delta |", "|---|---|---|---|---|"]
                    for b in ("start", "early", "late"):
                        if b in buckets:
                            v = buckets[b]
                            n = len(v)
                            ci = sum(x for x, _ in v) / n
                            cn = sum(y for _, y in v) / n
                            lines.append(f"| {b} | {n} | {100 * ci:.0f}% | {100 * cn:.0f}% | "
                                         f"{100 * (ci - cn):+.0f}pp |")
                    lines.append("")

    # ---------- policy eval ----------
    evals = sorted(run.glob("eval_*.json"))
    if evals:
        lines += ["## Policy evaluation (closed loop)", "", "| policy | n | success | 95% CI |", "|---|---|---|---|"]
        outcomes: dict[str, dict[tuple[str, int], bool]] = {}
        for e in evals:
            d = json.loads(e.read_text())
            key = Path(d["policy_path"]).name
            outcomes[key] = {(ep["task"], ep["episode"]): bool(ep["success"]) for ep in d["episodes"]}
            lines.append(f"| {key} | {d['n']} | {proportion_ci_95(d['successes'], d['n'])} |")
        lines.append("")
        names = list(outcomes)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                tab = paired_table(outcomes[a], outcomes[b], key="pair")
                p = mcnemar_exact(tab.a_only, tab.b_only)
                lines.append(f"- **{a} vs {b}**: delta = {100 * tab.delta():+.1f}pp, "
                             f"{tab.a_only} win / {tab.b_only} loss / {tab.discordant} discordant, "
                             f"exact McNemar p = {p:.4f}")
        lines.append("")

    text = "\n".join(lines)
    out = Path(out_file) if out_file else run / "report.md"
    out.write_text(text)
    try:  # figures are best-effort
        from .figures import make_figures

        for f in make_figures(str(run)):
            lines.append(f"![figure]({Path(f).name})")
        out.write_text("\n".join(lines))
    except Exception as exc:  # pragma: no cover
        print("figures skipped:", exc)
    print(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description="aggregate run artifacts into a report")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    make_report(args.run, args.out)


if __name__ == "__main__":
    main()
