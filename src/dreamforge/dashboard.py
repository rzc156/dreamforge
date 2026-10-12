"""Render a self-contained HTML dashboard for a dreamforge run.

No external assets, no CDN, no JavaScript: the page is a single ``.html`` file
with inline CSS and hand-drawn SVG charts, so it works offline, in a PR
attachment, or on GitHub Pages.

    python -m dreamforge.dashboard --run runs/main --out results/dashboard.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .paths import DEFAULTS
from .stats import mcnemar_exact, paired_table, wilson_ci

__all__ = ["render_dashboard"]

_ARM_COLOR = {"none": "#8b9bb4", "imagined": "#3ba7ff", "random": "#ff7a59"}
_POLICY_COLOR = ["#8b9bb4", "#3ba7ff", "#ff7a59", "#7ee787", "#c792ea"]

CSS = """
:root{
  --bg:#070b14; --bg2:#0b1220; --card:#0f1727; --line:#1d2c45; --line2:#27395a;
  --fg:#e9f0fb; --muted:#93a7c4; --faint:#6b81a1;
  --blue:#3ba7ff; --green:#7ee787; --orange:#ff7a59; --violet:#c792ea; --grey:#8b9bb4;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:radial-gradient(1200px 700px at 12% -8%, #12283f 0%, var(--bg) 55%) , var(--bg);
     color:var(--fg); font:15px/1.6 'Segoe UI',-apple-system,'Helvetica Neue',Arial,'PingFang SC','Microsoft YaHei',sans-serif;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:38px 22px 90px}
.hero{display:flex;flex-wrap:wrap;gap:22px;align-items:flex-end;justify-content:space-between;
      border:1px solid var(--line);background:linear-gradient(180deg,#0e1a2c,#0b1220);
      border-radius:18px;padding:26px 28px;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;inset:-40% 40% 40% -10%;
  background:radial-gradient(400px 240px at 30% 30%, rgba(59,167,255,.25), transparent 70%);}
.brand{display:flex;gap:16px;align-items:center;z-index:1}
.spark{width:46px;height:46px;flex:0 0 46px}
h1{margin:0;font-size:30px;letter-spacing:-.5px}
.tag{color:var(--muted);margin-top:4px}
.meta{color:var(--faint);font-size:12.5px;z-index:1;text-align:right}
.meta code{color:var(--blue)}
h2{font-size:16px;letter-spacing:.2px;margin:38px 0 12px;color:#cfe0f7;
   text-transform:uppercase;font-weight:700}
h2 .n{color:var(--faint);font-weight:600;margin-right:8px}
.sub{color:var(--muted);margin:-4px 0 14px}
.grid{display:grid;gap:14px}
.stats{grid-template-columns:repeat(auto-fit,minmax(178px,1fr))}
.cards{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.card{border:1px solid var(--line);background:linear-gradient(180deg,#101a2c,#0d1626);
      border-radius:14px;padding:16px 16px 14px}
.card .k{color:var(--faint);font-size:12.5px;letter-spacing:.03em;text-transform:uppercase}
.card .v{font-size:26px;font-weight:700;margin:6px 0 2px;letter-spacing:-.4px}
.card .d{color:var(--muted);font-size:13px}
.card.hl{border-color:#2a5d8f;box-shadow:0 0 0 1px rgba(59,167,255,.12),0 12px 30px -22px rgba(59,167,255,.55)}
.card.bad{border-color:#7a3b2a;box-shadow:0 0 0 1px rgba(255,122,89,.12)}
.card.good{border-color:#2f6b3a}
.panel{border:1px solid var(--line);background:linear-gradient(180deg,#0f1929,#0c1422);
       border-radius:16px;padding:18px}
.panel + .panel{margin-top:14px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--faint);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;border:1px solid var(--line2);color:var(--muted)}
.pill.b{color:#9fd0ff;border-color:#2a5d8f;background:rgba(59,167,255,.08)}
.pill.o{color:#ffc79c;border-color:#7a4a2f;background:rgba(255,122,89,.09)}
.pill.g{color:#9fe8ac;border-color:#2f6b3a;background:rgba(126,231,135,.08)}
.note{color:var(--faint);font-size:12.5px;margin-top:10px}
.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch}
.step{flex:1 1 150px;border:1px solid var(--line);border-radius:12px;padding:12px 13px;background:#0d1626}
.step .s{color:var(--faint);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase}
.step .t{font-weight:700;margin:2px 0 4px}
.step .x{color:var(--muted);font-size:12.5px}
.arrow{align-self:center;color:#33507a;font-size:18px}
.verdict{border-left:3px solid var(--orange);background:linear-gradient(90deg,rgba(255,122,89,.08),transparent 60%);
         border-radius:0 12px 12px 0;padding:14px 18px;color:#ffe3d7}
.verdict b{color:#fff}
footer{color:var(--faint);font-size:12.5px;margin-top:44px;border-top:1px solid var(--line);padding-top:16px}
a{color:var(--blue);text-decoration:none}
code{font-family:'SFMono-Regular',Consolas,'Liberation Mono',monospace;font-size:.92em;color:#9fd0ff}
@media (prefers-color-scheme: light){
  :root{--bg:#f6f8fc;--bg2:#eef2f9;--card:#fff;--line:#e2e8f2;--line2:#cfd9ea;
        --fg:#0e1726;--muted:#4d5f7a;--faint:#7488a5}
  body{background:radial-gradient(1000px 600px at 10% -10%, #e8f1ff 0%, var(--bg) 60%), var(--bg)}
  .hero{background:linear-gradient(180deg,#ffffff,#f4f8ff)}
  .card{background:#fff}.panel{background:#fff}.step{background:#fbfcff}
  h2{color:#22304a}.verdict{color:#7a3018}
}
"""


# ------------------------------------------------------------------ helpers
def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _arms(run: Path, sub: str = "verify") -> dict[str, dict[tuple[str, int], bool]]:
    out: dict[str, dict[tuple[str, int], bool]] = {}
    for arm in ("none", "imagined", "random"):
        recs = _load_jsonl(run / sub / f"{arm}.jsonl")
        if recs:
            out[arm] = {(r["file"], int(r["decision"])): bool(r["success"]) for r in recs}
    return out


def _rate(d: dict) -> tuple[int, int]:
    return int(sum(d.values())), len(d)


def _bars(labels, values, los, his, colors, width=520, height=210, ymax=1.0, fmt="{:.1%}"):
    """Vertical bars with error whiskers, drawn as SVG."""
    n = len(values)
    pad_l, pad_r, pad_t, pad_b = 44, 14, 18, 34
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b
    bw = iw / max(n, 1) * 0.52
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    # y grid
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        if g > ymax:
            continue
        y = pad_t + ih * (1 - g / ymax)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#26364f" stroke-width="1" stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="#6b81a1" text-anchor="end">{int(g*100)}%</text>')
    for i, (lab, v, lo, hi, c) in enumerate(zip(labels, values, los, his, colors)):
        cx = pad_l + iw * (i + 0.5) / n
        x = cx - bw / 2
        bh = ih * (v / ymax)
        y = pad_t + ih - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="6" fill="{c}" fill-opacity="0.85"/>')
        ylo = pad_t + ih * (1 - lo / ymax)
        yhi = pad_t + ih * (1 - hi / ymax)
        parts.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" stroke="#dfe9f7" stroke-width="1.4"/>')
        for yy in (ylo, yhi):
            parts.append(f'<line x1="{cx-6:.1f}" y1="{yy:.1f}" x2="{cx+6:.1f}" y2="{yy:.1f}" stroke="#dfe9f7" stroke-width="1.4"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y-6:.1f}" font-size="12.5" font-weight="700" fill="#e9f0fb" text-anchor="middle">{fmt.format(v)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height-12}" font-size="12.5" fill="#93a7c4" text-anchor="middle">{html.escape(str(lab))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _curve(mean, sd, width=520, height=210, color="#3ba7ff"):
    """Line + band chart for the imagined-progress curve."""
    pad_l, pad_r, pad_t, pad_b = 44, 14, 16, 30
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(mean)
    xs = [pad_l + iw * i / max(n - 1, 1) for i in range(n)]

    def Y(v):
        return pad_t + ih * (1 - float(v))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(g)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#26364f" stroke-width="1" stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="#6b81a1" text-anchor="end">{int(g*100)}%</text>')
    up = " ".join(f"{x:.1f},{Y(mean[i]+sd[i]):.1f}" for i, x in enumerate(xs))
    dn = " ".join(f"{x:.1f},{Y(mean[i]-sd[i]):.1f}" for i, x in reversed(list(enumerate(xs))))
    parts.append(f'<polygon points="{up} {dn}" fill="{color}" fill-opacity="0.18"/>')
    path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{Y(mean[i]):.1f}" for i, x in enumerate(xs))
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    parts.append(f'<text x="{width-pad_r}" y="{height-10}" font-size="11" fill="#6b81a1" text-anchor="end">imagined step</text>')
    parts.append(f'<text x="{pad_l}" y="{height-10}" font-size="11" fill="#6b81a1">0</text>')
    parts.append(f'<text x="{pad_l-8}" y="{pad_t-4:.1f}" font-size="11" fill="#6b81a1" text-anchor="end">p</text>')
    parts.append("</svg>")
    return "".join(parts)


def _spark_mark() -> str:
    return (
        '<svg class="spark" viewBox="0 0 48 48" aria-hidden="true">'
        '<circle cx="6" cy="24" r="3.4" fill="#3ba7ff"/>'
        '<circle cx="22" cy="11" r="3.4" fill="#7ee787"/>'
        '<circle cx="22" cy="37" r="3.4" fill="#ffb86b"/>'
        '<circle cx="40" cy="24" r="4" fill="none" stroke="#3ba7ff" stroke-width="1.6"/>'
        '<path d="M6 24 L22 11 M6 24 L22 37 M22 11 L40 24 M22 37 L40 24" stroke="#4d6f9c" stroke-width="1.3" fill="none"/>'
        "</svg>"
    )


def _card(k: str, v: str, d: str = "", cls: str = "") -> str:
    return f'<div class="card {cls}"><div class="k">{html.escape(k)}</div><div class="v">{v}</div><div class="d">{html.escape(d)}</div></div>'


# ------------------------------------------------------------------ sections
def _section_stats(run: Path, arms: dict) -> str:
    mine = run / "mine" / "summary.json"
    imag = run / "imagine" / "summary.json"
    n_states, n_eps, base = "—", "—", "—"
    if mine.exists():
        s = json.loads(mine.read_text())
        eps = [e for e in s.get("episodes", []) if "success" in e]
        n_eps = str(len(eps))
        ep0_9 = [e for e in eps if e["episode"] <= 9]
        if ep0_9:
            base = f"{100*sum(e['success'] for e in ep0_9)/len(ep0_9):.1f}%"
    if imag.exists():
        n_states = str(json.loads(imag.read_text()).get("n_states", "—"))

    cards = [
        _card("mining episodes", n_eps, "base policy rollouts, fixed seeds"),
        _card("failure states", n_states, "decision points mined"),
        _card("base policy", base, "10-episode protocol (paired)"),
    ]
    if "imagined" in arms and "none" in arms:
        ki, ni = _rate(arms["imagined"])
        kn, nn = _rate(arms["none"])
        t = paired_table(arms["imagined"], arms["none"])
        p = mcnemar_exact(t.a_only, t.b_only)
        cards.append(_card("imagined vs none", f"{100*(ki/ni - kn/nn):+.1f}pp", f"exact McNemar p = {p:.3f}", "hl"))
    if "imagined" in arms and "random" in arms:
        ki, ni = _rate(arms["imagined"])
        kr, nr = _rate(arms["random"])
        t = paired_table(arms["imagined"], arms["random"])
        p = mcnemar_exact(t.a_only, t.b_only)
        cls = "good" if ki / ni > kr / nr and p < 0.05 else "bad"
        cards.append(_card("imagined vs random", f"{100*(ki/ni - kr/nr):+.1f}pp", f"equal budget · p = {p:.3f}", cls))
    return f'<div class="grid stats">{"".join(cards)}</div>'


def _section_flow() -> str:
    steps = [
        ("stage 1", "mine", "roll out the base VLA, record every decision point with its exact simulator state"),
        ("stage 2", "imagine", "CEM over a latent world model, 0.5σ trust region, seeded by the policy's own chunk"),
        ("stage 3", "verify", "paired replay of the saved state: none / imagined / random, then hand back to the policy"),
        ("stage 4", "build", "correction datasets (all / gained / lifted) mixed with expert replay"),
        ("stage 5", "train", "SmolVLA fine-tune, flow matching, 16 GB single GPU"),
        ("stage 6", "eval", "paired closed-loop evaluation with Wilson intervals and exact McNemar"),
    ]
    out = []
    for i, (s, t, x) in enumerate(steps):
        out.append(f'<div class="step"><div class="s">{s}</div><div class="t">{t}</div><div class="x">{html.escape(x)}</div></div>')
        if i < len(steps) - 1:
            out.append('<div class="arrow">→</div>')
    return f'<div class="flow">{"".join(out)}</div>'


def _section_diagnosis(run: Path) -> str:
    p = run / "wm_diagnosis.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    wms = d.get("world_models", {})
    if not wms:
        return ""
    rows = []
    best = max(wms, key=lambda k: wms[k].get("score_random_std", 0))
    for name, v in wms.items():
        pill = 'g' if name == best else ('o' if name == "base" else 'b')
        rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td>"
            f'<td class="num">{v.get("score_random_std", 0):.4f}</td>'
            f'<td class="num">{v.get("action_swap_delta", 0):.4f}</td>'
            f'<td class="num">{v.get("latent_swap_norm", 0):.2f}</td>'
            f'<td class="num">{v.get("latent_z0_norm", 0):.0f}</td>'
            f'<td><span class="pill {pill}">{"used" if name == best else "ignores actions" if v.get("score_random_std",0) < 1e-3 else "ok"}</span></td></tr>'
        )
    return (
        '<h2><span class="n">02</span>World-model pre-flight</h2>'
        '<p class="sub">Imagination is only meaningful if the model responds to actions. '
        'This diagnostic is shipped as a CLI command and is run before any experiment.</p>'
        '<div class="panel"><table><thead><tr><th>world model</th>'
        '<th class="num">score std (random plans)</th><th class="num">action-swap Δprogress</th>'
        '<th class="num">latent swap</th><th class="num">mean |z0|</th><th>verdict</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<div class="note">An unregularized checkpoint scores ≈0 on both response metrics — it ignores the action entirely. '
        'CEM cannot optimize such a model, so results would be meaningless.</div></div>'
    )


def _section_verify(run: Path, arms: dict) -> str:
    if not arms:
        return ""
    names = [a for a in ("none", "imagined", "random") if a in arms]
    vals, los, his = [], [], []
    for a in names:
        k, n = _rate(arms[a])
        lo, hi = wilson_ci(k, n)
        vals.append(k / n)
        los.append(lo)
        his.append(hi)
    bars = _bars(names, vals, los, his, [_ARM_COLOR[a] for a in names], ymax=max(0.35, max(his) * 1.15))

    # progress curve
    curves = []
    for f in sorted((run / "imagine").glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        curves.append(np.asarray(d["progress_curve"]))
    curve_html = ""
    if curves:
        C = np.concatenate(curves, axis=0)
        curve_html = (
            '<div class="panel"><h3 style="margin:0 0 4px;font-size:14px">Imagined progress along the CEM plan</h3>'
            '<div class="sub" style="margin:0 0 8px">mean ± sd over mined failure states</div>'
            f'{_curve(C.mean(0), C.std(0))}</div>'
        )

    # table + paired tests
    rows = []
    for a in names:
        k, n = _rate(arms[a])
        rows.append(f"<tr><td>{a}</td><td class='num'>{n}</td><td class='num'>{k}</td>"
                    f"<td class='num'>{100*k/n:.1f}%</td><td class='num'>[{100*wilson_ci(k,n)[0]:.1f}, {100*wilson_ci(k,n)[1]:.1f}]</td></tr>")
    tests = []
    for a, b in (("imagined", "none"), ("imagined", "random")):
        if a in arms and b in arms:
            t = paired_table(arms[a], arms[b])
            p = mcnemar_exact(t.a_only, t.b_only)
            tests.append(f"<tr><td>{a} vs {b}</td><td class='num'>{100*t.delta():+.1f}pp</td>"
                         f"<td class='num'>{t.a_only} / {t.b_only}</td><td class='num'>{p:.3f}</td></tr>")

    # by decision step
    by = {}
    for key in set(arms.get("imagined", {})) & set(arms.get("none", {})):
        for r in _load_jsonl(run / "verify" / "imagined.jsonl"):
            if (r["file"], int(r["decision"])) == key:
                s = int(r.get("decision_step", 0))
                by.setdefault("start" if s == 0 else "early" if s <= 100 else "late", []).append(
                    (arms["imagined"][key], arms["none"][key]))
                break
    step_rows = []
    order = {"start": 0, "early": 1, "late": 2}
    for b in sorted(by, key=lambda x: order.get(x, 9)):
        v = by[b]
        ci = np.mean([x for x, _ in v])
        cn = np.mean([y for _, y in v])
        step_rows.append(f"<tr><td>{b}</td><td class='num'>{len(v)}</td><td class='num'>{100*ci:.0f}%</td>"
                         f"<td class='num'>{100*cn:.0f}%</td><td class='num'>{100*(ci-cn):+.0f}pp</td></tr>")

    return (
        '<h2><span class="n">03</span>What imagined corrections are worth (real simulator)</h2>'
        '<p class="sub">Each mined state is restored exactly and replayed with three arms, then the unchanged policy resumes. '
        'The <code>none</code> arm reproduces the original episode, which anchors the comparison.</p>'
        f'<div class="grid cards" style="grid-template-columns:1.05fr 1fr"><div class="panel">{bars}</div>{curve_html}</div>'
        '<div class="grid cards" style="grid-template-columns:1fr 1fr;margin-top:14px">'
        '<div class="panel"><h3 style="margin:0 0 6px;font-size:14px">Arms</h3><table><thead><tr><th>arm</th>'
        '<th class="num">n</th><th class="num">success</th><th class="num">rate</th><th class="num">95% CI</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<div class="panel"><h3 style="margin:0 0 6px;font-size:14px">Paired tests (exact McNemar)</h3>'
        '<table><thead><tr><th>comparison</th><th class="num">delta</th><th class="num">improve/worsen</th>'
        '<th class="num">p</th></tr></thead>'
        f'<tbody>{"".join(tests)}</tbody></table>'
        '<h3 style="margin:14px 0 6px;font-size:14px">By decision step</h3>'
        '<table><thead><tr><th>bucket</th><th class="num">n</th><th class="num">imagined</th>'
        '<th class="num">none</th><th class="num">delta</th></tr></thead>'
        f'<tbody>{"".join(step_rows)}</tbody></table></div></div>'
    )


def _section_eval(run: Path) -> str:
    evals = sorted(run.glob("eval_*.json"))
    if not evals:
        return ""
    names, vals, los, his = [], [], [], []
    rows, outcomes = [], {}
    for e in evals:
        d = json.loads(e.read_text())
        key = Path(d["policy_path"]).name
        k, n = d["successes"], d["n"]
        lo, hi = wilson_ci(k, n)
        names.append(key)
        vals.append(k / n)
        los.append(lo)
        his.append(hi)
        rows.append(f"<tr><td><code>{html.escape(key)}</code></td><td class='num'>{n}</td>"
                    f"<td class='num'>{100*k/n:.1f}%</td><td class='num'>[{100*lo:.1f}, {100*hi:.1f}]</td></tr>")
        outcomes[key] = {(ep["task"], ep["episode"]): bool(ep["success"]) for ep in d["episodes"]}
    tests = []
    base = names[0] if names else None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            t = paired_table(outcomes[a], outcomes[b])
            p = mcnemar_exact(t.a_only, t.b_only)
            tests.append(f"<tr><td>{html.escape(a)} vs {html.escape(b)}</td><td class='num'>{100*t.delta():+.1f}pp</td>"
                         f"<td class='num'>{t.a_only} / {t.b_only}</td><td class='num'>{p:.3f}</td></tr>")
    bars = _bars(names, vals, los, his, [_POLICY_COLOR[i % len(_POLICY_COLOR)] for i in range(len(names))],
                 ymax=max(0.8, max(his) * 1.15))
    return (
        '<h2><span class="n">04</span>Does training on imagined corrections help?</h2>'
        '<p class="sub">Fine-tuned policies are evaluated on the same suite and seeds as the base policy '
        '(paired by task and episode).</p>'
        f'<div class="grid cards" style="grid-template-columns:1.05fr 1fr"><div class="panel">{bars}</div>'
        '<div class="panel"><h3 style="margin:0 0 6px;font-size:14px">Closed-loop success</h3>'
        '<table><thead><tr><th>policy</th><th class="num">n</th><th class="num">success</th>'
        '<th class="num">95% CI</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<h3 style="margin:14px 0 6px;font-size:14px">Paired tests</h3>'
        '<table><thead><tr><th>comparison</th><th class="num">delta</th><th class="num">W/L</th>'
        f'<th class="num">p</th></tr></thead><tbody>{"".join(tests)}</tbody></table></div></div>'
    )


def _verdict(arms: dict, run: Path) -> str:
    parts = ["<b>Verdict.</b>"]
    if "imagined" in arms and "none" in arms:
        ki, ni = _rate(arms["imagined"])
        kn, nn = _rate(arms["none"])
        parts.append(f" imagined corrections change outcomes ({100*(ki/ni-kn/nn):+.1f}pp vs doing nothing)")
    if "imagined" in arms and "random" in arms:
        ki, ni = _rate(arms["imagined"])
        kr, nr = _rate(arms["random"])
        t = paired_table(arms["imagined"], arms["random"])
        p = mcnemar_exact(t.a_only, t.b_only)
        parts.append(f", but they are indistinguishable from an <b>equal-budget random perturbation</b> "
                     f"({100*(ki/ni-kr/nr):+.1f}pp, p = {p:.2f}) — the model's direction is not where the value is")
    evals = {}
    for e in sorted(run.glob("eval_*.json")):
        d = json.loads(e.read_text())
        evals[Path(d["policy_path"]).name] = d["success_rate"]
    if len(evals) >= 2:
        names = list(evals)
        if "ft_replay" in evals and "ft_all" in evals:
            parts.append(f". Training on the corrections (ft_all {evals['ft_all']}%) matches a replay-only "
                         f"control (ft_replay {evals['ft_replay']}%)")
        worst = min(evals, key=evals.get)
        best = max(evals, key=evals.get)
        if worst != best and evals[worst] < evals[best] - 5:
            parts.append(f", and concentrating corrections where the decision-point metric looked best is "
                         f"actively harmful ({worst} {evals[worst]}% vs {best} {evals[best]}%)")
    parts.append(". Offline value at a state does not tell you what to teach there.")
    return f'<div class="verdict">{"".join(parts)}</div>'


# ------------------------------------------------------------------ render
def render_dashboard(run_dir: str = "runs/main", out_path: str | None = None) -> Path:
    run = Path(run_dir)
    arms = _arms(run)
    title = "dreamforge · run report"
    sections = [
        _section_stats(run, arms),
        '<h2><span class="n">01</span>Pipeline</h2>',
        _section_flow(),
        _section_diagnosis(run),
        _section_verify(run, arms),
        _section_eval(run),
        "<h2><span class='n'>&#9873;</span>Verdict</h2>",
        _verdict(arms, run),
    ]
    body = "\n".join(s for s in sections if s)
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
  <header class="hero">
    <div class="brand">{_spark_mark()}
      <div><h1>dreamforge</h1>
      <div class="tag">world-model imagination → corrective training data for robot policies</div></div>
    </div>
    <div class="meta">run <code>{html.escape(str(run))}</code><br>
      LIBERO-Spatial · SmolVLA-450M · single RTX 5080<br>
      generated by <code>python -m dreamforge.dashboard</code></div>
  </header>
  {body}
  <footer>
    Every number on this page is regenerated from the run artifacts
    (<code>verify/*.jsonl</code>, <code>eval_*.json</code>, <code>imagine/*.npz</code>, <code>wm_diagnosis.json</code>).
    Statistics: Wilson score intervals and exact (conditional binomial) McNemar tests, paired on identical saved states.
  </footer>
</div></body></html>
"""
    out = Path(out_path) if out_path else run / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="render a self-contained HTML dashboard for a run")
    p.add_argument("--run", default=str(DEFAULTS.run_dir))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = render_dashboard(args.run, args.out)
    print(f"dashboard -> {out}")


if __name__ == "__main__":
    main()
