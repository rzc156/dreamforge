"""Small, dependency-light statistics helpers for paired binary evaluations.

Kept deliberately minimal and exact:
  - Wilson score interval for a binomial rate (with correct boundary handling).
  - Exact (conditional binomial) McNemar test for paired comparisons.
  - Minimal detectable effect / required-n helper for planning.

These mirror the analysis used in the main study so results can be audited
without a statistics stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["wilson_ci", "mcnemar_exact", "PairedTable", "paired_table", "proportion_ci_95"]


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, |err| < 1.15e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (returns fractions)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion_ci_95(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


@dataclass
class PairedTable:
    """2x2 discordance table for paired binary outcomes.

    n_both_success / n_a_only / n_b_only / n_both_fail
    (a_only = A succeeded & B failed, etc.)
    """
    a_only: int
    b_only: int
    both_success: int = 0
    both_fail: int = 0

    @property
    def n_pairs(self) -> int:
        return self.a_only + self.b_only + self.both_success + self.both_fail

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only

    def delta(self) -> float:
        """Difference in success rate A - B (fraction)."""
        n = self.n_pairs
        return (self.a_only - self.b_only) / n if n else 0.0


def _log_binom_pmf(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value (conditional binomial, p=0.5).

    Handles the degenerate case: no discordant pairs -> p = 1.0
    (a naive chi-square implementation returns 0.0, which is wrong).
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    # P(X <= k) with X ~ Binomial(n, 0.5), doubled for two-sided, capped at 1.
    tail = sum(math.exp(_log_binom_pmf(i, n)) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def paired_table(a: dict, b: dict, key: str = "pair_id") -> PairedTable:
    """Build a PairedTable from two {key: bool} outcome dicts."""
    tab = PairedTable(a_only=0, b_only=0)
    common = set(a) & set(b)
    for k in common:
        av, bv = bool(a[k]), bool(b[k])
        if av and not bv:
            tab.a_only += 1
        elif bv and not av:
            tab.b_only += 1
        elif av and bv:
            tab.both_success += 1
        else:
            tab.both_fail += 1
    return tab


if __name__ == "__main__":  # tiny self-check
    assert abs(_z(0.975) - 1.959964) < 1e-4
    lo, hi = wilson_ci(70, 100)
    assert lo < 0.7 < hi
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 2) < 0.05
    print("stats self-check OK")
    print("10 vs 2 discordant -> p =", round(mcnemar_exact(10, 2), 4))
