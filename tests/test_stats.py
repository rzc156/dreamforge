"""Unit tests for the statistics helpers (no GPU, no simulator)."""
from dreamforge.stats import mcnemar_exact, paired_table, proportion_ci_95, wilson_ci


def test_wilson_basic():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61


def test_wilson_boundaries():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.4
    lo, hi = wilson_ci(10, 10)
    assert hi >= 1.0 - 1e-9 and 0.6 < lo < 0.8


def test_mcnemar_degenerate():
    # no discordant pairs must NOT be significant
    assert mcnemar_exact(0, 0) == 1.0
    # symmetric discordance -> not significant
    assert mcnemar_exact(5, 5) == 1.0


def test_mcnemar_direction_and_scale():
    assert mcnemar_exact(9, 1) < 0.05
    assert mcnemar_exact(1, 9) < 0.05
    assert mcnemar_exact(7, 3) > 0.05


def test_paired_table():
    a = {"x": True, "y": False, "z": True, "w": False}
    b = {"x": True, "y": True, "z": False, "w": False}
    tab = paired_table(a, b)
    assert tab.a_only == 1 and tab.b_only == 1
    assert tab.both_success == 1 and tab.both_fail == 1
    assert abs(tab.delta()) < 1e-12


def test_ci_string():
    s = proportion_ci_95(70, 100)
    assert s.startswith("70.0% [") and s.endswith("]")
