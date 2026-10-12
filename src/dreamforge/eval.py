"""Alias module so ``python -m dreamforge.eval`` works (implementation in evaluate.py)."""
from .evaluate import EvalConfig, evaluate, main  # noqa: F401

if __name__ == "__main__":
    main()
