"""Alias module so ``python -m dreamforge.mine`` works (implementation in mining.py)."""
from .mining import MineConfig, mine, main  # noqa: F401

if __name__ == "__main__":
    main()
