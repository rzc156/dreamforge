"""Alias module so ``python -m dreamforge.build`` works (implementation in dataset.py)."""
from .dataset import BuildConfig, build_dataset, main  # noqa: F401

if __name__ == "__main__":
    main()
