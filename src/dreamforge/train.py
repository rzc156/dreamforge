"""Alias module so ``python -m dreamforge.train`` works (implementation in finetune.py)."""
from .finetune import FinetuneConfig, finetune, main  # noqa: F401

if __name__ == "__main__":
    main()
