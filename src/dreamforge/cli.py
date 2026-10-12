"""dreamforge CLI: mine -> imagine -> verify -> build -> train -> eval -> report."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("subcommands: mine | imagine | verify | build | train | eval | report | dashboard | diagnose | figures")
        return
    cmd, rest = argv[0], argv[1:]

    if cmd == "mine":
        from .mining import main as m
    elif cmd == "imagine":
        from .imagine import main as m
    elif cmd == "verify":
        from .verify import main as m
    elif cmd == "build":
        from .dataset import main as m
    elif cmd == "train":
        from .finetune import main as m
    elif cmd == "eval":
        from .evaluate import main as m
    elif cmd == "report":
        from .report import main as m
    elif cmd == "dashboard":
        from .dashboard import main as m
    elif cmd == "diagnose":
        from .diagnose import main as m
    elif cmd == "figures":
        from .figures import main as m
    else:
        raise SystemExit(f"unknown subcommand: {cmd}")
    sys.argv = [f"dreamforge {cmd}"] + rest
    m()


if __name__ == "__main__":
    main()
