"""Backward-compatible CLI wrapper for :mod:`ccus_gym.cli.batch_mappo`."""

from pathlib import Path
import sys

PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

from ccus_gym.cli.batch_mappo import main


if __name__ == "__main__":
    main()
