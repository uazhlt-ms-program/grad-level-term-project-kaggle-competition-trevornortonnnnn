"""Check that the project Docker environment is usable."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path


PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "IPython": "ipython",
}


def package_version(import_name: str, distribution_name: str) -> str:
    importlib.import_module(import_name)
    return importlib.metadata.version(distribution_name)


def main() -> None:
    print("Environment check")
    print("=================")
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version}")
    print(f"Platform:          {platform.platform()}")
    print()

    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 13):
        raise RuntimeError(f"Expected Python 3.13, found Python {major}.{minor}")

    print("Installed package versions")
    print("--------------------------")
    for import_name, distribution_name in PACKAGES.items():
        version = package_version(import_name, distribution_name)
        print(f"{distribution_name:15s} {version}")

    print()
    print("Project paths")
    print("-------------")
    for path in ["data", "scripts", "reports", "models", "submissions"]:
        p = Path(path)
        print(f"{path:12s} exists={p.exists()}")

    print()
    print("Environment check passed.")


if __name__ == "__main__":
    main()