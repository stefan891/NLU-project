# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "packaging",
# ]
# ///

"""Convert conda environment.yml to uv add ... syntax
e.g. for this environment.yml:
```yaml
name: test-env
dependencies:
- python>=3.11
- anaconda
- pip
- numpy
- pip:
  - matplotlib==2.0.0  # pin version for pip
```
You should expect this output:
```sh
$ uv run --script conda_to_uv.py environment.yml
uv add 'python>=3.11' numpy matplotlib==2.0.0
```
"""

import argparse
import sys
from pathlib import Path
import shlex

from typing import Iterable

from packaging.requirements import Requirement
import yaml

DEFAULT_IGNORE: set[str] = {"pip", "anaconda"}

def _base_name(spec: str) -> str:
    """Return the canonical, lower‑case package name from *spec* using `packaging`
    Raises
    ------
    ValueError
        If *spec* cannot be parsed by :class:`packaging.requirements.Requirement`.
    """
    try:
        return Requirement(spec).name.lower()
    except Exception as exc:  # packaging throws InvalidRequirement etc.
        raise ValueError(f"Cannot parse requirement '{spec}': {exc}") from exc

def _collect_specs(dependencies: Iterable, ignore: set[str]) -> list[str]:
    """Return requirement strings that are **not** in *ignore*."""
    collected: list[str] = []

    for dep in dependencies:
        if isinstance(dep, str):
            if _base_name(dep) not in ignore:
                collected.append(dep)
        elif isinstance(dep, dict):
            pip_section = dep.get("pip")
            if pip_section and isinstance(pip_section, list):
                for pkg in pip_section:
                    if isinstance(pkg, str) and _base_name(pkg) not in ignore:
                        collected.append(pkg)
    return collected

def convert_env_yml_to_uv_add(file_path: Path, ignore: set[str] | None = None) -> str:
    """Return a ready‑to‑run ``uv add`` command built from *file_path*."""
    ignore = set(ignore or DEFAULT_IGNORE)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        env_data = yaml.safe_load(file_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {file_path}: {exc}") from exc

    deps = env_data.get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError("'dependencies' field must be a list")

    specs = _collect_specs(deps, ignore)
    if not specs:
        return "# No Python dependencies found to convert."

    quoted = [shlex.quote(s) for s in specs]
    return "uv add " + " ".join(quoted)

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a shell‑escaped `uv add` command from a Conda environment.yml file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("env_file", type=Path, help="Path to environment.yml")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()

    try:
        cmd = convert_env_yml_to_uv_add(args.env_file)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(cmd)


if __name__ == "__main__":  # pragma: no cover
    main()