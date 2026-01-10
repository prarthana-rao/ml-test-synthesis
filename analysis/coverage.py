import subprocess
import json
from pathlib import Path
import sys

from config.paths import (
    TARGET_REPOS_DIR,
    VENVS_DIR,
    DATA_DIR,
)

VALIDATION_REPOS = {"attrs", "jinja2", "itsdangerous"}

# ---------------------------------------------------------
# Repo → package override (explicit, intentional)
# ---------------------------------------------------------
REPO_PACKAGE_OVERRIDES = {
    "attrs": "attr",           # attrs repo exposes runtime package 'attr'
    "jinja2": "jinja2",
    "itsdangerous": "itsdangerous",
}


class CoverageError(Exception):
    pass


# ---------------------------------------------------------
# Package detection (flat + src layout)
# ---------------------------------------------------------
def detect_package_name(repo_path: Path) -> str:
    """
    Detect the primary Python package in a repository.

    Supports:
    - Flat layout: repo/<pkg>/__init__.py
    - Src layout:  repo/src/<pkg>/__init__.py

    Fails explicitly if the package cannot be uniquely determined.
    """

    candidates = []

    # Flat layout
    for p in repo_path.iterdir():
        if (
            p.is_dir()
            and (p / "__init__.py").exists()
            and p.name not in {"tests", "test"}
        ):
            candidates.append(p.name)

    # src/ layout
    src_dir = repo_path / "src"
    if src_dir.exists():
        for p in src_dir.iterdir():
            if p.is_dir() and (p / "__init__.py").exists():
                candidates.append(p.name)

    candidates = sorted(set(candidates))

    # Explicit override takes precedence
    repo_name = repo_path.name
    if repo_name in REPO_PACKAGE_OVERRIDES:
        return REPO_PACKAGE_OVERRIDES[repo_name]

    if len(candidates) != 1:
        raise CoverageError(
            f"Could not uniquely detect Python package in {repo_path}. "
            f"Found candidates: {candidates}"
        )

    return candidates[0]


# ---------------------------------------------------------
# Pytest exclusions (global, conservative)
# ---------------------------------------------------------
def pytest_args() -> list[str]:
    """
    Global pytest exclusions for environment-sensitive tests.

    - Excludes static type-checking tests (mypy / pyright)
    - Excludes deprecation-warning assertion tests
    - Does NOT affect runtime execution paths
    """
    return ["-k", "not mypy and not TestAssoc"]


# ---------------------------------------------------------
# Resolve repo-specific virtualenv python
# ---------------------------------------------------------
def resolve_venv_python(repo_name: str) -> Path:
    """
    Resolve the Python executable for the repository-specific virtualenv.
    """
    venv_python = VENVS_DIR / repo_name / "bin" / "python"

    if not venv_python.exists():
        raise CoverageError(
            f"Virtualenv not found for repo '{repo_name}': {venv_python}"
        )

    return venv_python


# ---------------------------------------------------------
# Coverage collection (ONE repo at a time)
# ---------------------------------------------------------
def collect_coverage(repo_path: Path) -> dict:
    repo_path = repo_path.resolve()

    if not repo_path.exists():
        raise CoverageError(f"Repo path does not exist: {repo_path}")

    repo_name = repo_path.name
    package_name = detect_package_name(repo_path)
    venv_python = resolve_venv_python(repo_name)

    # -----------------------------
    # Step 1: Run coverage + pytest
    # -----------------------------
    cmd = [
        str(venv_python),
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",        # ignore repo-specific configs
        f"--source={package_name}",  # restrict to Python package only
        "-m",
        "pytest",
    ]
    cmd += pytest_args()

    try:
        subprocess.run(
            cmd,
            cwd=repo_path,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise CoverageError(
            f"Coverage execution failed for repository: {repo_name}"
        )

    # -----------------------------
    # Step 2: Generate JSON report
    # -----------------------------
    try:
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "coverage",
                "json",
                "-o",
                "coverage.json",
            ],
            cwd=repo_path,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise CoverageError(
            f"Coverage JSON generation failed for repository: {repo_name}"
        )

    coverage_file = repo_path / "coverage.json"
    if not coverage_file.exists():
        raise CoverageError("coverage.json not generated")

    with open(coverage_file, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# CLI entrypoint (explicit, one-shot)
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analysis/coverage.py <repo-name>")
        print("Example: python analysis/coverage.py attrs")
        sys.exit(1)

    repo_name = sys.argv[1]

    if repo_name not in VALIDATION_REPOS:
        print(
            f"[SKIP] Coverage is only collected for validation repos.\n"
            f"Requested: {repo_name}\n"
            f"Allowed: {sorted(VALIDATION_REPOS)}"
        )
        sys.exit(0)

    repo_path = TARGET_REPOS_DIR / repo_name


    DATA_DIR.mkdir(exist_ok=True)

    try:
        coverage_data = collect_coverage(repo_path)

        output_file = DATA_DIR / f"{repo_name}_coverage.json"
        with open(output_file, "w") as f:
            json.dump(coverage_data, f, indent=2)

        print(f"[OK] Coverage saved to: {output_file}")

    except CoverageError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)
