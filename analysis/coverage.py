import subprocess
import json
from pathlib import Path
import sys
import os

from config.paths import TARGET_REPOS_DIR, VENVS_DIR, DATA_DIR


CI_MODE = os.getenv("CI_MODE") == "1"
CI_WORKSPACE = Path(os.getenv("CI_WORKSPACE", DATA_DIR))
TARGET_REPO = Path(os.getenv("TARGET_REPO", TARGET_REPOS_DIR))


VALIDATION_REPOS = {"attrs", "jinja2", "itsdangerous"}

REPO_PACKAGE_OVERRIDES = {
    "attrs": "attr",
    "jinja2": "jinja2",
    "itsdangerous": "itsdangerous",
}


class CoverageError(Exception):
    pass


# ---------------------------------------------------------
# Package detection
# ---------------------------------------------------------
def detect_package_name(repo_path: Path) -> str:
    candidates = []

    for p in repo_path.iterdir():
        if p.is_dir() and (p / "__init__.py").exists() and p.name not in {"tests", "test"}:
            candidates.append(p.name)

    src_dir = repo_path / "src"
    if src_dir.exists():
        for p in src_dir.iterdir():
            if p.is_dir() and (p / "__init__.py").exists():
                candidates.append(p.name)

    repo_name = repo_path.name
    if repo_name in REPO_PACKAGE_OVERRIDES:
        return REPO_PACKAGE_OVERRIDES[repo_name]

    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise CoverageError(f"Could not uniquely detect package. Found: {candidates}")

    return candidates[0]


# ---------------------------------------------------------
# Resolve python (venv for research, CI python otherwise)
# ---------------------------------------------------------
def resolve_python(repo_name: str) -> Path:
    if CI_MODE:
        return Path(sys.executable)

    py = VENVS_DIR / repo_name / "bin" / "python"
    if not py.exists():
        raise CoverageError(f"Missing repo venv python: {py}")
    return py

def pytest_args() -> list[str]:
    return ["-k", "not mypy and not TestAssoc"]


# ---------------------------------------------------------
# Coverage execution
# ---------------------------------------------------------
def collect_coverage(repo_path: Path, python_exec: Path) -> dict:
    package = detect_package_name(repo_path)

    cmd = [
        str(python_exec),
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",
        f"--source={package}",
        "-m",
        "pytest",
    ]
    cmd += pytest_args()

    try:
        subprocess.run(cmd, cwd=repo_path, check=True)
    except subprocess.CalledProcessError:
        raise CoverageError("Coverage run failed")

    out_dir = CI_WORKSPACE / "coverage" if CI_MODE else repo_path
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "coverage.json"

    try:
        subprocess.run(
            [str(python_exec), "-m", "coverage", "json", "-o", str(json_out)],
            cwd=repo_path,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise CoverageError("Coverage JSON failed")

    with open(json_out, "r") as f:
        data = json.load(f)
    
    with open(json_out, "w") as f:
        json.dump(data, f, indent=2)

    return data 


# ---------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---------------- CI MODE ----------------
    # --- inside __main__ CI block replace repo resolution only ---

    if CI_MODE:
        repo = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET_REPO
        py = Path(sys.executable)

        try:
            cov = collect_coverage(repo, py)
            print("[OK] CI coverage collected")
        except CoverageError as e:
            print(f"[ERROR] {e}")
            sys.exit(2)

        sys.exit(0)

    # ---------------- RESEARCH MODE ----------------
    if len(sys.argv) != 2:
        print("Usage: python analysis/coverage.py <repo-name>")
        sys.exit(1)

    repo_name = sys.argv[1]
    if repo_name not in VALIDATION_REPOS:
        print("[SKIP] Only validation repos supported")
        sys.exit(0)

    repo_path = TARGET_REPOS_DIR / repo_name
    python_exec = resolve_python(repo_name)

    DATA_DIR.mkdir(exist_ok=True)

    try:
        collect_coverage(repo_path, python_exec)
        print(f"[OK] Coverage saved for → {repo_name}")
    except CoverageError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)
