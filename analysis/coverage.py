import subprocess
import json
from pathlib import Path
import sys


class CoverageError(Exception):
    pass


def collect_coverage(repo_path: str) -> dict:
    repo_path = Path(repo_path).resolve()

    if not repo_path.exists():
        raise CoverageError(f"Repo path does not exist: {repo_path}")

    repo_name = repo_path.name

    # Resolve repo-specific venv (workspace/venvs/<repo>)
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    exe_name = "python.exe" if sys.platform == "win32" else "python"

    venv_python = (
        repo_path.parents[1]
        / "venvs"
        / repo_name
        / bin_dir
        / exe_name
    )

    if not venv_python.exists():
        raise CoverageError(
            f"Python venv not found for repo '{repo_name}' at {venv_python}"
        )

    # Run tests under coverage
    try:
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "coverage",
                "run",
                "-m",
                "pytest",
            ],
            cwd=repo_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise CoverageError(
            f"Coverage execution failed:\n{e.stderr.decode()}"
        )

    # Generate JSON report
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise CoverageError(
            f"Coverage JSON generation failed:\n{e.stderr.decode()}"
        )

    coverage_file = repo_path / "coverage.json"

    if not coverage_file.exists():
        raise CoverageError("coverage.json not generated")

    with open(coverage_file, "r") as f:
        return json.load(f)
