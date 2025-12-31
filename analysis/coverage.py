import subprocess
import json
from pathlib import Path


class CoverageError(Exception):
    pass


def collect_coverage(repo_path: str) -> dict:
    repo_path = Path(repo_path).resolve()

    if not repo_path.exists():
        raise CoverageError(f"Repo path does not exist: {repo_path}")

    repo_name = repo_path.name

    # Assume venvs are stored at ../venvs/<repo_name> relative to tool
    venv_python = (
        repo_path.parents[1]
        / "venvs"
        / repo_name
        / "bin"
        / "python"
    )

    if not venv_python.exists():
        raise CoverageError(
            f"Python venv not found for repo '{repo_name}' at {venv_python}"
        )

    # Run coverage using the repo's Python environment
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
