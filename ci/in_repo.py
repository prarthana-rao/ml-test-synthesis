from pathlib import Path
import shutil
import sys

from ci.runner import run_analysis, CIError


def ensure_tool_available(tool: str):
    if shutil.which(tool) is None:
        raise CIError(
            f"Required tool not found in PATH: {tool}. "
            f"Ensure dependencies are installed before running CI."
        )


def main():
    """
    CI entrypoint.

    Usage:
        python -m ci.in_repo [repo_path]

    - If repo_path is provided → analyze that repo
    - Otherwise → analyze current working directory
    """

    if len(sys.argv) > 2:
        print("Usage: python -m ci.in_repo [repo_path]")
        sys.exit(1)

    repo_root = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) == 2
        else Path.cwd().resolve()
    )

    print(f"📁 Running CI analysis in repo: {repo_root}")

    ensure_tool_available("pytest")
    ensure_tool_available("coverage")

    if not (repo_root / "pyproject.toml").exists() and not (
        repo_root / "setup.py"
    ).exists():
        print("⚠️  Warning: No pyproject.toml or setup.py found. Proceeding anyway.")

    try:
        run_analysis(repo_root)
    except CIError as e:
        print(f"\n❌ CI FAILED: {e}")
        sys.exit(1)

    print("\n🎉 CI PASSED")


if __name__ == "__main__":
    main()
