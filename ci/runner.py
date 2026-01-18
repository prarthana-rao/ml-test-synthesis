import subprocess
import sys
import os
from pathlib import Path


class CIError(Exception):
    pass


def run_step(module: str, project_root: Path, repo_root: Path):
    print(f"\n--- [CI STEP] {module} ---")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["CI_MODE"] = "1"
    env["CI_WORKSPACE"] = str(project_root / "ci_workspace")

    if module == "analysis.coverage":
        cmd = [
            sys.executable,
            str(project_root / "analysis" / "coverage.py"),
            str(repo_root)   # 🔥 Explicit repo passed
        ]
        cwd = repo_root
    else:
        env["TARGET_REPO"] = str(repo_root)
        cmd = [sys.executable, "-m", module]
        cwd = project_root

    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError:
        raise CIError(f"Step failed: {module}")


def run_analysis(repo_root: Path):
    repo_root = repo_root.resolve()
    project_root = Path(__file__).resolve().parents[1]

    print("\n🚦 STARTING CI ANALYSIS")
    print(f"📁 Target repo: {repo_root}")
    print(f"🧠 Tool root  : {project_root}")

    run_step("ml.build_validation_dataset", project_root, repo_root)
    run_step("ml.inference", project_root, repo_root)
    run_step("analysis.coverage", project_root, repo_root)
    run_step("analysis.post_ml_aggregate", project_root, repo_root)

    print("\n✅ CI ANALYSIS COMPLETE")
