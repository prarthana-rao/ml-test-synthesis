#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

from analysis.risk import classify_risk
from recommendations.rules import recommend_tests

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_CSV = PROCESSED_DIR / "ml_smell_predictions.csv"
OUTPUT_FULL = PROCESSED_DIR / "final_results.csv"
OUTPUT_TOPK = PROCESSED_DIR / "final_results_topk.csv"

TOP_K = 30

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def extract_repo_and_file(file_path: str):
    p = Path(file_path).resolve()
    parts = p.parts

    if "target-repos" not in parts:
        raise ValueError(f"Invalid File_Path format: {file_path}")

    idx = parts.index("target-repos")
    repo = parts[idx + 1]
    relative = Path(*parts[idx + 2:])

    return repo, str(relative)


def load_coverage(repo_name: str) -> dict:
    coverage_file = DATA_DIR / f"{repo_name}_coverage.json"
    if not coverage_file.exists():
        return {}

    with open(coverage_file) as f:
        return json.load(f).get("files", {})


def compute_function_coverage(row, coverage_files) -> float:
    file_path = row["file_path"]
    start = int(row["start_line"])
    end = int(row["end_line"])

    if start > end:
        return 0.0

    total_lines = end - start + 1
    executed = set()

    # ✅ KEEP THIS LOGIC — THIS IS WHY IT WORKS
    for covered_file, data in coverage_files.items():
        if covered_file.endswith(file_path):
            executed_lines = set(data.get("executed_lines", []))
            executed = executed_lines.intersection(range(start, end + 1))
            break

    if total_lines == 0:
        return 0.0

    return round((len(executed) / total_lines) * 100, 2)


def coverage_bucket(p: float) -> str:
    if p == 0:
        return "ZERO"
    if p <= 30:
        return "LOW"
    if p <= 70:
        return "MEDIUM"
    return "HIGH"

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    # ---------------- Schema normalization ----------------
    if "CC" in df.columns and "cc" not in df.columns:
        df = df.rename(columns={"CC": "cc"})

    if "File_Path" in df.columns:
        df = df.rename(columns={"File_Path": "file_path"})

    if "Method_Name" in df.columns:
        df = df.rename(columns={"Method_Name": "method_name"})

    # ---------------- Repo + relative path ----------------
    extracted = df["file_path"].apply(extract_repo_and_file)
    df["repo_name"] = extracted.apply(lambda x: x[0])
    df["file_path"] = extracted.apply(lambda x: x[1])

    # ---------------- Load coverage once per repo ----------------
    coverage_cache = {
        repo: load_coverage(repo)
        for repo in df["repo_name"].unique()
    }

    # ---------------- Function-level coverage ----------------
    df["coverage_percent"] = df.apply(
        lambda r: compute_function_coverage(
            r, coverage_cache.get(r["repo_name"], {})
        ),
        axis=1,
    )

    df["coverage_bucket"] = df["coverage_percent"].apply(coverage_bucket)

    # ---------------- Risk ----------------
    df["risk_category"] = df.apply(
        lambda r: classify_risk(r["smell_label"], r["coverage_bucket"]),
        axis=1,
    )

    # ---------------- Recommendations ----------------
    def build_recs(r):
        return "; ".join(
            recommend_tests(
                {
                    "risk_category": r["risk_category"],
                    "coverage_bucket": r["coverage_bucket"],
                    "cc": r.get("cc", 0),
                    "lloc": r.get("lloc", 0),
                    "difficulty": r.get("difficulty", 0),
                }
            )
        )

    df["recommendations"] = df.apply(build_recs, axis=1)

    # ---------------- Final schema ----------------
    final_cols = [
        "repo_name",
        "file_path",
        "method_name",
        "start_line",
        "end_line",
        "cc",
        "lloc",
        "difficulty",
        "smell_label",
        "coverage_percent",
        "coverage_bucket",
        "risk_category",
        "recommendations",
    ]

    df = df[[c for c in final_cols if c in df.columns]]

    PROCESSED_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FULL, index=False)
    print(f"[OK] Full results written to {OUTPUT_FULL}")

    # ---------------- TOP-K Hidden Risk ----------------
    df_hr = df[df["smell_label"] == "HIGH"]

    if df_hr.empty:
        print("[WARN] No Hidden Risk functions found")
        return

    sort_cols = ["lloc"]
    if "ml_confidence" in df.columns:
        sort_cols.insert(0, "ml_confidence")

    df_hr = df_hr.sort_values(by=sort_cols, ascending=False)
    df_topk = df_hr.groupby("repo_name").head(TOP_K)

    df_topk.to_csv(OUTPUT_TOPK, index=False)
    print(f"[OK] TOP-{TOP_K} results written to {OUTPUT_TOPK}")


if __name__ == "__main__":
    main()
