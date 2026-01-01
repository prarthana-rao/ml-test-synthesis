import pandas as pd
from analysis.pipeline_demo import run_pipeline
from analysis.coverage import collect_coverage
from pathlib import Path


def run_pipeline_from_ml(repo_base_path: str, ml_output_csv: str, top_k: int = 30):
    df = pd.read_csv(ml_output_csv)

    df["repo"] = df["File_Path"].apply(lambda p: p.split("/")[0])
    df_high = df[df["smell_label"] == "HIGH"]

    if df_high.empty:
        print("No HIGH smell functions found.")
        return []

    sort_cols = ["ml_confidence", "lloc"] if "ml_confidence" in df_high.columns else ["lloc"]
    df_high = df_high.sort_values(by=sort_cols, ascending=False)

    selected = df_high.groupby("repo").head(top_k)

    results = []

    for repo_name, repo_data in selected.groupby("repo"):
        repo_path = f"{repo_base_path}/{repo_name}"

        print(f"\n🚀 Running coverage once for: {repo_name}")
        try:
            coverage_data = collect_coverage(repo_path)

            for _, row in repo_data.iterrows():
                file_path = row["File_Path"]
                repo_prefix = repo_name + "/"

                if file_path.startswith(repo_prefix):
                    file_path = file_path[len(repo_prefix):]

                function = {
                    "file": file_path,
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                    "cc": row["cc"],
                    "lloc": row["lloc"],
                    "difficulty": row["difficulty"],
                    "smell_label": row["smell_label"],
                }

                result = run_pipeline(
                    repo_path,
                    function,
                    coverage_data=coverage_data
                )

                # Enrich result with repo + method metadata
                result.update({
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "method_name": row.get("Method_Name", ""),
                    "cc": row["cc"],
                    "lloc": row["lloc"],
                    "difficulty": row["difficulty"],
                })

                results.append(result)

        except Exception as e:
            print(f"Failed processing repo {repo_name}: {e}")

    # --------------------------------------------------
    # STORAGE (added section – nothing above modified)
    # --------------------------------------------------
    if results:
        df_results = pd.DataFrame(results)

        # Serialize recommendations list → CSV-safe string
        df_results["recommendations"] = df_results["recommendations"].apply(
            lambda recs: "; ".join(recs)
        )

        # Reorder columns to final agreed schema
        final_columns = [
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

        df_results = df_results[final_columns]

        output_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "final_results.csv"
        df_results.to_csv(output_path, index=False)

        print(f"\n✅ Final results written to: {output_path}")

    return results

if __name__ == "__main__":
    from config.paths import TARGET_REPOS_DIR, PROCESSED_DATA_DIR

    ML_OUTPUT_CSV = PROCESSED_DATA_DIR / "ml_smell_predictions.csv"

    print("\n📊 Starting post-ML analysis pipeline...")

    run_pipeline_from_ml(
        repo_base_path=str(TARGET_REPOS_DIR),
        ml_output_csv=str(ML_OUTPUT_CSV),
        top_k=30
    )
    print("\n✅ Post-ML analysis pipeline completed.")
