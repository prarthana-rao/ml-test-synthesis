from pathlib import Path


def compute_function_coverage(function, coverage_data):
    """
    function dict must contain:
      - file: relative path from repo root
      - start_line
      - end_line
    """

    file_path = Path(function["file"]).as_posix()

    files_data = coverage_data.get("files", {})

    if file_path not in files_data:
        return 0.0, "ZERO"

    executed_lines = set(
        files_data[file_path].get("executed_lines", [])
    )

    start = function["start_line"]
    end = function["end_line"]

    if start is None or end is None or start > end:
        return 0.0, "ZERO"

    total_lines = end - start + 1

    covered_lines = sum(
        1 for line in range(start, end + 1)
        if line in executed_lines
    )

    if total_lines <= 0:
        return 0.0, "ZERO"

    percent = (covered_lines / total_lines) * 100.0

    bucket = bucket_coverage(percent)

    return round(percent, 2), bucket


def bucket_coverage(percent):
    if percent == 0:
        return "ZERO"
    elif percent <= 30:
        return "LOW"
    elif percent <= 70:
        return "MEDIUM"
    else:
        return "HIGH"
