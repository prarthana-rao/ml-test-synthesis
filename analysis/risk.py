def classify_risk(smell_label: str, coverage_bucket: str) -> str:
    """
    smell_label: 'HIGH' or 'LOW'
    coverage_bucket: 'ZERO', 'LOW', 'MEDIUM', 'HIGH'
    """

    smell_label = smell_label.upper()
    coverage_bucket = coverage_bucket.upper()

    if smell_label == "HIGH" and coverage_bucket in ("ZERO", "LOW"):
        return "Hidden Risk"

    if smell_label == "HIGH" and coverage_bucket in ("MEDIUM", "HIGH"):
        return "Refactor Candidate"

    if smell_label == "LOW" and coverage_bucket in ("ZERO", "LOW"):
        return "Low Value"

    return "Safe Zone"
