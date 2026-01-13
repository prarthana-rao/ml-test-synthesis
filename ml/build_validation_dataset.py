#!/usr/bin/env python3
"""
script_metrics_only.py
(Extracts metrics for all methods, no classification, no sampling)
"""
import os
import csv
import ast
from collections import Counter

from config.paths import VALIDATION_REPOS
from radon.complexity import cc_visit
from radon.raw import analyze as raw_analyze
from radon.metrics import h_visit
from pathlib import Path
from config.paths import TARGET_REPOS_DIR, VALIDATION_DATA_DIR

# ---------- CONFIG ----------
OUTPUT_CSV_FILE = VALIDATION_DATA_DIR / "long_method_validation_dataset.csv"

VALIDATION_REPOS = {"attrs", "jinja2", "itsdangerous"}

FIELDNAMES = [
    'File_Path', 'Method_Name', 'start_line', 'end_line',
    'CC', 'lloc', 'scloc', 'comments',
    'calculated_length', 'volume', 'difficulty',
    'effort', 'time', 'bugs'
]

# ---------- Utilities ----------

def is_test_path(path: str, filename: str = "") -> bool:
    p = path.lower()
    f = filename.lower()
    
    # Check directory
    is_test_dir = any(x in p for x in ["/test", "/testing", "/_test", "site-packages"])
    
    # Check filename
    is_test_file = f.startswith("test_") or f.endswith("_test.py") or f == "conftest.py" or f == "strategies.py"
    
    return is_test_dir or is_test_file


def get_node_end_lineno(node):
    end = getattr(node, 'end_lineno', None)
    if end is not None:
        return end
    max_lineno = getattr(node, 'lineno', None)
    for child in ast.walk(node):
        if hasattr(child, 'lineno'):
            try:
                if child.lineno and (max_lineno is None or child.lineno > max_lineno):
                    max_lineno = child.lineno
            except Exception:
                pass
    return max_lineno


def overlap_length(a_start, a_end, b_start, b_end):
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return 0
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0, end - start + 1)


def match_cc_block_for_node(cc_blocks, node_start, node_end):
    best_block = None
    best_overlap = 0
    for block in cc_blocks:
        b_start = getattr(block, 'lineno', None)
        b_end = getattr(block, 'endline', None)
        if b_end is None:
            b_end = b_start
        ol = overlap_length(node_start, node_end, b_start, b_end)
        if ol > best_overlap:
            best_overlap = ol
            best_block = block
    if best_block is None or best_overlap == 0:
        return None
    try:
        return best_block.complexity
    except Exception:
        return getattr(best_block, 'complexity', getattr(best_block, 'cc', None))


def get_method_source(content, node):
    try:
        src = ast.get_source_segment(content, node)
        if src:
            return src
    except Exception:
        pass
    lines = content.splitlines(keepends=True)
    start = getattr(node, 'lineno', None)
    end = get_node_end_lineno(node)
    if start is None or end is None:
        return None
    try:
        return ''.join(lines[start - 1:end])
    except Exception:
        return None

# ---------- Analysis per method ----------

def analyze_method(node, file_content, full_cc_list, file_path, counters=None):
    method_name = node.name
    node_start = getattr(node, 'lineno', None)
    node_end = get_node_end_lineno(node)

    method_code = get_method_source(file_content, node)
    if not method_code:
        if counters is not None:
            counters['skip_source'] += 1
        return None

    cc = match_cc_block_for_node(full_cc_list, node_start, node_end)
    if cc is None:
        if counters is not None:
            counters['skip_cc'] += 1
        return None

    try:
        raw = raw_analyze(method_code)
        lloc = getattr(raw, 'lloc', None)
        scloc = getattr(raw, 'sloc', None) or getattr(raw, 'sloc', 0)
        comments = getattr(raw, 'comments', 0)
    except Exception:
        if counters is not None:
            counters['skip_raw'] += 1
        return None

    try:
        hal_reports = h_visit(method_code)
        if hal_reports:
            hal = hal_reports[0]
            calculated_length = getattr(hal, 'length', None)
            if calculated_length is None:
                N1 = getattr(hal, 'N1', None)
                N2 = getattr(hal, 'N2', None)
                if N1 is not None and N2 is not None:
                    calculated_length = N1 + N2
            volume = getattr(hal, 'volume', None)
            difficulty = getattr(hal, 'difficulty', None)
            effort = getattr(hal, 'effort', None)
            time_metric = getattr(hal, 'time', None)
            bugs = getattr(hal, 'bugs', None)
        else:
            calculated_length = volume = difficulty = effort = time_metric = bugs = None
    except Exception:
        if counters is not None:
            counters['skip_halstead'] += 1
        return None

    if counters is not None:
        counters['added'] += 1

    return {
        'File_Path': file_path.replace('\\', '/'),
        'Method_Name': method_name,
        'start_line': node_start,
        'end_line': node_end,
        'CC': cc,
        'lloc': lloc,
        'scloc': scloc,
        'comments': comments,
        'calculated_length': calculated_length,
        'volume': volume,
        'difficulty': difficulty,
        'effort': effort,
        'time': time_metric,
        'bugs': bugs
    }

# ---------- File processing ----------

def process_file(file_path, counters=None):
    rows = []
    if counters is None:
        counters = Counter()
    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            content = fh.read()
    except Exception:
        counters['fail_read'] += 1
        return rows

    try:
        tree = ast.parse(content)
    except Exception:
        counters['fail_parse'] += 1
        return rows

    try:
        full_cc_list = cc_visit(content)
    except Exception:
        counters['fail_cc_visit'] += 1
        full_cc_list = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            res = analyze_method(node, content, full_cc_list, file_path, counters=counters)
            if res:
                rows.append(res)
    return rows

# ---------- Build dataset ----------

def build_dataset(projects_root=TARGET_REPOS_DIR, output_csv=OUTPUT_CSV_FILE):
    all_rows = []
    counters = Counter()

    for repo_path in projects_root.iterdir():
        if not repo_path.is_dir():
            continue

        repo_name = repo_path.name
        if repo_name not in VALIDATION_REPOS:
            continue

        print(f"Processing evaluation repo: {repo_name}")

        for root, dirs, files in os.walk(repo_path):
            # Prune directory search
            dirs[:] = [d for d in dirs if not d.startswith('.') and not "test" in d.lower()]
            
            if is_test_path(root): # Skip if the whole directory is a test dir
                continue

            for file in files:
                if file.endswith('.py'):
                    # NEW: Also check the individual filename
                    if is_test_path(root, file):
                        continue
                        
                    file_path = os.path.join(root, file)
                    # ... process file ...
                    rows = process_file(file_path, counters=counters)
                    all_rows.extend(rows)

    print(f"Total methods collected: {len(all_rows)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Dataset written to: {output_csv}")
    print("Counters:", dict(counters))



if __name__ == "__main__":
    build_dataset()
