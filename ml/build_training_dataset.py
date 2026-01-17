"""
script_fixed_clean.py
(All debug prints removed)
"""
import os
import csv
import ast
import random
from collections import Counter

from radon.complexity import cc_visit
from radon.raw import analyze as raw_analyze
from radon.metrics import h_visit
from pathlib import Path
from config.paths import TARGET_REPOS_DIR, TRAINING_DATA_DIR, TRAINING_REPOS

# ---------- CONFIG ----------
OUTPUT_CSV_FILE = TRAINING_DATA_DIR / "long_method_training_dataset.csv"

# ---------- TRAINING REPOS ----------
# TRAINING_REPOS = {"requests", "flask", "click"}


LLOC_THRESHOLD = 30
CC_THRESHOLD = 10

MAX_SMELLY_SAMPLES = 200
MAX_NON_SMELLY_SAMPLES = 800

FIELDNAMES = [
    'File_Path', 'Method_Name', 'start_line', 'end_line', 'is_Long_Method',
    'CC', 'lloc', 'scloc', 'comments',
    'calculated_length', 'volume', 'difficulty',
    'effort', 'time', 'bugs'
]

# ---------- Utilities ----------

def is_test_path(path: str) -> bool:
    p = path.lower()
    return (
        "/test/" in p or
        "/tests/" in p or
        "\\test\\" in p or
        "\\tests\\" in p
    )


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

# ---------- Label -----------

def get_smell_label(lloc, cc):
    return 1 if (lloc is not None and cc is not None and (lloc > LLOC_THRESHOLD and cc > CC_THRESHOLD)) else 0

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
        'is_Long_Method': get_smell_label(lloc, cc),
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
        if repo_name not in TRAINING_REPOS:
            continue


        print(f"Processing training repo: {repo_name}")

        for root, dirs, files in os.walk(repo_path):
            if is_test_path(root):
                continue
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rows = process_file(file_path, counters=counters)
                    all_rows.extend(rows)

    # ---- sampling logic unchanged ----
    smelly_list = [r for r in all_rows if r.get('is_Long_Method') == 1]
    non_smelly_list = [r for r in all_rows if r.get('is_Long_Method') == 0]

    if len(smelly_list) > MAX_SMELLY_SAMPLES:
        smelly_sampled = random.sample(smelly_list, MAX_SMELLY_SAMPLES)
    else:
        smelly_sampled = smelly_list

    if len(non_smelly_list) > MAX_NON_SMELLY_SAMPLES:
        non_smelly_sampled = random.sample(non_smelly_list, MAX_NON_SMELLY_SAMPLES)
    else:
        non_smelly_sampled = non_smelly_list

    final_data = smelly_sampled + non_smelly_sampled
    random.shuffle(final_data)

    print(f"Smelly samples collected: {len(smelly_sampled)}")
    print(f"Non-smelly samples collected: {len(non_smelly_sampled)}")
    print(f"Final dataset size: {len(final_data)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(final_data)


if __name__ == "__main__":
    build_dataset()

