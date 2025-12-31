### Machine Learning–Guided Code Smell Detection for Targeted Test Synthesis in Python

This project presents a practical system for identifying testing risk in Python codebases by combining machine learning–based code smell detection with dynamic test coverage analysis. Rather than automatically generating tests, the system prioritizes testing effort by identifying high-risk, under-tested code regions and synthesizing actionable test recommendations.

## 1. Project Overview


Modern Python projects often contain complex code that is insufficiently tested. While code smells indicate maintainability issues, and coverage tools indicate test execution, neither alone provides guidance on *where testing effort should be focused*.

This project addresses that gap by:

1. Detecting code smells using a machine learning model trained on static code metrics
2. Measuring runtime test coverage at the function level
3. Combining both signals to classify **testing risk**
4. Generating **test synthesis guidance** to help developers prioritize testing and refactoring

## 2. System Architecture

The system operates in two distinct phases:

### Offline Phase (Machine Learning)
- Extract static code metrics using Radon
- Label functions using heuristic-based smell definitions
- Train an SVM model to predict smelly functions
- Persist the trained model for inference

### Online Phase (Analysis Pipeline)
- Analyze a target repository
- Predict code smells using the trained model
- Collect test coverage using `coverage.py`
- Map coverage to individual functions
- Classify each function into a risk category
- Generate targeted test recommendations
    

## 3. Workspace Structure

The project uses an external workspace to ensure isolation and reproducibility.

```
workspace/
├── target-repos/        # External repositories (read-only)
├── venvs/               # One virtual environment per repo + tool
└── ml-test-synthesis/   # This project
```

> Target repositories are never committed and are treated as immutable inputs.

## 4. Repository Structure
```
ml-test-synthesis/  
├── analysis/ 			# Runtime analysis pipeline (coverage, risk, inference)  
├── cli/ 				# Command-line interface (optional)  
├── config/ 			# Configuration files (repo definitions, thresholds)  
├── data/ 				# Datasets and derived artifacts  
│ ├── training/ 		# Training datasets  
│ └── validation/ 		# Validation / evaluation datasets  
├── ml/ 				# Offline ML utilities (dataset building, training)  
├── models/ 			# Trained ML models  
├── recommendations/ 	# Rule-based test synthesis logic  
├── reporting/ 			# Reports, summaries, and figures  
├── scripts/ 			# Automation scripts (workspace setup)  
├── .gitignore  
├── README.md  
└── requirements.txt
```

## 5. Automated Setup (Recommended)

The entire workspace can be set up automatically using a cross-platform Python script.

### Prerequisites

-   Python 3.9+
    
-   Git
    

### Setup Command

```bash
python scripts/setup_workspace.py
```

This script:

-   Creates the workspace structure

-   Clones all target repositories
    
-   Checks out pinned tags/commits
    
-   Creates isolated virtual environments
    
-   Installs all required dependencies
    


## 6. Repository Pinning & Reproducibility

All repositories are pinned to specific tagged releases or commit hashes to ensure deterministic results.  
Repository versions are defined directly in the setup script.

This ensures:

-   Identical analysis across machines
    
-   Stable ML results
    
-   Reproducible evaluation

## 6.1 Training and Validation Repositories

To ensure meaningful evaluation and avoid data leakage, the project clearly separates repositories used for **training** and **validation**.

### Training Repositories
These repositories are used exclusively to build the machine learning dataset and train the code smell detection model.

Examples:
- `requests`
- `flask`
- `click`

Static code metrics are extracted from these repositories and heuristically labeled to train the SVM model.

---

### Validation Repositories
These repositories are **never used during training**.  
They are used to evaluate the model’s ability to generalize to unseen codebases and to validate the end-to-end analysis pipeline.

Examples:
- `attrs`
- `jinja2`
- `itsdangerous`

All reported analysis results, risk classifications, and test recommendations on these repositories reflect **out-of-sample behavior**.

---

### Demo Repositories
Some repositories may be reused for demonstration purposes (e.g., screenshots or pipeline walkthroughs).  
This reuse does not affect training or validation, as demo execution does not influence the trained model.

    


## 7. Virtual Environments

|Purpose|Virtual Environment  |
|--|--|
|  Run Analysis pipeline| `workspace/venvs/ml-test-synthesis` |
| Run coverage for `requests`|`workspace/venvs/requests`|
|Run coverage for `flask`|`workspace/venvs/flask`|
|Run coverage for `click`|`workspace/venvs/click`|

> **Important:** Never install target repository dependencies into the tool environment.


## 8. Running the Analysis Pipeline

Activate the tool environment:

```bash
source workspace/venvs/ml-test-synthesis/bin/activate   # Linux/macOS
workspace\venvs\ml-test-synthesis\Scripts\activate      # Windows
```

Run the pipeline demo:

```bash
python -m analysis.pipeline_demo
```

This executes:

-   Coverage analysis
    
-   Function-level coverage mapping
    
-   ML-based smell prediction
    
-   Risk classification
    
-   Test recommendation synthesis
    

## 9. Risk Categories

Each function is classified into one of four categories:

|Category|  Description|
|--|--|
| Hidden Risk | Smelly code with low or zero coverage |
|Refactor Candidate|Smelly but adequately tested code|
|Low Value|Simple, untested code|
|Safe Zone|Clean and well-tested code|


## 10. Test Synthesis Strategy

The system generates **test guidance**, not test code.  
Recommendations are based on:

-   Coverage gaps
    
-   Cyclomatic complexity
    
-   Method size
    
-   Dependency complexity
    

This approach avoids fragile automated test generation while remaining practical for real-world teams.

## 11. Datasets and Machine Learning

-   Raw metrics are stored in `data/raw/`
    
-   Processed datasets are stored in `data/processed/`
    
-   Trained models are stored in `models/`
    

The ML model is trained using **heuristic-labeled data**, and the approach is explicitly framed as _learning to approximate rule-based smell detection across projects_.

Validation repositories are excluded from dataset construction and are used only for evaluating generalization and pipeline behavior.

## 12. Reproducibility Statement

All experiments are fully reproducible.  
Repositories are pinned to immutable versions, environments are isolated, and the entire setup process is automated.

## 13. Limitations

-   Smell labels are heuristic-based rather than manually curated
    
-   Coverage is approximated at the function level using line execution
    
-   The system provides guidance, not automated test generation

These trade-offs are intentional to maintain scalability and robustness.

## 14. Conclusion

This project demonstrates that combining machine learning–based code smell detection with coverage-aware analysis enables practical, risk-driven testing decisions in Python systems.  
The approach is scalable, reproducible, and aligned with real-world development constraints.