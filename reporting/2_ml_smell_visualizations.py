import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Setup Paths
# "../" moves up from 'reporting' to the project root 'ml-test-synthesis'
DATA_DIR = "../data/processed/"
REPORTS_DIR = "../data/reports"

# Ensure reports directory exists at the root level
os.makedirs(REPORTS_DIR, exist_ok=True)

ML_PREDICTIONS_FILE = os.path.join(DATA_DIR, "ml_smell_predictions.csv")

def load_and_clean_ml_data(file_path):
    """
    Standardizes headers, simplifies absolute paths, and removes duplicates 
    for the ML predictions dataset.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    df = pd.read_csv(file_path)

    # A. Simplify paths (removes system-specific prefixes)
    def simplify_path(path):
        p = str(path).replace('\\', '/').lower().strip()
        if 'target-repos/' in p:
            return p.split('target-repos/')[-1]
        return p

    # B. Standardize columns and values
    df.columns = df.columns.str.lower()
    df['method_name'] = df['method_name'].astype(str).str.strip().str.lower()
    df['file_path'] = df['file_path'].apply(simplify_path)
    
    # Standardize smell_label to UPPERCASE for palette consistency
    if 'smell_label' in df.columns:
        df['smell_label'] = df['smell_label'].astype(str).str.upper().str.strip()

    # C. Remove duplicates to ensure unique method counts
    df = df.drop_duplicates(subset=['method_name', 'file_path'], keep='first')
    
    return df

# --- MAIN EXECUTION ---
df_ml = load_and_clean_ml_data(ML_PREDICTIONS_FILE)

# Set global aesthetic style
sns.set_theme(style="whitegrid")

if df_ml is not None:
    # --- VISUALIZATION 1: Code Metric Correlation Matrix ---
    # Goal: Identify which metrics drive complexity and bug risk.
    plt.figure(figsize=(10, 8))
    numeric_cols = ['cc', 'lloc', 'difficulty', 'effort', 'bugs', 'ml_confidence']
    available_cols = [c for c in numeric_cols if c in df_ml.columns]
    
    corr = df_ml[available_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Correlation Level'})
    
    plt.title('Code Metric Correlation Matrix', fontsize=15)
    plt.savefig(os.path.join(REPORTS_DIR, '2_ml_metric_correlation.png'), bbox_inches='tight')
    plt.close()
    print(" Created: 2_ml_metric_correlation.png")


    # --- VISUALIZATION 2: Bugs Probability vs. Structural Complexity ---
    # Goal: Check if complex code paths (CC) correlate with higher predicted bugs.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_ml, 
        x='cc', 
        y='bugs', 
        hue='smell_label', 
        palette={'HIGH': '#d62728', 'LOW': '#1f77b4'}, 
        alpha=0.6
    )
    plt.title('Bugs Probability vs. Structural Complexity', fontsize=14)
    plt.xlabel('Cyclomatic Complexity (CC)')
    plt.ylabel('Predicted Bugs Count')
    plt.savefig(os.path.join(REPORTS_DIR, '2_ml_bugs_complexity.png'), bbox_inches='tight')
    plt.close()
    print(" Created: 2_ml_bugs_complexity.png")


    # --- VISUALIZATION 3: Top 10 Methods by Maintenance Effort ---
    # Goal: Identify specific "Hotspots" that require the most developer time.
    plt.figure(figsize=(12, 6))
    # Select top 10 by effort
    top_10_effort = df_ml.nlargest(10, 'effort')
    # Shorten names for cleaner display
    top_10_effort['display_name'] = top_10_effort['method_name'].apply(
        lambda x: x[:25] + '...' if len(x) > 25 else x
    )
    
    sns.barplot(x='effort', y='display_name', data=top_10_effort, palette='Reds_r')
    
    plt.title('Top 10 Methods by Maintenance Effort (Halstead)', fontsize=14)
    plt.xlabel('Halstead Effort Score')
    plt.ylabel('Method Name')
    plt.savefig(os.path.join(REPORTS_DIR, '2_ml_top_10_effort.png'), bbox_inches='tight')
    plt.close()
    print(" Created: 2_ml_top_10_effort.png")

print(f"\n Success! ML metric reports have been generated in: {os.path.abspath(REPORTS_DIR)}")