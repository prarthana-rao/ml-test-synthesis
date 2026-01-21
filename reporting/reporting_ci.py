import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Setup Paths
# Since you are running from ml-test-synthesis/reporting:
# ../ci_workspace/processed/ takes you to the data
# ../ci_workspace/reports/ takes you to the reports folder
DATA_DIR = "../ci_workspace/processed/"
REPORTS_DIR = "../ci_workspace/reports"

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

FINAL_RESULTS_FILE = os.path.join(DATA_DIR, "final_results.csv")
ML_PREDICTIONS_FILE = os.path.join(DATA_DIR, "ml_smell_predictions.csv")

def load_and_clean(file_path):
    """Loads CSV, standardizes headers, simplifies paths, and removes duplicates."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()

    # Simplify absolute paths
    def simplify_path(path):
        p = str(path).replace('\\', '/').lower().strip()
        if 'target-repos/' in p:
            return p.split('target-repos/')[-1]
        return p

    df['method_name'] = df['method_name'].astype(str).str.strip().str.lower()
    df['file_path'] = df['file_path'].apply(simplify_path)
    
    # Standardize smell_label to UPPERCASE for color mapping
    if 'smell_label' in df.columns:
        df['smell_label'] = df['smell_label'].astype(str).str.upper().str.strip()

    # Ensure repo_name consistency
    if 'repo_name' in df.columns:
        df['repo_name'] = df['repo_name'].astype(str).str.lower().str.strip()
        repos = df['repo_name'].unique()
        mask = ~df['file_path'].str.startswith(tuple(repos))
        df.loc[mask, 'file_path'] = df['repo_name'] + "/" + df['file_path']

    # Deduplicate unique methods
    df = df.drop_duplicates(subset=['method_name', 'file_path'], keep='first')
    return df

# --- MAIN EXECUTION ---
df_final = load_and_clean(FINAL_RESULTS_FILE)
df_ml = load_and_clean(ML_PREDICTIONS_FILE)

# Set global aesthetic style
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# VISUALIZATION 1: High Smells Geography (from final_results)
# ---------------------------------------------------------
if df_final is not None:
    plt.figure(figsize=(12, 7))
    truth_counts = df_final.groupby(['repo_name', 'smell_label']).size().unstack(fill_value=0)
    
    # Ensure both HIGH and LOW labels exist
    for label in ['HIGH', 'LOW']:
        if label not in truth_counts.columns: truth_counts[label] = 0
    truth_counts = truth_counts[['HIGH', 'LOW']]

    ax1 = truth_counts.plot(kind='bar', color=['#d62728', '#1f77b4'], ax=plt.gca(), width=0.8)
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.text(p.get_x() + p.get_width()/2., h + 3, f'{int(h)}', 
                     ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.title('High-Risk vs Safe-Zone Function Counts by Repository', fontsize=14)
    plt.ylabel('Number of Unique Methods')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(REPORTS_DIR, '1_actual_risk_landscape.png'), bbox_inches='tight')
    plt.close()
    print("Created: 1_actual_risk_landscape.png")

# ---------------------------------------------------------
# VISUALIZATION 2: Risk Distribution per Repository (from final_results)
# ---------------------------------------------------------
if df_final is not None:
    repos = df_final['repo_name'].unique()
    fig, axes = plt.subplots(1, len(repos), figsize=(18, 6))
    if len(repos) == 1: axes = [axes]

    colors = sns.color_palette('pastel')
    for i, repo in enumerate(repos):
        repo_data = df_final[df_final['repo_name'] == repo]
        risk_counts = repo_data['risk_category'].value_counts()
        axes[i].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
        axes[i].set_title(f'Risk Profile: {repo.capitalize()}')

    plt.suptitle('Risk Category Distribution per Repository', fontsize=16)
    plt.savefig(os.path.join(REPORTS_DIR, '2_risk_distribution_per_repo.png'), bbox_inches='tight')
    plt.close()
    print(" Created: 2_risk_distribution_per_repo.png")

# ---------------------------------------------------------
# VISUALIZATION 3: Top 10 Methods by Maintenance Effort (from ml_smell_predictions)
# ---------------------------------------------------------
if df_ml is not None:
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
    plt.savefig(os.path.join(REPORTS_DIR, '3_ml_top_10_effort.png'), bbox_inches='tight')
    plt.close()
    print(" Created: 3_ml_top_10_effort.png")

print(f"\n Success! All reports have been generated in: {os.path.abspath(REPORTS_DIR)}")