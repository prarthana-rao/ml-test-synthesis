import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Setup Paths
DATA_DIR = "../data/processed/"
REPORTS_DIR = "../data/reports"

# Ensure reports directory exists at the root level
os.makedirs(REPORTS_DIR, exist_ok=True)

FINAL_RESULTS_FILE = os.path.join(DATA_DIR, "final_results.csv")

def load_and_clean_data(file_path):
    """
    Standardizes headers, simplifies absolute paths, ensures repository 
    consistency, and removes duplicates.
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
    
    # Standardize labels to UPPERCASE for consistency
    if 'smell_label' in df.columns:
        df['smell_label'] = df['smell_label'].astype(str).str.upper().str.strip()
    
    if 'repo_name' in df.columns:
        df['repo_name'] = df['repo_name'].astype(str).str.lower().str.strip()
        # Prepend repo name to path if missing to ensure uniqueness
        repos = df['repo_name'].unique()
        mask = ~df['file_path'].str.startswith(tuple(repos))
        df.loc[mask, 'file_path'] = df['repo_name'] + "/" + df['file_path']

    # C. Remove duplicates to ensure unique method counts
    df = df.drop_duplicates(subset=['method_name', 'file_path'], keep='first')
    
    return df

# --- MAIN EXECUTION ---
df_final = load_and_clean_data(FINAL_RESULTS_FILE)

# Set global aesthetic style
sns.set_theme(style="whitegrid")

if df_final is not None:
    # --- VISUALIZATION 1: High Smells Geography ---
    plt.figure(figsize=(12, 7))
    truth_counts = df_final.groupby(['repo_name', 'smell_label']).size().unstack(fill_value=0)
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


    # --- VISUALIZATION 2: Average Coverage by Repo ---
    plt.figure(figsize=(10, 6))
    avg_cov = df_final.groupby('repo_name')['coverage_percent'].mean().sort_values(ascending=False)
    ax2 = avg_cov.plot(kind='bar', color='skyblue')
    plt.title('Average Code Coverage by Repository')
    plt.ylabel('Mean Coverage (%)')
    plt.xticks(rotation=45)
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(REPORTS_DIR, '1_coverage_by_repo.png'), bbox_inches='tight')
    plt.close()
    print("Created: 1_coverage_by_repo.png")


    # --- UPDATED VISUALIZATION 3: Risk Distribution per Repository ---
    repos = df_final['repo_name'].unique()
    fig, axes = plt.subplots(1, len(repos), figsize=(18, 6))
    
    if len(repos) == 1:
        axes = [axes] # Handle single repo case

    colors = sns.color_palette('pastel')
    
    for i, repo in enumerate(repos):
        repo_data = df_final[df_final['repo_name'] == repo]
        risk_counts = repo_data['risk_category'].value_counts()
        
        axes[i].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
                    startangle=140, colors=colors)
        axes[i].set_title(f'Risk Profile: {repo.capitalize()}')

    plt.suptitle('Risk Category Distribution per Repository', fontsize=16)
    plt.savefig(os.path.join(REPORTS_DIR, '1_risk_distribution_per_repo.png'), bbox_inches='tight')
    plt.close()
    print("Created: 1_risk_distribution_per_repo.png")


    # --- VISUALIZATION 4: Quality Audit (Smell vs Coverage) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='smell_label', 
        y='coverage_percent', 
        data=df_final, 
        hue='smell_label', 
        palette={'HIGH': '#d62728', 'LOW': '#1f77b4'},
        legend=False
    )
    plt.title('Audit: Do High-Smell Methods have enough Coverage?')
    plt.ylabel('Coverage (%)')
    plt.savefig(os.path.join(REPORTS_DIR, '1_smell_vs_coverage.png'), bbox_inches='tight')
    plt.close()
    print("Created: 1_smell_vs_coverage.png")

print(f"\n Success! All reports have been generated in: {os.path.abspath(REPORTS_DIR)}")