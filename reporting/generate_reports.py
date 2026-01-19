import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Setup Paths
DATA_DIR = "../data/processed/"
OFFENDERS_FILE = os.path.join(DATA_DIR, "final_results_topk.csv")
TRUTH_FILE = os.path.join(DATA_DIR, "final_results.csv")
ML_FILE = os.path.join(DATA_DIR, "ml_smell_predictions.csv") 

def load_and_clean():
    # Load raw files
    df_offenders = pd.read_csv(OFFENDERS_FILE)
    df_truth = pd.read_csv(TRUTH_FILE)
    df_ml = pd.read_csv(ML_FILE)

    def simplify_path(path):
        # Convert to string and fix slashes
        p = str(path).replace('\\', '/').lower().strip()
        # Specifically handle your absolute paths: /home/karthikpk/.../target-repos/
        if 'target-repos/' in p:
            return p.split('target-repos/')[-1]
        return p

    # Standardize column names to lowercase BEFORE deduplication
    for df in [df_offenders, df_truth, df_ml]:
        df.columns = df.columns.str.lower()
        df['method_name'] = df['method_name'].astype(str).str.strip().str.lower()
        df['file_path'] = df['file_path'].apply(simplify_path)
        
        # Ensure repo_name is consistent (jinja2, attrs, etc.)
        if 'repo_name' in df.columns:
            df['repo_name'] = df['repo_name'].astype(str).str.lower().str.strip()
            # If path doesn't start with repo name (e.g. 'src/...'), prepend it
            mask = ~df['file_path'].str.startswith(tuple(df['repo_name'].unique()))
            df.loc[mask, 'file_path'] = df['repo_name'] + "/" + df['file_path']

    # --- CRITICAL: Remove Duplicates in memory ---
    # Your ML file has many duplicate method entries (e.g. 67 __init__ calls)
    df_ml = df_ml.drop_duplicates(subset=['method_name', 'file_path'], keep='first')
    df_truth = df_truth.drop_duplicates(subset=['method_name', 'file_path'], keep='first')

    return df_offenders, df_truth, df_ml

# --- MAIN EXECUTION ---
df_offenders, df_truth, df_ml = load_and_clean()

# Merge Truth and ML for comparison
# This uses the standardized 'method_name' and 'file_path'
combined = pd.merge(df_truth, df_ml, on=['method_name', 'file_path'])
combined['smell_label_x'] = combined['smell_label_x'].str.upper()
combined['smell_label_y'] = combined['smell_label_y'].str.upper()

if combined.empty:
    print("ERROR: No matches found during merge. Check path formats.")
    exit()

# --- VISUALIZATION 1: High Smells Geography ---
plt.figure(figsize=(12, 7))
truth_counts = df_truth.groupby(['repo_name', 'smell_label']).size().unstack(fill_value=0)

# Ensure 'HIGH' and 'LOW' exist for the bar chart
for label in ['HIGH', 'LOW']:
    if label not in truth_counts.columns: truth_counts[label] = 0
truth_counts = truth_counts[['HIGH', 'LOW']]

# Plot Grouped bars (Red for High risk, Blue for Safe Zone)
ax = truth_counts.plot(kind='bar', color=['#d62728', '#1f77b4'], ax=plt.gca(), width=0.8)

# Add count labels above every bar
for p in ax.patches:
    h = p.get_height()
    if h >= 0:
        ax.text(p.get_x() + p.get_width()/2., h + 3, f'{int(h)}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.title('High-Risk vs Safe-Zone Function Counts', fontsize=14)
plt.ylabel('Number of Unique Methods')
plt.xlabel('Repository Name')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.ylim(0, truth_counts['LOW'].max() + 100)

plt.savefig('1_actual_risk_landscape.png', bbox_inches='tight')
print("Created: 1_actual_risk_landscape.png")

# --- VISUALIZATION 2: The Confusion Matrix ---
plt.figure(figsize=(8, 6))
conf_matrix = pd.crosstab(combined['smell_label_x'], combined['smell_label_y'], 
                          rownames=['Actual (Truth)'], colnames=['Predicted (AI)'])

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.title('AI Prediction Accuracy: Ground Truth vs. ML')
plt.savefig('2_ai_prediction_matrix.png', bbox_inches='tight')
print("Created: 2_ai_prediction_matrix.png")

# --- VISUALIZATION 3: Logic Check (Confidence vs CC) ---
plt.figure(figsize=(10, 6))
# Using 'cc' (standardized to lower) and 'ml_confidence' from your ML CSV
sns.regplot(data=df_ml, x='cc', y='ml_confidence', 
            scatter_kws={'alpha':0.4, 'color':'gray'}, 
            line_kws={'color':'red', 'lw':2})

plt.title('Model Logic Verification: Complexity vs. AI Confidence')
plt.xlabel('Cyclomatic Complexity (CC)')
plt.ylabel('ML Prediction Confidence (0.0 to 1.0)')
plt.savefig('3_ai_confidence.png', bbox_inches='tight')
print("Created: 3_ai_confidence.png")

print(f"\n🚀 Success! Visualizations generated based on {len(combined)} unique method matches.")