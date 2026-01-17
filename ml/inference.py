import pandas as pd
import joblib
import numpy as np
from config.paths import MODELS_DIR, VALIDATION_DATA_DIR, PROCESSED_DATA_DIR
from pathlib import Path

import os
CI_MODE = os.getenv("CI_MODE") == "1"
CI_WORKSPACE = Path(os.getenv("CI_WORKSPACE", PROCESSED_DATA_DIR))


# --- 1. Config ---
unseen_file = VALIDATION_DATA_DIR / "long_method_validation_dataset.csv"
model_filename = MODELS_DIR / "smell_detector.pkl"
scaler_filename = MODELS_DIR / "scaler.pkl"
output_file = PROCESSED_DATA_DIR / "ml_smell_predictions.csv" # Renamed for clarity

unseen_file = (CI_WORKSPACE / "metrics" / "long_method_validation_dataset.csv") if CI_MODE else unseen_file
output_file = (CI_WORKSPACE / "processed" / "ml_smell_predictions.csv") if CI_MODE else output_file


necessary_features = ['scloc', 'lloc', 'effort', 'time', 'bugs', 'volume', 'difficulty', 'calculated_length']

try:
    # --- 2. Load Resources ---
    df_new = pd.read_csv(unseen_file, encoding='latin1')
    clf = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    # --- 3. Preprocessing ---
    X_new = df_new[necessary_features].fillna(0) # Safety first
    X_new_scaled = scaler.transform(X_new)

    # --- 4. Prediction ---
    preds = clf.predict(X_new_scaled)
    # We keep probability ONLY for logging/metadata, NOT for decision making
    probs = clf.predict_proba(X_new_scaled)[:, 1] 

    # --- 5. Clean Mapping (The Hard Line) ---
    # Map 1 -> HIGH, 0 -> LOW to match risk.py expectations
    df_new['smell_label'] = np.where(preds == 1, "HIGH", "LOW")
    
    # Optional: Keep the raw probability for the final report CSV, 
    # but ensure it's not used for "Risk" logic here.
    df_new['ml_confidence'] = np.round(probs, 4)

    # --- 6. Output Generation ---
    # We do NOT sort by probability. We keep the original order or sort by Method_Name.
    final_report = df_new.sort_values(by='Method_Name')

    cols_to_show = ['Method_Name', 'smell_label', 'ml_confidence']
    if 'File_Path' in final_report.columns:
        cols_to_show.insert(0, 'File_Path')

    print("\n--- 🎯 ML Smell Detection Results ---")
    print(final_report[cols_to_show].head(10))

    # Save to CSV - this will be read by your analysis module
    final_report.to_csv(output_file, index=False)
    print(f"\n✅ Predictions complete. Output saved to: {output_file}")

except Exception as e:
    print(f"❌ An error occurred: {e}")