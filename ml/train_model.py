import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from config.paths import MODELS_DIR, TRAINING_DATA_DIR
import warnings

warnings.filterwarnings('ignore')

# --- 1. Config ---
train_file = TRAINING_DATA_DIR / "long_method_training_dataset.csv"
model_filename = MODELS_DIR / "smell_detector.pkl"
scaler_filename = MODELS_DIR / "scaler.pkl"
necessary_features = ['scloc', 'lloc', 'effort', 'time', 'bugs', 'volume', 'difficulty', 'calculated_length']
target_column = 'is_Long_Method'

try:
    df = pd.read_csv(train_file, encoding='latin1')
    X = df[necessary_features]
    y = df[target_column]

    # --- Train/Test Split ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Scaling ---
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # --- Evaluation ---
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # --- Final Retrain on 100% Data for Deployment ---
    final_scaler = MinMaxScaler()
    X_full_scaled = final_scaler.fit_transform(X)
    final_model = SVC(kernel='rbf', probability=True, random_state=42)
    final_model.fit(X_full_scaled, y)

    # --- Save Resources ---
    joblib.dump(final_model, model_filename)
    joblib.dump(final_scaler, scaler_filename)
    print("✅ Model and Scaler saved successfully.")

except Exception as e:
    print(f"❌ Error during training: {e}")