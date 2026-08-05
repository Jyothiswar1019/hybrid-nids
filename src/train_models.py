import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_models(data_path, holdout_class='u2r', 
                 rf_path='models/rf_model.pkl', iso_path='models/iso_model.pkl'):
    # Load the CSV with selected features
    df = pd.read_csv(data_path)
    
    # Get feature columns (all except 'attack_cat')
    selected_features = [col for col in df.columns if col != 'attack_cat']
    print("Features used for training:", selected_features)
    
    X = df[selected_features]
    y = df['attack_cat']
    
    # Separate normal and attacks
    normal_mask = (y == 'normal')
    X_normal = X[normal_mask]
    
    # For training classifier, exclude the holdout class (simulate zero-day)
    train_mask = ~(y == holdout_class)
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_path)
    print("RF saved to", rf_path)
    
    # Train Isolation Forest on normal data only
    print("Training Isolation Forest on normal traffic...")
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_normal)
    joblib.dump(iso, iso_path)
    print("Isolation Forest saved to", iso_path)
    
    # Evaluate on holdout class
    X_holdout = X[y == holdout_class]
    y_holdout = y[y == holdout_class]
    if len(X_holdout) > 0:
        pred_rf = rf.predict(X_holdout)
        pred_iso = iso.predict(X_holdout)
        anomaly_flag = (pred_iso == -1)
        print(f"Holdout class '{holdout_class}' samples: {len(X_holdout)}")
        print("RF predictions (distribution):", np.unique(pred_rf, return_counts=True))
        print("Anomaly detection (zero-day flagged):", np.sum(anomaly_flag), "out of", len(X_holdout))
    
    return rf, iso

if __name__ == "__main__":
    # Now we don't need to pass selected_features; the CSV provides it.
    train_models('data/selected_features.csv')