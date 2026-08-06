import os
import sys
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import scale_features

# All 41 features of NSL-KDD (order must match training)
ALL_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# Global cache for artifacts
_artifacts_cache = None

def load_artifacts():
    """Lazy-load all required models and encoders."""
    global _artifacts_cache
    if _artifacts_cache is None:
        print("🔃 Loading XGBoost + Autoencoder models...")
        # New models
        xgb = joblib.load(os.path.join(ROOT_DIR, 'models/xgb_model.pkl'))
        label_encoder = joblib.load(os.path.join(ROOT_DIR, 'models/label_encoder.pkl'))
        autoencoder = load_model(os.path.join(ROOT_DIR, 'models/autoencoder_model.keras'))
        ae_scaler = joblib.load(os.path.join(ROOT_DIR, 'models/ae_scaler.pkl'))
        ae_threshold = joblib.load(os.path.join(ROOT_DIR, 'models/ae_threshold.pkl'))
        # Preprocessing artifacts (unchanged)
        scaler = joblib.load(os.path.join(ROOT_DIR, 'models/scaler.pkl'))
        le_dict = joblib.load(os.path.join(ROOT_DIR, 'models/label_encoders.pkl'))
        selector = joblib.load(os.path.join(ROOT_DIR, 'models/selector.pkl'))
        _artifacts_cache = (xgb, label_encoder, autoencoder, ae_scaler, ae_threshold,
                            scaler, le_dict, selector)
        print("✅ Models loaded.")
    return _artifacts_cache

def safe_transform(le, value, default):
    """Safely encode a categorical value using the given label encoder."""
    val_str = str(value).strip()
    # Try exact uppercase match first
    if isinstance(value, str) and value.upper() in le.classes_:
        return le.transform([value.upper()])[0]
    if val_str in le.classes_:
        return le.transform([val_str])[0]
    # Try different capitalisations
    for variation in [val_str.lower(), val_str.upper(), val_str.capitalize()]:
        if variation in le.classes_:
            return le.transform([variation])[0]
    # Fallback to default or first class
    try:
        return le.transform([default])[0]
    except:
        return le.transform([le.classes_[0]])[0]

def preprocess_input(raw_input_dict):
    """
    Convert raw user input (dict) into a scaled and feature‑selected array
    ready for model prediction.
    """
    # Load artifacts (we need scaler, le_dict, selector)
    _, _, _, _, _, scaler, le_dict, selector = load_artifacts()

    # Fill all features with default 0.0
    full_dict = {feat: 0.0 for feat in ALL_FEATURES}
    for key, value in raw_input_dict.items():
        if key in full_dict:
            full_dict[key] = value

    # Ensure categorical fields have valid defaults
    if 'protocol_type' not in raw_input_dict or str(full_dict.get('protocol_type', '')).strip() in ['0', '0.0', '']:
        full_dict['protocol_type'] = 'tcp'
    if 'service' not in raw_input_dict or str(full_dict.get('service', '')).strip() in ['0', '0.0', '']:
        full_dict['service'] = 'http'
    if 'flag' not in raw_input_dict or str(full_dict.get('flag', '')).strip() in ['0', '0.0', '']:
        full_dict['flag'] = 'SF'

    # Create DataFrame with correct column order
    df = pd.DataFrame([full_dict])[ALL_FEATURES]

    # Encode categorical columns
    default_map = {'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF'}
    for col in ['protocol_type', 'service', 'flag']:
        le = le_dict[col]
        df[col] = safe_transform(le, full_dict[col], default_map[col])

    # Scale numerical features
    df, _ = scale_features(df, scaler=scaler, fit=False)

    # Apply feature selection
    X_selected = selector.transform(df.values)
    return X_selected

def predict_sample(raw_input_dict, confidence_threshold=0.5):
    """
    Predict using the hybrid XGBoost + Autoencoder system.
    Returns: (prediction, confidence, type_label)
    """
    # Load all artifacts
    (xgb, label_encoder, autoencoder, ae_scaler, ae_threshold,
     scaler, le_dict, selector) = load_artifacts()

    # Preprocess input (returns selected feature array)
    X = preprocess_input(raw_input_dict)   # shape: (1, n_selected_features)

    # ---- Autoencoder anomaly detection ----
    X_ae = ae_scaler.transform(X)          # scale with autoencoder's scaler
    recon = autoencoder.predict(X_ae, verbose=0)
    mse = np.mean(np.square(X_ae - recon), axis=1)[0]
    is_anomaly = mse > ae_threshold

    # ---- XGBoost classifier ----
    probs = xgb.predict_proba(X)[0]
    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    # ---- Fusion decision ----
    if is_anomaly:
        # High reconstruction error → treat as zero‑day
        return "Zero-Day Attack", mse, 'anomaly'
    else:
        if pred_class == 'normal':
            return "Normal Traffic", max_prob, 'normal'
        else:
            if max_prob >= confidence_threshold:
                return pred_class, max_prob, 'known'
            else:
                return "Zero-Day Attack", max_prob, 'anomaly'
