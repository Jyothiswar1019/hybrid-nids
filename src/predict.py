import os
import joblib
import numpy as np
import pandas as pd
from src.utils import scale_features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# Global cache for artifacts (loaded once)
_artifacts_cache = None

def load_artifacts():
    """Lazy-load artifacts – only loads once."""
    global _artifacts_cache
    if _artifacts_cache is None:
        print("🔃 Loading ML models (this may take a few seconds)...")
        rf = joblib.load(os.path.join(ROOT_DIR, 'models/rf_model.pkl'))
        iso = joblib.load(os.path.join(ROOT_DIR, 'models/iso_model.pkl'))
        scaler = joblib.load(os.path.join(ROOT_DIR, 'models/scaler.pkl'))
        le_dict = joblib.load(os.path.join(ROOT_DIR, 'models/label_encoders.pkl'))
        selector = joblib.load(os.path.join(ROOT_DIR, 'models/selector.pkl'))
        _artifacts_cache = (rf, iso, scaler, le_dict, selector)
        print("✅ Models loaded.")
    return _artifacts_cache

def safe_transform(le, value, default):
    """Safely encode categorical values."""
    val_str = str(value).strip()
    if isinstance(value, str) and value.upper() in le.classes_:
        return le.transform([value.upper()])[0]
    if val_str in le.classes_:
        return le.transform([val_str])[0]
    for variation in [val_str.lower(), val_str.upper(), val_str.capitalize()]:
        if variation in le.classes_:
            return le.transform([variation])[0]
    try:
        return le.transform([default])[0]
    except:
        return le.transform([le.classes_[0]])[0]

def preprocess_input(raw_input_dict):
    rf, iso, scaler, le_dict, selector = load_artifacts()

    full_dict = {feat: 0.0 for feat in ALL_FEATURES}
    for key, value in raw_input_dict.items():
        if key in full_dict:
            full_dict[key] = value

    if 'protocol_type' not in raw_input_dict or str(full_dict.get('protocol_type', '')).strip() in ['0', '0.0', '']:
        full_dict['protocol_type'] = 'tcp'
    if 'service' not in raw_input_dict or str(full_dict.get('service', '')).strip() in ['0', '0.0', '']:
        full_dict['service'] = 'http'
    if 'flag' not in raw_input_dict or str(full_dict.get('flag', '')).strip() in ['0', '0.0', '']:
        full_dict['flag'] = 'SF'

    df = pd.DataFrame([full_dict])[ALL_FEATURES]
    default_map = {'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF'}
    for col in ['protocol_type', 'service', 'flag']:
        le = le_dict[col]
        df[col] = safe_transform(le, full_dict[col], default_map[col])

    df, _ = scale_features(df, scaler=scaler, fit=False)
    X_selected = selector.transform(df.values)
    return X_selected

def predict_sample(raw_input_dict, confidence_threshold=0.5):
    rf, iso, scaler, le_dict, selector = load_artifacts()
    X = preprocess_input(raw_input_dict)
    probs = rf.predict_proba(X)[0]
    max_prob = np.max(probs)
    pred_class = rf.classes_[np.argmax(probs)]

    if max_prob >= confidence_threshold:
        return pred_class, max_prob, 'known'
    else:
        anomaly = iso.predict(X)[0]
        if anomaly == -1:
            return "Zero-Day Attack", max_prob, 'anomaly'
        else:
            return "Normal (low confidence)", max_prob, 'normal'