import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_nsl_kdd(path):
    """Load NSL-KDD dataset (assumes CSV with header)"""
    columns = [
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
        'dst_host_srv_rerror_rate', 'attack_cat', 'difficulty_level'
    ]
    df = pd.read_csv(path, names=columns)
    # drop difficulty_level (not needed)
    df.drop('difficulty_level', axis=1, inplace=True)
    return df

def encode_categorical(df):
    """Encode categorical columns: protocol_type, service, flag"""
    le_dict = {}
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def scale_features(df, scaler=None, fit=False):
    """Scale numerical features (all except attack_cat and the three categorical)"""
    num_cols = [col for col in df.columns if col not in ['attack_cat', 'protocol_type', 'service', 'flag']]
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('attack_cat', axis=1)
    y = df['attack_cat']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)