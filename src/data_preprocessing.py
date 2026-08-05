import pandas as pd
import joblib
from utils import load_nsl_kdd, encode_categorical, scale_features, split_data

def preprocess_pipeline(data_path, save_scaler=False, scaler_path='models/scaler.pkl',
                        save_label_encoders=False, le_path='models/label_encoders.pkl'):
    # Load raw data
    df = load_nsl_kdd(data_path)
    
    # Encode categorical
    df, le_dict = encode_categorical(df)
    
    # Separate features and target
    X = df.drop('attack_cat', axis=1)
    y = df['attack_cat']
    
    # Scale numerical features
    X_scaled, scaler = scale_features(X, fit=True)
    
    # Combine back for splitting (we'll split later)
    df_processed = X_scaled.copy()
    df_processed['attack_cat'] = y
    
    # Save scaler and encoders if needed
    if save_scaler:
        joblib.dump(scaler, scaler_path)
    if save_label_encoders:
        joblib.dump(le_dict, le_path)
    
    return df_processed, scaler, le_dict

if __name__ == "__main__":
    df_proc, scaler, le = preprocess_pipeline('data/KDDTrain+.txt', save_scaler=True, save_label_encoders=True)
    print(df_proc.head())
    
    # 👇 ADD THIS LINE TO SAVE THE PROCESSED DATA
    df_proc.to_csv('data/processed_data.csv', index=False)