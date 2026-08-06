import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


def build_autoencoder(input_dim, encoding_dim=None):
    """Simple symmetric autoencoder. Bottleneck (encoding_dim) forces the
    network to learn a compressed representation of *normal* traffic only,
    so unusual/attack traffic reconstructs poorly -> high error -> anomaly."""
    if encoding_dim is None:
        encoding_dim = max(2, input_dim // 4)
    hidden_dim = max(input_dim // 2, encoding_dim)

    inp = Input(shape=(input_dim,))
    encoded = Dense(hidden_dim, activation='relu')(inp)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(hidden_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=inp, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def train_models(data_path, holdout_class='u2r',
                  xgb_path='models/xgb_model.pkl',
                  encoder_path='models/label_encoder.pkl',
                  ae_path='models/autoencoder_model.keras',
                  scaler_path='models/ae_scaler.pkl',
                  threshold_path='models/ae_threshold.pkl'):
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

    # ---- XGBoost (supervised multiclass classifier) ----
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    os.makedirs(os.path.dirname(encoder_path) or '.', exist_ok=True)
    joblib.dump(le, encoder_path)
    print("Label classes seen during training:", list(le.classes_))

    print("Training XGBoost classifier...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train_enc)
    os.makedirs(os.path.dirname(xgb_path) or '.', exist_ok=True)
    joblib.dump(xgb_clf, xgb_path)
    print("XGBoost model saved to", xgb_path)

    # ---- Autoencoder (unsupervised anomaly detector, replaces Isolation Forest) ----
    print("Training autoencoder on normal traffic only...")
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    os.makedirs(os.path.dirname(scaler_path) or '.', exist_ok=True)
    joblib.dump(scaler, scaler_path)

    autoencoder = build_autoencoder(input_dim=X_normal_scaled.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(
        X_normal_scaled, X_normal_scaled,   # reconstruct its own input
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    os.makedirs(os.path.dirname(ae_path) or '.', exist_ok=True)
    autoencoder.save(ae_path)

    # Anomaly threshold = 95th percentile of reconstruction error on normal data
    recon_train = autoencoder.predict(X_normal_scaled, verbose=0)
    train_mse = np.mean(np.square(X_normal_scaled - recon_train), axis=1)
    threshold = float(np.percentile(train_mse, 95))
    joblib.dump(threshold, threshold_path)
    print(f"Autoencoder saved to {ae_path}")
    print(f"Anomaly threshold (95th percentile reconstruction MSE): {threshold:.6f}")

    # Evaluate on holdout class
    X_holdout = X[y == holdout_class]
    y_holdout = y[y == holdout_class]
    if len(X_holdout) > 0:
        pred_xgb_enc = xgb_clf.predict(X_holdout)
        pred_xgb = le.inverse_transform(pred_xgb_enc)

        X_holdout_scaled = scaler.transform(X_holdout)
        recon_holdout = autoencoder.predict(X_holdout_scaled, verbose=0)
        holdout_mse = np.mean(np.square(X_holdout_scaled - recon_holdout), axis=1)
        anomaly_flag = holdout_mse > threshold

        print(f"Holdout class '{holdout_class}' samples: {len(X_holdout)}")
        print("XGB predictions (distribution):", np.unique(pred_xgb, return_counts=True))
        print("Anomaly detection (zero-day flagged):", np.sum(anomaly_flag), "out of", len(X_holdout))

    return xgb_clf, autoencoder, le


if __name__ == "__main__":
    train_models('data/selected_features.csv')
