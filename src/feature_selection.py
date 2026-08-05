import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

def select_features(X, y, k=20, save_selector=False, selector_path='models/selector.pkl'):
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    if save_selector:
        joblib.dump(selector, selector_path)
    
    return X_new, selected_features, selector

if __name__ == "__main__":
    df = pd.read_csv('data/processed_data.csv')  # assume we saved earlier
    X = df.drop('attack_cat', axis=1)
    y = df['attack_cat']
    X_sel, feat_list, sel = select_features(X, y, k=10, save_selector=True)
    print("Selected features:", feat_list)
    # Save the reduced dataframe
    df_sel = pd.DataFrame(X_sel, columns=feat_list)
    df_sel['attack_cat'] = y.values
    df_sel.to_csv('data/selected_features.csv', index=False)