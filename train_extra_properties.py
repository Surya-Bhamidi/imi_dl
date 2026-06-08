"""Train models on additional MPEA properties: Calculated Density, Exp Young's Modulus, Plastic Elongation."""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import extract_composition_features, encode_categoricals, prepare_numeric_features

EXTRA_TARGETS = {
    'PROPERTY: Calculated Density (g/cm$^3$)': 'Calculated Density (g/cm³)',
    'PROPERTY: Exp. Young modulus (GPa)': 'Exp. Young Modulus (GPa)',
    'PROPERTY: Elongation plastic (%)': 'Plastic Elongation (%)',
}

def main():
    df = pd.read_csv('MPEA_dataset_clean.csv')
    
    # Build features (same as main pipeline)
    comp_features, domain_features = extract_composition_features(df)
    cat_features, _ = encode_categoricals(df)
    num_features = prepare_numeric_features(df)
    X = pd.concat([comp_features, domain_features, cat_features, num_features], axis=1)
    
    results = {}
    
    for col, name in EXTRA_TARGETS.items():
        if col not in df.columns:
            print(f"Skipping {name}: column not found")
            continue
        y = pd.to_numeric(df[col], errors='coerce')
        valid = y.notna()
        n_valid = valid.sum()
        if n_valid < 50:
            print(f"Skipping {name}: only {n_valid} samples")
            continue
        
        X_v = X[valid].values.astype(np.float64)
        y_v = y[valid].values.astype(np.float64)
        
        X_train, X_test, y_train, y_test = train_test_split(X_v, y_v, test_size=0.2, random_state=42)
        
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        prop_results = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        prop_results['RF'] = round(r2_score(y_test, rf_pred), 4)
        
        # Extra Trees
        et = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        et.fit(X_train_s, y_train)
        et_pred = et.predict(X_test_s)
        prop_results['ET'] = round(r2_score(y_test, et_pred), 4)
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
        xgb.fit(X_train_s, y_train)
        xgb_pred = xgb.predict(X_test_s)
        prop_results['XGBoost'] = round(r2_score(y_test, xgb_pred), 4)
        
        # Hybrid (blend ET + XGBoost as proxy since DNN needs full retraining)
        hybrid_pred = 0.5 * et_pred + 0.5 * xgb_pred
        prop_results['Hybrid'] = round(r2_score(y_test, hybrid_pred), 4)
        
        # MAE for hybrid
        prop_results['Hybrid_MAE'] = round(mean_absolute_error(y_test, hybrid_pred), 4)
        
        results[name] = prop_results
        print(f"\n{name} ({n_valid} samples):")
        for model, score in prop_results.items():
            print(f"  {model}: {score}")
    
    with open('extra_property_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: extra_property_results.json")

if __name__ == "__main__":
    main()
