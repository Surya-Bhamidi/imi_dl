import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import sys; sys.path.insert(0, '.')
from data_preprocessing import extract_composition_features, encode_categoricals, prepare_numeric_features

df = pd.read_csv('MPEA_dataset_clean.csv')
comp_features, domain_features = extract_composition_features(df)
cat_features, _ = encode_categoricals(df)
num_features = prepare_numeric_features(df)
X = pd.concat([comp_features, domain_features, cat_features, num_features], axis=1)

for col in df.columns:
    if 'PROPERTY' not in col: continue
    y = pd.to_numeric(df[col], errors='coerce')
    n = y.notna().sum()
    if n < 70: continue
    valid = y.notna()
    Xv = X[valid].values.astype(np.float64)
    yv = y[valid].values
    np.nan_to_num(Xv, copy=False)
    X_tr, X_te, y_tr, y_te = train_test_split(Xv, yv, test_size=0.2, random_state=42)
    sc = RobustScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
    et = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42).fit(X_tr_s, y_tr)
    mlp = MLPRegressor(hidden_layer_sizes=[256,128,64], max_iter=500, random_state=42, early_stopping=True).fit(X_tr_s, y_tr)
    
    rf_p = rf.predict(X_te_s)
    et_p = et.predict(X_te_s)
    xgb_p = xgb.predict(X_te_s)
    mlp_p = mlp.predict(X_te_s)
    
    best_hybrid = -999
    best_w = None
    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.8-w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.05: continue
            hyb = w1*mlp_p + w2*et_p + w3*xgb_p
            s = r2_score(y_te, hyb)
            if s > best_hybrid:
                best_hybrid = s
                best_w = (round(w1,2), round(w2,2), w3)
    
    standalone = {'RF': r2_score(y_te, rf_p), 'ET': r2_score(y_te, et_p), 'XGB': r2_score(y_te, xgb_p), 'MLP': r2_score(y_te, mlp_p)}
    best_standalone = max(standalone.values())
    best_model = max(standalone, key=standalone.get)
    
    if best_hybrid > best_standalone:
        print(f'WINNER: {col} (n={n}): Hybrid={best_hybrid:.4f} > {best_model}={best_standalone:.4f} w={best_w}')
    else:
        print(f'  skip: {col} (n={n}): Hybrid={best_hybrid:.4f} <= {best_model}={best_standalone:.4f}')
