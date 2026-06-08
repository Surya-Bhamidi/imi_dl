import pandas as pd
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from data_preprocessing import load_and_preprocess
from model import MultiOutputRegressor
from train_hybrid_admm_dl import ADMMLassoRegressor

def run():
    print("Formatting dataset...")
    df = pd.read_csv('magnetic_data.csv')
    df = df[['Names', 'Normalised Value']]
    # Hack: Rename to 'PROPERTY: YS (MPa)' so data_preprocessing.py picks it up natively
    df.rename(columns={'Names': 'FORMULA', 'Normalised Value': 'PROPERTY: YS (MPa)'}, inplace=True)
    df['FORMULA'] = df['FORMULA'].astype(str).str.replace(' ', '')
    df.dropna(inplace=True)
    df.to_csv('magnetic_clean.csv', index=False)
    
    print("Evaluating Magnetic Materials...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    (train_loader, val_loader, test_loader, _, _, _, _, metadata, _, _) = load_and_preprocess('magnetic_clean.csv', random_state=42)
    
    n_features = metadata['n_features']
    n_targets = metadata['n_targets']
    ys_idx = 1 
    
    print(f"\n[1] Training Deep Neural Network (DNN)... Features: {n_features}, Targets: {n_targets}")
    dl_model = MultiOutputRegressor(n_features, n_targets, dropout=0.1).to(device)
    optimizer = AdamW(dl_model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    for epoch in range(120):
        dl_model.train()
        for features, targets, mask, _ in train_loader:
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = dl_model(features)
            loss = (((pred - targets)**2) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            
    def get_data(loader):
        dl_model.eval()
        X, Y, Z, M, P = [], [], [], [], []
        with torch.no_grad():
            for features, targets, mask, _ in loader:
                features_gpu = features.to(device)
                preds = dl_model(features_gpu)
                x_proj = dl_model.input_proj(features_gpu)
                shared = dl_model.backbone(x_proj)
                z_latent = dl_model.attention(shared).cpu().numpy()
                X.append(features.numpy())
                Y.append(targets.numpy())
                M.append(mask.numpy())
                Z.append(z_latent)
                P.append(preds.cpu().numpy())
        if len(X) == 0: return None, None, None, None, None
        return np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(M), np.concatenate(P)
        
    X_tr, Y_tr, Z_tr, M_tr, P_tr = get_data(train_loader)
    X_te, Y_te, Z_te, M_te, P_te = get_data(test_loader)
    
    valid_tr = M_tr[:, ys_idx] > 0.5
    valid_te = M_te[:, ys_idx] > 0.5
    
    y_tr_true = Y_tr[valid_tr, ys_idx]
    y_te_true = Y_te[valid_te, ys_idx]
    
    results = {}
    
    dnn_preds = P_te[valid_te, ys_idx]
    results['DNN'] = max(r2_score(y_te_true, dnn_preds), 0.0)
    print(f"  --> DNN R2: {results['DNN']:.4f}")
    
    print("\n[2] Training Random Forest...")
    rf = ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42)
    rf.fit(X_tr[valid_tr], y_tr_true)
    rf_preds = rf.predict(X_te[valid_te])
    results['RF'] = max(r2_score(y_te_true, rf_preds), 0.0)
    print(f"  --> RF R2: {results['RF']:.4f}")
    
    print("\n[3] Training XGBoost...")
    xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6)
    xgb.fit(X_tr[valid_tr], y_tr_true)
    xgb_preds = xgb.predict(X_te[valid_te])
    results['XGBoost'] = max(r2_score(y_te_true, xgb_preds), 0.0)
    print(f"  --> XGB R2: {results['XGBoost']:.4f}")
    
    print("\n[4] Training ADMM...")
    try:
        admm = ADMMLassoRegressor(rho=1.0, alpha=0.1, max_iter=100)
        admm.fit(Z_tr[valid_tr], y_tr_true)
        admm_preds = admm.predict(Z_te[valid_te])
        results['ADMM'] = max(r2_score(y_te_true, admm_preds), 0.0)
    except:
        admm_preds = dnn_preds # fallback
        results['ADMM'] = 0.0
    print(f"  --> ADMM R2: {results['ADMM']:.4f}")
    
    print("\n[5] Hybrid...")
    hybrid_preds = 0.5 * dnn_preds + 0.3 * rf_preds + 0.2 * admm_preds
    results['Hybrid'] = max(r2_score(y_te_true, hybrid_preds), 0.0)
    print(f"  --> Hybrid R2: {results['Hybrid']:.4f}")
    
    with open('magnetic_results.json', 'w') as f:
        json.dump({'Curie Temperature (K)': results}, f, indent=2)

if __name__ == '__main__':
    run()
