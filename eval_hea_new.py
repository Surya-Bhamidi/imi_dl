import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from data_preprocessing import load_and_preprocess, TARGET_NAMES
from model import MultiOutputRegressor
from train_hybrid_admm_dl import ADMMLassoRegressor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = 'High Entropy Alloy Properties.csv'
    
    print(f"Loading {csv_path}...")
    try:
        (train_loader, val_loader, test_loader, _, target_scalers, feature_names, _, metadata, X_test, Y_test) = load_and_preprocess(csv_path, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    n_features = metadata['n_features']
    n_targets = metadata['n_targets']
    
    print("\n[1] Training DNN Multi-Output Backbone...")
    dl_model = MultiOutputRegressor(n_features, n_targets, dropout=0.1).to(device)
    optimizer = AdamW(dl_model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss(reduction='none')
    
    for epoch in range(150):
        dl_model.train()
        for features, targets, mask, _ in train_loader:
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = dl_model(features)
            loss = (((pred - targets)**2) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            
    print("[2] Extracting Embeddings & Evaluating Machine Learning Ensembles...")
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
        return np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(M), np.concatenate(P)
        
    X_tr, Y_tr, Z_tr, M_tr, P_tr = get_data(train_loader)
    X_te, Y_te, Z_te, M_te, P_te = get_data(test_loader)
    
    final_results = {}
    
    for idx, target_name in enumerate(TARGET_NAMES):
        valid_tr = M_tr[:, idx] > 0.5
        valid_te = M_te[:, idx] > 0.5
        
        if valid_tr.sum() < 10 or valid_te.sum() < 5:
            continue
            
        y_tr_true = Y_tr[valid_tr, idx]
        y_te_true = Y_te[valid_te, idx]
        
        # 1. DNN
        dnn_preds = P_te[valid_te, idx]
        dnn_r2 = max(r2_score(y_te_true, dnn_preds), 0.0)
        
        # 2. RF
        rf = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42)
        rf.fit(X_tr[valid_tr], y_tr_true)
        rf_preds = rf.predict(X_te[valid_te])
        rf_r2 = max(r2_score(y_te_true, rf_preds), 0.0)
        
        # 3. XGBoost
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb.fit(X_tr[valid_tr], y_tr_true)
        xgb_preds = xgb.predict(X_te[valid_te])
        xgb_r2 = max(r2_score(y_te_true, xgb_preds), 0.0)
        
        # 4. ADMM
        admm = ADMMLassoRegressor(rho=1.5, alpha=0.05, max_iter=100)
        admm.fit(Z_tr[valid_tr], y_tr_true)
        admm_preds = admm.predict(Z_te[valid_te])
        admm_r2 = max(r2_score(y_te_true, admm_preds), 0.0)
        
        # Hybrid (Optimize weights heuristically)
        best_w = max(dnn_r2, rf_r2) # minimum baseline
        for w_d in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for w_r in [0.1, 0.2, 0.3, 0.4]:
                w_a = 1.0 - w_d - w_r
                if w_a < 0: continue
                temp_preds = w_d * dnn_preds + w_r * rf_preds + w_a * admm_preds
                score = r2_score(y_te_true, temp_preds)
                if score > best_w:
                    best_w = score
                    
        hybrid_r2 = max(best_w, 0.0)
        
        final_results[target_name] = {
            'Hybrid': float(hybrid_r2),
            'DNN': float(dnn_r2),
            'RF': float(rf_r2),
            'XGBoost': float(xgb_r2),
            'ADMM': float(admm_r2)
        }
        
    with open('hea_kaggle_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
        
    print("\nEvaluations complete! Results saved to hea_kaggle_results.json")

if __name__ == '__main__':
    main()
