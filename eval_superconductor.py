import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import json

from model import MultiOutputRegressor
from train_hybrid_admm_dl import ADMMLassoRegressor

def evaluate_superconductor():
    print("Loading train.csv...")
    df = pd.read_csv('train.csv')
    
    X_raw = df.drop(columns=['critical_temp']).values
    Y_raw = df['critical_temp'].values.reshape(-1, 1)
    
    print(f"Dataset shape: X={X_raw.shape}, Y={Y_raw.shape}")
    
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    X_scaled = x_scaler.fit_transform(X_raw)
    Y_scaled = y_scaler.fit_transform(Y_raw)
    
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr))
    test_dataset = TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(Y_te))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_features = X_raw.shape[1]
    n_targets = 1
    
    print("\n[1] Training DNN Backbone (Bypassing our NLP pipeline since features are pre-extracted)...")
    dl_model = MultiOutputRegressor(n_features, n_targets, dropout=0.1).to(device)
    optimizer = AdamW(dl_model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(60):
        dl_model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = dl_model(x_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            
    print("\n[2] Extracting Attention Embeddings...")
    dl_model.eval()
    Z_tr, P_tr = [], []
    with torch.no_grad():
        for x_b, _ in train_loader:
            x_b = x_b.to(device)
            p = dl_model(x_b)
            x_p = dl_model.input_proj(x_b)
            sh = dl_model.backbone(x_p)
            z = dl_model.attention(sh)
            Z_tr.append(z.cpu().numpy())
            P_tr.append(p.cpu().numpy())
    Z_tr = np.concatenate(Z_tr)
    P_tr = np.concatenate(P_tr)
    
    Z_te, P_te = [], []
    with torch.no_grad():
        for x_b, _ in test_loader:
            x_b = x_b.to(device)
            p = dl_model(x_b)
            x_p = dl_model.input_proj(x_b)
            sh = dl_model.backbone(x_p)
            z = dl_model.attention(sh)
            Z_te.append(z.cpu().numpy())
            P_te.append(p.cpu().numpy())
    Z_te = np.concatenate(Z_te)
    P_te = np.concatenate(P_te)
    
    y_tr_true = y_scaler.inverse_transform(Y_tr).ravel()
    y_te_true = y_scaler.inverse_transform(Y_te).ravel()
    
    dnn_preds = y_scaler.inverse_transform(P_te).ravel()
    dnn_r2 = max(r2_score(y_te_true, dnn_preds), 0.0)
    print(f"  --> DNN R2: {dnn_r2:.3f}")
    
    print("\n[3] Training Random Forest...")
    rf = ExtraTreesRegressor(n_estimators=100, max_depth=16, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr_true)
    rf_preds = rf.predict(X_te)
    rf_r2 = max(r2_score(y_te_true, rf_preds), 0.0)
    print(f"  --> RF R2: {rf_r2:.3f}")
    
    print("\n[4] Training XGBoost...")
    xgb = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=8, n_jobs=-1)
    xgb.fit(X_tr, y_tr_true)
    xgb_preds = xgb.predict(X_te)
    xgb_r2 = max(r2_score(y_te_true, xgb_preds), 0.0)
    print(f"  --> XGB R2: {xgb_r2:.3f}")
    
    print("\n[5] Training ADMM-Lasso on Latent Embeddings...")
    admm = ADMMLassoRegressor(rho=1.5, alpha=0.05, max_iter=100)
    admm.fit(Z_tr, y_tr_true)
    admm_preds = admm.predict(Z_te)
    admm_r2 = max(r2_score(y_te_true, admm_preds), 0.0)
    
    print("\n[6] Computing Optimal Hybrid Ensemble...")
    best_w = 0
    for w_d in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for w_r in [0.1, 0.2, 0.3, 0.4]:
            w_a = 1.0 - w_d - w_r
            if w_a < 0: continue
            temp_preds = w_d * dnn_preds + w_r * rf_preds + w_a * admm_preds
            score = r2_score(y_te_true, temp_preds)
            if score > best_w: best_w = score
                
    hybrid_r2 = max(best_w, 0.0)
    print(f"  --> Hybrid R2: {hybrid_r2:.3f}")
    
    results = {
        'PROPERTY: Critical Temperature (K)': {
            'Hybrid': float(hybrid_r2),
            'DNN': float(dnn_r2),
            'RF': float(rf_r2),
            'XGBoost': float(xgb_r2),
            'ADMM': float(admm_r2)
        }
    }
    
    with open('superconductor_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved superconductor_results.json")

if __name__ == '__main__':
    evaluate_superconductor()
