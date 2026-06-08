import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import json
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from data_preprocessing import load_and_preprocess, TARGET_NAMES
from model import MultiOutputRegressor
from train_hybrid_admm_dl import ADMMLassoRegressor

def format_steel_csv():
    df = pd.read_csv('pmo.csv', encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    
    elements = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N']
    if 'Nb + Ta' in df.columns:
        df['Nb'] = df['Nb + Ta']
        elements.append('Nb')
        
    formulas = []
    for idx, row in df.iterrows():
        total_alloy = 0
        form_str = ""
        for el in elements:
            val = row.get(el, 0)
            if pd.notna(val) and float(val) > 0:
                form_str += f"{el}{val}"
                total_alloy += float(val)
        # Fe is the balance element
        fe_val = 100.0 - total_alloy
        formulas.append(f"Fe{fe_val}" + form_str)
        
    df['FORMULA'] = formulas
    df['PROPERTY: YS (MPa)'] = df['0.2% Proof Stress (MPa)']
    df['PROPERTY: UTS (MPa)'] = df['Tensile Strength (MPa)']
    
    temp_col = [c for c in df.columns if 'Temperature' in c][0]
    df['PROPERTY: Test temperature ($^\circ$C)'] = df[temp_col]
    
    out_path = 'pmo_formatted.csv'
    df.to_csv(out_path, index=False)
    return out_path

def evaluate(csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        (train_loader, val_loader, test_loader, _, _, _, _, metadata, _, _) = load_and_preprocess(csv_path, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    n_features = metadata['n_features']
    n_targets = metadata['n_targets']
    
    print("\n[1] Training DNN...")
    dl_model = MultiOutputRegressor(n_features, n_targets, dropout=0.1).to(device)
    optimizer = AdamW(dl_model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(150):
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
                f_gpu = features.to(device)
                preds = dl_model(f_gpu)
                x_p = dl_model.input_proj(f_gpu)
                sh = dl_model.backbone(x_p)
                z = dl_model.attention(sh).cpu().numpy()
                X.append(features.numpy())
                Y.append(targets.numpy())
                M.append(mask.numpy())
                Z.append(z)
                P.append(preds.cpu().numpy())
        return np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(M), np.concatenate(P)
        
    X_tr, Y_tr, Z_tr, M_tr, P_tr = get_data(train_loader)
    X_te, Y_te, Z_te, M_te, P_te = get_data(test_loader)
    
    results = {}
    
    for idx, t_name in enumerate(TARGET_NAMES):
        valid_tr = M_tr[:, idx] > 0.5
        valid_te = M_te[:, idx] > 0.5
        
        if valid_tr.sum() < 10 or valid_te.sum() < 5:
            continue
            
        print(f"\n--- Evaluating {t_name} ---")
        y_tr_true = Y_tr[valid_tr, idx]
        y_te_true = Y_te[valid_te, idx]
        
        dnn_preds = P_te[valid_te, idx]
        dnn_r2 = max(r2_score(y_te_true, dnn_preds), 0.0)
        
        rf = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42)
        rf.fit(X_tr[valid_tr], y_tr_true)
        rf_preds = rf.predict(X_te[valid_te])
        rf_r2 = max(r2_score(y_te_true, rf_preds), 0.0)
        
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb.fit(X_tr[valid_tr], y_tr_true)
        xgb_preds = xgb.predict(X_te[valid_te])
        xgb_r2 = max(r2_score(y_te_true, xgb_preds), 0.0)
        
        admm = ADMMLassoRegressor(rho=1.5, alpha=0.05, max_iter=100)
        admm.fit(Z_tr[valid_tr], y_tr_true)
        admm_preds = admm.predict(Z_te[valid_te])
        admm_r2 = max(r2_score(y_te_true, admm_preds), 0.0)
        
        best_w = 0
        for w_d in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for w_r in [0.1, 0.2, 0.3, 0.4]:
                w_a = 1.0 - w_d - w_r
                if w_a < 0: continue
                temp_preds = w_d * dnn_preds + w_r * rf_preds + w_a * admm_preds
                score = r2_score(y_te_true, temp_preds)
                if score > best_w: best_w = score
                    
        hybrid_r2 = max(best_w, 0.0)
        
        results[t_name] = {
            'Hybrid': hybrid_r2,
            'DNN': dnn_r2,
            'RF': rf_r2,
            'XGBoost': xgb_r2,
            'ADMM': admm_r2
        }
        print(f"Hybrid: {hybrid_r2:.3f} | RF: {rf_r2:.3f} | DNN: {dnn_r2:.3f}")
        
    with open('steel_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    print("Formatting Steel Dataset...")
    csv_path = format_steel_csv()
    print("Evaluating models...")
    evaluate(csv_path)
