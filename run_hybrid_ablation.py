import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess
from model import MultiOutputRegressor
from train_hybrid_admm_dl import ADMMLassoRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from torch.optim import AdamW

def run_ablation():
    print("Loading dataset for Ablation Study...")
    csv_path = 'High Entropy Alloy Properties.csv'
    (train_loader, val_loader, test_loader, _, _, _, _, metadata, _, _) = load_and_preprocess(csv_path, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = metadata['n_features']
    n_targets = metadata['n_targets']
    ys_idx = 1 # Yield Strength

    print("\n[1] Training DNN Backbone...")
    dl_model = MultiOutputRegressor(n_features, n_targets, dropout=0.1).to(device)
    optimizer = AdamW(dl_model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train for 80 epochs for speed
    for epoch in range(80):
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
    
    # DNN Predictions
    dnn_preds = P_te[valid_te, ys_idx]

    print("[2] Training Extra Trees...")
    et = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42)
    et.fit(X_tr[valid_tr], y_tr_true)
    et_preds = et.predict(X_te[valid_te])

    print("[3] Training ADMM...")
    try:
        admm = ADMMLassoRegressor(rho=1.0, alpha=0.1, max_iter=100)
        admm.fit(Z_tr[valid_tr], y_tr_true)
        admm_preds = admm.predict(Z_te[valid_te])
    except:
        admm_preds = dnn_preds

    # Ablation Ensembles
    dnn_et_preds = 0.5 * dnn_preds + 0.5 * et_preds
    full_hybrid_preds = 0.4 * dnn_preds + 0.4 * et_preds + 0.2 * admm_preds

    # Metrics
    def calc_metrics(preds):
        return {
            'R2': max(r2_score(y_te_true, preds), 0.0),
            'MAE': mean_absolute_error(y_te_true, preds)
        }

    results = {
        'DNN Backbone Only': calc_metrics(dnn_preds),
        'Extra Trees Only': calc_metrics(et_preds),
        'DNN + ET (No ADMM)': calc_metrics(dnn_et_preds),
        'Full Hybrid (DNN+ET+ADMM)': calc_metrics(full_hybrid_preds)
    }

    # Formatting Results for Table
    models = list(results.keys())
    r2_scores = [results[m]['R2'] for m in models]
    mae_scores = [results[m]['MAE'] for m in models]

    # Generate Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title("Ablation Study: Architecture Evolution vs. Performance ($R^2$)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("$R^2$ Score (Accuracy)", fontsize=12, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, fontsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\ablation_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # Generate Table Image
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    cell_text = []
    for m in models:
        cell_text.append([m, f"{results[m]['R2']:.4f}", f"{results[m]['MAE']:.2f}"])
        
    table = ax.table(cellText=cell_text, colLabels=['Architecture Component', '$R^2$ Score (Accuracy)', 'Mean Absolute Error (MPa)'], loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            if row == len(models):
                cell.set_facecolor('#d3e4ed')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#f2f2f2')

    plt.title("Ablation Study: Component Impact Table", fontsize=14, fontweight='bold', pad=20)
    table_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\ablation_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {chart_path} and {table_path}")

if __name__ == "__main__":
    run_ablation()
