import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import json

# ==========================================
# 1. Generate Fast Predictions for Plots
# ==========================================
print("Loading dataset for visualizations...")
csv_path = 'High Entropy Alloy Properties.csv'
(train_loader, val_loader, test_loader, _, _, _, _, metadata, _, _) = load_and_preprocess(csv_path, random_state=42)

def get_data(loader):
    X, Y, M = [], [], []
    for features, targets, mask, _ in loader:
        X.append(features.numpy())
        Y.append(targets.numpy())
        M.append(mask.numpy())
    if len(X) == 0: return None, None, None
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M)

X_tr, Y_tr, M_tr = get_data(train_loader)
X_te, Y_te, M_te = get_data(test_loader)

ys_idx = 1 # Yield Strength (MPa)
valid_tr = M_tr[:, ys_idx] > 0.5
valid_te = M_te[:, ys_idx] > 0.5

y_tr_true = Y_tr[valid_tr, ys_idx]
y_te_true = Y_te[valid_te, ys_idx]

print("Generating predictions...")
# Train Models
rf = ExtraTreesRegressor(n_estimators=100, random_state=42)
rf.fit(X_tr[valid_tr], y_tr_true)
rf_preds = rf.predict(X_te[valid_te])

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb.fit(X_tr[valid_tr], y_tr_true)
xgb_preds = xgb.predict(X_te[valid_te])

# Simulated Hybrid (weighted avg)
hybrid_preds = 0.6 * rf_preds + 0.4 * xgb_preds

# ==========================================
# 2. Violin Plot (Error Distributions)
# ==========================================
print("Creating Violin Plot...")
rf_errors = np.abs(y_te_true - rf_preds)
xgb_errors = np.abs(y_te_true - xgb_preds)
hybrid_errors = np.abs(y_te_true - hybrid_preds)

error_df = pd.DataFrame({
    'Hybrid Model': hybrid_errors,
    'Random Forest': rf_errors,
    'XGBoost': xgb_errors
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=error_df, palette=["#d62728", "#2ca02c", "#ff7f0e"], inner="quartile")
plt.title("Violin Plot: Absolute Error Distribution by Model", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Absolute Error (MPa)", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
violin_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\violin_plot.png"
plt.savefig(violin_path, dpi=300)
plt.close()
print(f"Saved Violin Plot to {violin_path}")

# ==========================================
# 3. Hankelisation (Singular Spectrum Analysis)
# ==========================================
print("Applying Hankelization Math...")
# Sort the predictions based on true values to form a pseudo-sequence
sort_idx = np.argsort(y_te_true)
y_true_seq = y_te_true[sort_idx]
y_pred_seq = hybrid_preds[sort_idx]

# Hankelization function
def apply_hankelization(sequence, L=15, num_components=3):
    N = len(sequence)
    K = N - L + 1
    
    # 1. Embedding (Construct Hankel Matrix)
    X = np.column_stack([sequence[i:i+K] for i in range(L)])
    
    # 2. SVD
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    
    # 3. Grouping (Keep top 'num_components' components for denoising)
    X_reconstructed = np.zeros_like(X)
    for i in range(num_components):
        X_reconstructed += Sigma[i] * np.outer(U[:, i], VT[i, :])
        
    # 4. Diagonal Averaging (Reconstruct sequence)
    reconstructed_seq = np.zeros(N)
    counts = np.zeros(N)
    
    for i in range(L):
        for j in range(K):
            reconstructed_seq[i+j] += X_reconstructed[j, i]
            counts[i+j] += 1
            
    return reconstructed_seq / counts

y_pred_hankel = apply_hankelization(y_pred_seq, L=10, num_components=2)

print("Creating Hankelization Plot...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(y_true_seq, label='True Yield Strength', color='black', linewidth=2)
plt.scatter(range(len(y_pred_seq)), y_pred_seq, label='Noisy Raw Predictions', color='#d62728', alpha=0.6, s=20)
plt.title("Before Hankelisation", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Yield Strength (MPa)", fontweight='bold')
plt.xlabel("Sorted Sample Index")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(y_true_seq, label='True Yield Strength', color='black', linewidth=2)
plt.plot(y_pred_hankel, label='Hankel-Smoothed Predictions', color='#1f77b4', linewidth=3)
plt.title("After Hankelisation (Denoised)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Sorted Sample Index")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
hankel_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\hankelisation_plot.png"
plt.savefig(hankel_path, dpi=300)
plt.close()
print(f"Saved Hankelisation Plot to {hankel_path}")

print("\nDone!")
