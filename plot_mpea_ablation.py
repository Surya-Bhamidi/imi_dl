import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\Presentation_Plots"
os.makedirs(out_dir, exist_ok=True)

properties = ["HV", "YS\n(MPa)", "UTS\n(MPa)", "Elong.\n(%)", "YM\n(GPa)"]
props_short = ["HV", "YS", "UTS", "Elong.", "YM"]

# ============================================================
# PLOT 1: Architectural Ablation (DNN / ET / ADMM / Hybrid)
# ============================================================
dnn_scores   = [0.806, 0.695, 0.754, 0.696, 0.913]
et_scores    = [0.857, 0.808, 0.707, 0.577, 0.993]
admm_scores  = [0.866, 0.843, 0.775, 0.623, 0.988]
hybrid_scores = [0.855, 0.816, 0.825, 0.731, 0.975]

x = np.arange(len(properties))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 7))
bars1 = ax.bar(x - width*1.5, dnn_scores, width, label='DNN (PINN)', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x - width*0.5, et_scores, width, label='Extra Trees', color='#e67e22', edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + width*0.5, admm_scores, width, label='ADMM-Lasso', color='#2ecc71', edgecolor='black', linewidth=0.8)
bars4 = ax.bar(x + width*1.5, hybrid_scores, width, label='Hybrid PINN (Ours)', color='#e74c3c', edgecolor='black', linewidth=1.2)

ax.set_ylabel("$R^2$ Score", fontsize=14, fontweight='bold')
ax.set_title("Ablation Study: Component-Level Contribution (PINN Hybrid)", fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(properties, fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i, score in enumerate(hybrid_scores):
    ax.text(x[i] + width*1.5, score + 0.008, f"{score:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mpea_ablation_chart.png"), dpi=300)
plt.close()
print("✓ Saved: mpea_ablation_chart.png")

# ============================================================
# PLOT 2: Full Model Comparison (11 models)
# ============================================================
models = ['CNN-1D', 'BiLSTM', 'SVR', 'GRU', 'LSTM', 'RNN',
          'DNN\n(PINN)', 'Extra\nTrees', 'ADMM\nLasso', 'XGBoost', 'Hybrid\nPINN']
avg_r2 = [0.207, 0.274, 0.357, 0.524, 0.569, 0.643,
          0.773, 0.788, 0.819, 0.836, 0.841]
colors = ['#95a5a6']*6 + ['#3498db', '#e67e22', '#2ecc71', '#f39c12', '#e74c3c']

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(models, avg_r2, color=colors, edgecolor='black', linewidth=0.8)
bars[-1].set_linewidth(2.0)  # highlight hybrid

for bar, val in zip(bars, avg_r2):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel("Average $R^2$ (5 Properties)", fontsize=14, fontweight='bold')
ax.set_title("Model Comparison on MPEA Dataset (All 11 Architectures)", fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.axhline(y=0.841, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "model_comparison_chart.png"), dpi=300)
plt.close()
print("✓ Saved: model_comparison_chart.png")

# ============================================================
# PLOT 3: PINN Constraint Ablation (With vs Without)
# ============================================================
without_pinn = [0.943, 0.913, 0.815, 0.753, 0.987]
with_pinn    = [0.855, 0.816, 0.825, 0.731, 0.975]

x = np.arange(len(props_short))
width = 0.3

fig, ax = plt.subplots(figsize=(11, 7))
bars1 = ax.bar(x - width/2, without_pinn, width, label='Without PINN', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, with_pinn, width, label='With PINN (3 Constraints)', color='#e74c3c', edgecolor='black', linewidth=1.2)

# Add delta annotations
for i in range(len(props_short)):
    delta = with_pinn[i] - without_pinn[i]
    color = '#27ae60' if delta > 0 else '#c0392b'
    sign = '+' if delta > 0 else ''
    y_pos = max(without_pinn[i], with_pinn[i]) + 0.012
    ax.text(x[i], y_pos, f"Δ={sign}{delta:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))

ax.set_ylabel("$R^2$ Score", fontsize=14, fontweight='bold')
ax.set_title("Effect of Physics-Informed Constraints on Hybrid Model", fontsize=15, fontweight='bold', pad=25)
ax.set_xticks(x)
ax.set_xticklabels(props_short, fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(0.6, 1.08)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pinn_ablation_chart.png"), dpi=300)
plt.close()
print("✓ Saved: pinn_ablation_chart.png")

# ============================================================
# PLOT 4: Per-Property Heatmap (all models × all properties)
# ============================================================
model_names = ['CNN-1D', 'BiLSTM', 'SVR', 'GRU', 'LSTM', 'RNN',
               'DNN (PINN)', 'Extra Trees', 'ADMM-Lasso', 'XGBoost', 'Hybrid PINN']
prop_names = ['HV', 'YS (MPa)', 'UTS (MPa)', 'Elong. (%)', 'YM (GPa)']

data = np.array([
    [0.268, 0.120, 0.122, 0.111, 0.412],  # CNN-1D
    [0.440, 0.236, 0.253, 0.178, 0.263],  # BiLSTM
    [0.253, 0.283, 0.287, 0.126, 0.834],  # SVR
    [0.651, 0.397, 0.405, 0.444, 0.725],  # GRU
    [0.755, 0.441, 0.584, 0.343, 0.719],  # LSTM
    [0.726, 0.640, 0.581, 0.534, 0.731],  # RNN
    [0.806, 0.695, 0.754, 0.696, 0.913],  # DNN PINN
    [0.857, 0.808, 0.707, 0.577, 0.993],  # ET
    [0.866, 0.843, 0.775, 0.623, 0.988],  # ADMM
    [0.892, 0.807, 0.880, 0.601, 0.998],  # XGBoost
    [0.855, 0.816, 0.825, 0.731, 0.975],  # Hybrid PINN
])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)

ax.set_xticks(np.arange(len(prop_names)))
ax.set_yticks(np.arange(len(model_names)))
ax.set_xticklabels(prop_names, fontsize=11, fontweight='bold')
ax.set_yticklabels(model_names, fontsize=11, fontweight='bold')

for i in range(len(model_names)):
    for j in range(len(prop_names)):
        color = 'white' if data[i, j] < 0.4 else 'black'
        ax.text(j, i, f"{data[i,j]:.3f}", ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

ax.set_title("$R^2$ Heatmap: All Models × All Properties", fontsize=14, fontweight='bold', pad=15)
fig.colorbar(im, ax=ax, label="$R^2$ Score", shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "model_heatmap.png"), dpi=300)
plt.close()
print("✓ Saved: model_heatmap.png")

print("\n✅ All 4 plots saved to Presentation_Plots/")
