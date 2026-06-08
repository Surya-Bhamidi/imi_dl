import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

out_dir = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\Final_Plots"
os.makedirs(out_dir, exist_ok=True)

properties = ["HV", "YS\n(MPa)", "UTS\n(MPa)", "Elong.\n(%)", "YM\n(GPa)"]
props_short = ["HV", "YS", "UTS", "Elong.", "YM"]

# ============================================================
# PLOT 1: Component Ablation (DNN / ET / ADMM / Hybrid) - Post-PINN
# ============================================================
dnn_scores    = [0.806, 0.695, 0.754, 0.696, 0.913]
et_scores     = [0.857, 0.808, 0.707, 0.577, 0.993]
admm_scores   = [0.866, 0.843, 0.775, 0.623, 0.988]
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
plt.savefig(os.path.join(out_dir, "mpea_ablation_chart.png"), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] mpea_ablation_chart.png")


# ============================================================
# PLOT 2: PINN Constraint Ablation (With vs Without)
# ============================================================
without_pinn = [0.943, 0.913, 0.815, 0.753, 0.987]
with_pinn    = [0.855, 0.816, 0.825, 0.731, 0.975]

x = np.arange(len(props_short))
width = 0.3

fig, ax = plt.subplots(figsize=(11, 7))
bars1 = ax.bar(x - width/2, without_pinn, width, label='Without PINN', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, with_pinn, width, label='With PINN (3 Constraints)', color='#e74c3c', edgecolor='black', linewidth=1.2)

for i in range(len(props_short)):
    delta = with_pinn[i] - without_pinn[i]
    color = '#27ae60' if delta > 0 else '#c0392b'
    sign = '+' if delta > 0 else ''
    y_pos = max(without_pinn[i], with_pinn[i]) + 0.012
    ax.text(x[i], y_pos, f"d={sign}{delta:.3f}", ha='center', va='bottom',
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
plt.savefig(os.path.join(out_dir, "pinn_ablation_chart.png"), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] pinn_ablation_chart.png")


# ============================================================
# PLOT 3: Model Comparison Bar Chart (6 models only)
# ============================================================
models = ['SVR', 'DNN\n(PINN)', 'Extra\nTrees', 'ADMM\nLasso', 'XGBoost', 'Hybrid\nPINN']
avg_r2 = [0.357, 0.773, 0.788, 0.819, 0.836, 0.841]

colors = ['#9E9E9E', '#5C9BD5', '#ED7D31', '#70AD47', '#FFC000', '#C00000']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(models)), avg_r2, color=colors, width=0.65,
              edgecolor='black', linewidth=0.8, zorder=3)

for bar, val in zip(bars, avg_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(y=0.841, color='#C00000', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylabel('Average $R^2$ (5 Properties)', fontsize=13, fontweight='bold')
ax.set_title('Model Comparison on MPEA Dataset (6 Architectures)', fontsize=15, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "model_comparison_chart.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] model_comparison_chart.png")


# ============================================================
# PLOT 4: Heatmap (6 models x 5 properties)
# ============================================================
heatmap_models = ['SVR', 'DNN (PINN)', 'Extra Trees', 'ADMM-Lasso', 'XGBoost', 'Hybrid PINN']
prop_names = ['HV', 'YS (MPa)', 'UTS (MPa)', 'Elong. (%)', 'YM (GPa)']
scores = np.array([
    [0.253, 0.283, 0.287, 0.126, 0.834],
    [0.806, 0.695, 0.754, 0.696, 0.913],
    [0.857, 0.808, 0.707, 0.577, 0.993],
    [0.866, 0.843, 0.775, 0.623, 0.988],
    [0.892, 0.807, 0.880, 0.601, 0.998],
    [0.855, 0.816, 0.825, 0.731, 0.975],
])

cmap_colors = ['#D32F2F', '#FF6F00', '#FFC107', '#8BC34A', '#2E7D32']
cmap = LinearSegmentedColormap.from_list('r2_cmap', cmap_colors, N=256)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(scores, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)

for i in range(len(heatmap_models)):
    for j in range(len(prop_names)):
        val = scores[i, j]
        text_color = 'white' if val < 0.4 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=text_color)

ax.set_xticks(range(len(prop_names)))
ax.set_xticklabels(prop_names, fontsize=12, fontweight='bold')
ax.set_yticks(range(len(heatmap_models)))
ax.set_yticklabels(heatmap_models, fontsize=12, fontweight='bold')
ax.set_title('$R^2$ Heatmap: All Models x All Properties', fontsize=15, fontweight='bold', pad=15)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('$R^2$ Score', fontsize=12, fontweight='bold')

ax.set_xticks(np.arange(-0.5, len(prop_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(heatmap_models), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', size=0)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "model_heatmap.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] model_heatmap.png")


# ============================================================
# Copy unchanged plots from Updated_Plots
# ============================================================
import shutil
src = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\Updated_Plots"
for f in ["violin_plot.png", "xai_feature_prominence.png"]:
    src_path = os.path.join(src, f)
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(out_dir, f))
        print(f"[OK] Copied {f}")

print("\nAll plots saved to Final_Plots/")
