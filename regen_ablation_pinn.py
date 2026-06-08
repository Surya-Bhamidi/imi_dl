import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\Updated_Plots"

properties = ["HV", "YS\n(MPa)", "UTS\n(MPa)", "Elong.\n(%)", "YM\n(GPa)"]
props_short = ["HV", "YS", "UTS", "Elong.", "YM"]

# ============================================================
# PLOT 1: Component Ablation WITH PINN (current paper values)
# These are the With-PINN constrained values from Table III
# ============================================================
dnn_scores    = [0.806, 0.695, 0.754, 0.696, 0.913]
et_scores     = [0.857, 0.808, 0.707, 0.577, 0.993]
admm_scores   = [0.866, 0.843, 0.775, 0.623, 0.988]
hybrid_scores = [0.855, 0.816, 0.825, 0.731, 0.975]

# Without PINN values (from _hybrid_results.json / PINN ablation table)
dnn_nopinn    = [0.9435, 0.8852, 0.8423, 0.7944, 0.9823]
hybrid_nopinn = [0.943,  0.913,  0.815,  0.753,  0.987]

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
print("[OK] Saved: mpea_ablation_chart.png (component ablation, post-PINN)")


# ============================================================
# PLOT 2: PINN Constraint Ablation (With vs Without)
# Shows the accuracy change when PINN constraints are applied
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
print("[OK] Saved: pinn_ablation_chart.png (PINN with vs without)")

print("\nDone! Both ablation plots saved to Updated_Plots/")
