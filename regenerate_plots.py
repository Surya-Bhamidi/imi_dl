import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

# Data for the 6 comparable models only
models = ['SVR', 'DNN\n(PINN)', 'Extra\nTrees', 'ADMM\nLasso', 'XGBoost', 'Hybrid\nPINN']
avg_r2 = [0.357, 0.773, 0.788, 0.819, 0.836, 0.841]

properties = ['HV', 'YS (MPa)', 'UTS (MPa)', 'Elong. (%)', 'YM (GPa)']
heatmap_models = ['SVR', 'DNN (PINN)', 'Extra Trees', 'ADMM-Lasso', 'XGBoost', 'Hybrid PINN']
scores = np.array([
    [0.253, 0.283, 0.287, 0.126, 0.834],  # SVR
    [0.806, 0.695, 0.754, 0.696, 0.913],  # DNN (PINN)
    [0.857, 0.808, 0.707, 0.577, 0.993],  # Extra Trees
    [0.866, 0.843, 0.775, 0.623, 0.988],  # ADMM-Lasso
    [0.892, 0.807, 0.880, 0.601, 0.998],  # XGBoost
    [0.855, 0.816, 0.825, 0.731, 0.975],  # Hybrid PINN
])

# ── Plot 1: Model Comparison Bar Chart ──
fig, ax = plt.subplots(figsize=(12, 6))

# Color scheme: grey for baselines, colored for components + proposed
colors = ['#9E9E9E',   # SVR - grey
          '#4285F4',   # DNN (PINN) - blue
          '#FB8C00',   # Extra Trees - orange
          '#43A047',   # ADMM-Lasso - green
          '#FB8C00',   # XGBoost - orange (external baseline)
          '#D32F2F']   # Hybrid PINN - red (ours)

# Better: distinguish external baselines vs components vs ours
colors = ['#9E9E9E',   # SVR - grey (external baseline)
          '#5C9BD5',   # DNN (PINN) - blue (component)
          '#ED7D31',   # Extra Trees - orange (component)
          '#70AD47',   # ADMM-Lasso - green (component)
          '#FFC000',   # XGBoost - gold (external baseline)
          '#C00000']   # Hybrid PINN - dark red (ours)

bars = ax.bar(range(len(models)), avg_r2, color=colors, width=0.65,
              edgecolor='black', linewidth=0.8, zorder=3)

# Add value labels on top of each bar
for bar, val in zip(bars, avg_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Hybrid reference line
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
plt.savefig('Updated_Plots/model_comparison_chart.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] model_comparison_chart.png saved")


# ── Plot 2: Heatmap ──
fig, ax = plt.subplots(figsize=(10, 6))

# Custom colormap: red (low) -> yellow (mid) -> green (high)
from matplotlib.colors import LinearSegmentedColormap
cmap_colors = ['#D32F2F', '#FF6F00', '#FFC107', '#8BC34A', '#2E7D32']
cmap = LinearSegmentedColormap.from_list('r2_cmap', cmap_colors, N=256)

im = ax.imshow(scores, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)

# Add text annotations
for i in range(len(heatmap_models)):
    for j in range(len(properties)):
        val = scores[i, j]
        # White text on dark cells, black on light
        text_color = 'white' if val < 0.4 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=text_color)

ax.set_xticks(range(len(properties)))
ax.set_xticklabels(properties, fontsize=12, fontweight='bold')
ax.set_yticks(range(len(heatmap_models)))
ax.set_yticklabels(heatmap_models, fontsize=12, fontweight='bold')

ax.set_title('$R^2$ Heatmap: All Models × All Properties', fontsize=15, fontweight='bold', pad=15)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('$R^2$ Score', fontsize=12, fontweight='bold')

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, len(properties), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(heatmap_models), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', size=0)

plt.tight_layout()
plt.savefig('Updated_Plots/model_heatmap.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] model_heatmap.png saved")

print("\nDone! Both plots regenerated with 6 comparable models only.")
