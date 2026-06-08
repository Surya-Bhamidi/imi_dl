import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

out_dir = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\Final_Plots"

df = pd.read_csv(r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\MPEA_dataset_clean.csv")

# Select the 5 mechanical property targets
props = {
    'HV': 'PROPERTY: HV',
    'YS (MPa)': 'PROPERTY: YS (MPa)',
    'UTS (MPa)': 'PROPERTY: UTS (MPa)',
    'Elongation (%)': 'PROPERTY: Elongation (%)',
    'Young\'s Mod. (GPa)': 'PROPERTY: Calculated Young modulus (GPa)'
}

prop_df = df[[v for v in props.values()]].copy()
prop_df.columns = list(props.keys())

# Use pairwise correlation (each cell uses all available pairs)
corr = prop_df.corr(method='pearson', min_periods=10)
n_total = len(prop_df.dropna(how='all'))  # samples with at least one property

fig, ax = plt.subplots(figsize=(8, 6.5))

# Custom diverging colormap
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('custom_div',
    ['#1a3a6b', '#3a7bc8', '#ffffff', '#c0392b', '#7b1a1a'], N=256)

im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

labels = list(props.keys())
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', rotation=30, ha='right')
ax.set_yticklabels(labels, fontsize=12, fontweight='bold')

# Annotate each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.values[i, j]
        text_color = 'white' if abs(val) > 0.55 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=text_color)

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', size=0)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Pearson Correlation Coefficient', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

ax.set_title('Inter-Property Correlation Matrix\n(MPEA Dataset, $n$ = ' + str(n_total) + ' samples)',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
out_path = out_dir + r"\correlation_heatmap.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Saved: {out_path}")
print(f"\nCorrelation matrix:\n{corr.round(2)}")
