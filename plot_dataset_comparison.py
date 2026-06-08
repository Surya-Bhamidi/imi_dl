import matplotlib.pyplot as plt
import numpy as np
import json

# Data from MPEA run (Yield Strength R2)
mpea_scores = [0.9126, 0.8852, 0.8904, 0.8067, 0.8839]

with open('matbench_results.json', 'r') as f:
    matbench_data = json.load(f)
    
models = ['Hybrid', 'DNN', 'Random Forest', 'XGBoost', 'ADMM']
# Matbench results
matbench_scores = [
    matbench_data['Hybrid Model (Ours)'],
    matbench_data['DNN (Deep Learning)'],
    matbench_data['Random Forest'],
    matbench_data['XGBoost'],
    matbench_data['ADMM']
]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, mpea_scores, width, label='MPEA Dataset (~800 samples)', color='#1f77b4')
rects2 = ax.bar(x + width/2, matbench_scores, width, label='Matbench Steels (~300 samples)', color='#ff7f0e')

ax.set_ylabel('R² Score (Yield Strength)')
ax.set_title('Cross-Dataset Comparison: Yield Strength Prediction vs. Dataset Size', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)

for c in ax.containers:
    ax.bar_label(c, fmt='%.3f', padding=3, fontsize=10)

# Add a horizontal line at 0 for aesthetics
ax.axhline(0, color='black', linewidth=0.8)

fig.tight_layout()
plt.savefig('cross_dataset_chart.png', dpi=300)
print("Saved cross_dataset_chart.png")
