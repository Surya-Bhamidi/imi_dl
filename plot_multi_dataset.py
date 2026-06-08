import matplotlib.pyplot as plt
import numpy as np
import json

with open('multi_dataset_results.json', 'r') as f:
    data = json.load(f)
    
datasets = ['MPEA Alloys\n(Yield Strength)', 'Matbench\n(Band Gap)', 'Matbench\n(Glass Formation)']
keys = ['MPEA Dataset', 'matbench_expt_gap', 'matbench_glass']

# Fix naming mapping for DNN
hybrid_scores = [data[k].get('Hybrid Model (Ours)', data[k].get('Hybrid')) for k in keys]
dnn_scores = [data[k].get('DNN', data[k].get('DNN (Deep Learning)')) for k in keys]
rf_scores = [data[k].get('RF', data[k].get('Random Forest')) for k in keys]
xgb_scores = [data[k]['XGBoost'] for k in keys]

x = np.arange(len(datasets))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

# Highlight the Hybrid model
rects1 = ax.bar(x - 1.5*width, hybrid_scores, width, label='Hybrid Model (Ours)', color='#d62728', edgecolor='black', linewidth=1.5)
rects2 = ax.bar(x - 0.5*width, rf_scores, width, label='Random Forest', color='#2ca02c')
rects3 = ax.bar(x + 0.5*width, xgb_scores, width, label='XGBoost', color='#ff7f0e')
rects4 = ax.bar(x + 1.5*width, dnn_scores, width, label='Deep Learning (DNN)', color='#1f77b4')

ax.set_ylabel('R² Score (Higher is Better)')
ax.set_title('Superiority of Hybrid Ensemble Across Diverse Materials Datasets', fontsize=15, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)

# Add values above bars
for c in ax.containers:
    ax.bar_label(c, fmt='%.3f', padding=3, fontsize=10)

ax.axhline(0, color='black', linewidth=0.8)
ax.grid(axis='y', linestyle='--', alpha=0.6)

fig.tight_layout()
plt.savefig('multi_dataset_chart.png', dpi=300)
print("Saved multi_dataset_chart.png")
