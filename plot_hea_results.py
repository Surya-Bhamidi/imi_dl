import matplotlib.pyplot as plt
import numpy as np
import json
import os

with open('hea_kaggle_results.json', 'r') as f:
    data = json.load(f)

properties = list(data.keys())
labels = ['Hardness\n(HV)', 'Yield Strength\n(MPa)', 'UTS\n(MPa)', 'Elongation\n(%)', 'Young Modulus\n(GPa)']

hybrid = [data[p]['Hybrid'] for p in properties]
rf = [data[p]['RF'] for p in properties]
xgb = [data[p]['XGBoost'] for p in properties]
dnn = [data[p]['DNN'] for p in properties]

x = np.arange(len(properties))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))

# Define colors matching our previous charts
rects1 = ax.bar(x - width, hybrid, width, label='Hybrid Model (Ours)', color='#d62728', edgecolor='black', linewidth=1.5)
rects3 = ax.bar(x, xgb, width, label='XGBoost', color='#ff7f0e')
rects4 = ax.bar(x + width, dnn, width, label='Deep Learning (DNN)', color='#1f77b4')

ax.set_ylabel('R² Score (Higher is Better)', fontsize=13, fontweight='bold')
ax.set_title('Benchmarking Architecture on the Kaggle HEA Dataset', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

# Put legend outside the plot to avoid overlapping bars
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
ax.set_ylim(0, 1.25) # Extra room for legend and labels
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=45)

autolabel(rects1)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
out_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\hea_evaluation_chart.png"
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
