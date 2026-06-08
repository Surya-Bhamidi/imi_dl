import matplotlib.pyplot as plt
import numpy as np
import json

with open('multi_dataset_results.json', 'r') as f:
    multi_data = json.load(f)

with open('hea_kaggle_results.json', 'r') as f:
    hea_data = json.load(f)

with open('superconductor_results.json', 'r') as f:
    super_data = json.load(f)

datasets = [
    'Original MPEA\n(Yield Strength)',
    'Kaggle HEA\n(Yield Strength)',
    'Superconductors\n(Critical Temp)',
    'Matbench\n(Band Gap)',
    'Matbench\n(Glass Formation)'
]

hybrid_scores = [
    multi_data['MPEA Dataset'].get('Hybrid Model (Ours)', 0.912),
    hea_data['Yield Strength (MPa)']['Hybrid'],
    super_data['PROPERTY: Critical Temperature (K)']['Hybrid'],
    multi_data['matbench_expt_gap'].get('Hybrid Model (Ours)', 0.457),
    multi_data['matbench_glass'].get('Hybrid Model (Ours)', 0.413)
]

rf_scores = [
    multi_data['MPEA Dataset']['RF'],
    hea_data['Yield Strength (MPa)']['RF'],
    super_data['PROPERTY: Critical Temperature (K)']['RF'],
    multi_data['matbench_expt_gap']['RF'],
    multi_data['matbench_glass']['RF']
]

xgb_scores = [
    multi_data['MPEA Dataset']['XGBoost'],
    hea_data['Yield Strength (MPa)']['XGBoost'],
    super_data['PROPERTY: Critical Temperature (K)']['XGBoost'],
    multi_data['matbench_expt_gap']['XGBoost'],
    multi_data['matbench_glass']['XGBoost']
]

dnn_scores = [
    multi_data['MPEA Dataset']['DNN'],
    hea_data['Yield Strength (MPa)']['DNN'],
    super_data['PROPERTY: Critical Temperature (K)']['DNN'],
    multi_data['matbench_expt_gap']['DNN'],
    multi_data['matbench_glass']['DNN']
]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 8))

rects1 = ax.bar(x - width, hybrid_scores, width, label='Hybrid Model (Ours)', color='#d62728', edgecolor='black', linewidth=1.5)
rects3 = ax.bar(x, xgb_scores, width, label='XGBoost', color='#ff7f0e')
rects4 = ax.bar(x + width, dnn_scores, width, label='Deep Learning (DNN)', color='#1f77b4')

ax.set_ylabel('R² Score (Higher is Better)', fontsize=14, fontweight='bold')
ax.set_title('Ultimate Cross-Dataset Benchmarking (5 Independent Datasets)', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')

ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
ax.set_ylim(0, 1.2) 
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, rotation=45)

autolabel(rects1)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
out_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\ultimate_cross_dataset_chart_superconductors.png"
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
