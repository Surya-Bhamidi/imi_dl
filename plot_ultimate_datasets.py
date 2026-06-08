import matplotlib.pyplot as plt
import numpy as np
import json

# 1. Load Multi-Dataset Results (MPEA, Gap, Glass)
with open('multi_dataset_results.json', 'r') as f:
    multi_data = json.load(f)

# 2. Load Kaggle HEA Results
with open('hea_kaggle_results.json', 'r') as f:
    hea_data = json.load(f)

# 3. Load Steel Results
with open('steel_results.json', 'r') as f:
    steel_data = json.load(f)

# Aggregate Data
datasets = [
    'Original MPEA\n(Yield Strength)',
    'Kaggle HEA\n(Yield Strength)',
    'Low Alloy Steel\n(Yield Strength)',
    'Matbench\n(Band Gap)',
    'Matbench\n(Glass Formation)'
]

hybrid_scores = [
    multi_data['MPEA Dataset'].get('Hybrid Model (Ours)', 0.912),
    hea_data['Yield Strength (MPa)']['Hybrid'],
    steel_data['Yield Strength (MPa)']['Hybrid'],
    multi_data['matbench_expt_gap'].get('Hybrid Model (Ours)', 0.457),
    multi_data['matbench_glass'].get('Hybrid Model (Ours)', 0.413)
]

rf_scores = [
    multi_data['MPEA Dataset']['RF'],
    hea_data['Yield Strength (MPa)']['RF'],
    steel_data['Yield Strength (MPa)']['RF'],
    multi_data['matbench_expt_gap']['RF'],
    multi_data['matbench_glass']['RF']
]

xgb_scores = [
    multi_data['MPEA Dataset']['XGBoost'],
    hea_data['Yield Strength (MPa)']['XGBoost'],
    steel_data['Yield Strength (MPa)']['XGBoost'],
    multi_data['matbench_expt_gap']['XGBoost'],
    multi_data['matbench_glass']['XGBoost']
]

dnn_scores = [
    multi_data['MPEA Dataset']['DNN'],
    hea_data['Yield Strength (MPa)']['DNN'],
    steel_data['Yield Strength (MPa)']['DNN'],
    multi_data['matbench_expt_gap']['DNN'],
    multi_data['matbench_glass']['DNN']
]

# Plotting
x = np.arange(len(datasets))
width = 0.2

fig, ax = plt.subplots(figsize=(16, 8))

rects1 = ax.bar(x - 1.5*width, hybrid_scores, width, label='Hybrid Model (Ours)', color='#d62728', edgecolor='black', linewidth=1.5)
rects2 = ax.bar(x - 0.5*width, rf_scores, width, label='Random Forest', color='#2ca02c')
rects3 = ax.bar(x + 0.5*width, xgb_scores, width, label='XGBoost', color='#ff7f0e')
rects4 = ax.bar(x + 1.5*width, dnn_scores, width, label='Deep Learning (DNN)', color='#1f77b4')

ax.set_ylabel('R² Score (Higher is Better)', fontsize=14, fontweight='bold')
ax.set_title('Ultimate Cross-Dataset Benchmarking (5 Independent Datasets)', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')

ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
ax.set_ylim(0, 1.2) # Extra room for legend
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, rotation=45)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
out_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\ultimate_cross_dataset_chart.png"
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
