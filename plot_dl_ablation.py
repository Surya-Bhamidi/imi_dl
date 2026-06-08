import matplotlib.pyplot as plt
import numpy as np
import json

with open('_hybrid_results.json', 'r') as f:
    data = json.load(f)

properties = ['Hardness (HV)', 'Yield Strength (MPa)', 'UTS (MPa)', 'Elongation (%)', 'Young Modulus (GPa)']
labels = ['Hardness\n(HV)', 'Yield Strength\n(MPa)', 'UTS\n(MPa)', 'Elongation\n(%)', 'Young Modulus\n(GPa)']

hybrid = [data['hybrid'][p] for p in properties]
dnn = [data['dl_standalone'][p] for p in properties]
rnn = [data['rnn'][p] for p in properties]
lstm = [data['lstm'][p] for p in properties]
gru = [data['gru'][p] for p in properties]

x = np.arange(len(properties))
width = 0.15

fig, ax = plt.subplots(figsize=(15, 8))

# Removed RF/XGBoost, focusing purely on Deep Learning
rects1 = ax.bar(x - 2*width, hybrid, width, label='Hybrid Model (Ours)', color='#d62728', edgecolor='black', linewidth=1.5)
rects2 = ax.bar(x - width, dnn, width, label='Deep Learning (DNN)', color='#1f77b4')
rects3 = ax.bar(x, rnn, width, label='RNN Baseline', color='#9467bd')
rects4 = ax.bar(x + width, lstm, width, label='LSTM Baseline', color='#8c564b')
rects5 = ax.bar(x + 2*width, gru, width, label='GRU Baseline', color='#e377c2')

ax.set_ylabel('R² Score (Higher is Better)', fontsize=14, fontweight='bold')
ax.set_title('Deep Learning Architecture Ablation Study', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5)
ax.set_ylim(0, 1.25) 
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()
out_path = r"C:\Users\bhara\.gemini\antigravity\brain\1595fb12-0302-4ada-a005-41162795f2e3\artifacts\dl_ablation_chart.png"
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
