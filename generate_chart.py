import json
import matplotlib.pyplot as plt
import numpy as np

with open('_hybrid_results.json', 'r') as f:
    data = json.load(f)

properties = ["Hardness (HV)", "Yield Strength (MPa)", "UTS (MPa)", "Elongation (%)", "Young Modulus (GPa)", "OVERALL_AVG_R2"]

models = ["hybrid", "dl_standalone", "rf_standalone", "admm_standalone", "xgboost", "svr", "rnn", "lstm", "gru", "bilstm", "cnn1d"]
model_labels = ["Hybrid Model (Ours)", "DNN (Deep Learning)", "Random Forest", "ADMM", "XGBoost", "SVR", "RNN", "LSTM", "GRU", "Bi-LSTM", "1D CNN"]

x = np.arange(len(properties))
width = 0.075

fig, ax = plt.subplots(figsize=(22, 8))

for i, model in enumerate(models):
    scores = [data.get(model, {}).get(prop, 0) for prop in properties]
    # Center the bars
    offset = width * i - (width * len(models) / 2) + width/2
    ax.bar(x + offset, scores, width, label=model_labels[i])

ax.set_ylabel('R² Score')
ax.set_title('Comprehensive Model Performance Comparison (R² Score)')
ax.set_xticks(x)
ax.set_xticklabels(properties, rotation=0, fontsize=10, fontweight='bold')
ax.legend(loc='lower left', ncol=11, fontsize=9, bbox_to_anchor=(0.0, -0.15))
ax.set_ylim(0, 1.1)

# Add value labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.2f', padding=3, fontsize=7, rotation=90)

fig.tight_layout()
# Give enough space at the bottom for the legend
plt.subplots_adjust(bottom=0.2)
plt.savefig('comparison_chart.png', dpi=300, bbox_inches='tight')
print("Chart saved as comparison_chart.png")
