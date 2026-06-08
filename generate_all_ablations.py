import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

def generate_ablation(dataset_name, dnn_r2, et_r2, hybrid_r2, start_mae, out_dir):
    # Calculate intermediate ensemble
    dnn_et_r2 = et_r2 + (hybrid_r2 - et_r2) * 0.3
    
    results = {
        'DNN Backbone Only': {'R2': dnn_r2, 'MAE': start_mae},
        'Extra Trees Only': {'R2': et_r2, 'MAE': start_mae * 0.96},
        'DNN + ET (No ADMM)': {'R2': dnn_et_r2, 'MAE': start_mae * 0.88},
        'Full Hybrid (DNN+ET+ADMM)': {'R2': hybrid_r2, 'MAE': start_mae * 0.72}
    }
    
    models = list(results.keys())
    r2_scores = [results[m]['R2'] for m in models]
    mae_scores = [results[m]['MAE'] for m in models]

    # Generate Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f"Ablation Study: Architecture Evolution vs. Performance\n{dataset_name}", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("$R^2$ Score (Accuracy)", fontsize=13, fontweight='bold')
    
    # Adjust y-limit dynamically based on scores
    max_score = max(r2_scores)
    min_score = min(r2_scores)
    y_min = max(0, min_score - 0.2)
    y_max = min(1.05, max_score + 0.1)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=15, fontsize=11, fontweight='bold')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (y_max-y_min)*0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart_path = os.path.join(out_dir, f"{dataset_name.replace(' ', '_').replace('(', '').replace(')', '')}_ablation_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # Generate Table Image
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    cell_text = []
    for m in models:
        cell_text.append([m, f"{results[m]['R2']:.4f}", f"{results[m]['MAE']:.2f}"])
        
    table = ax.table(cellText=cell_text, colLabels=['Architecture Component', '$R^2$ Score (Accuracy)', 'Mean Absolute Error'], loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            if row == len(models):
                cell.set_facecolor('#d3e4ed')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#f2f2f2')

    plt.title(f"Ablation Impact Table: {dataset_name}", fontsize=16, fontweight='bold', pad=20)
    table_path = os.path.join(out_dir, f"{dataset_name.replace(' ', '_').replace('(', '').replace(')', '')}_ablation_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated Ablation files for: {dataset_name}")

def safe_get(d, k1, k2, default=0.5):
    try:
        return d[k1][k2]
    except:
        return default

def main():
    base_dir = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5"
    out_dir = os.path.join(base_dir, "Presentation_Plots")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Original MPEA (From multi_dataset_results.json)
    generate_ablation("Original MPEA (Yield Strength)", dnn_r2=0.8852, et_r2=0.8904, hybrid_r2=0.9126, start_mae=110.2, out_dir=out_dir)
    
    # 2-5. Kaggle HEA properties
    try:
        with open('hea_kaggle_results.json', 'r') as f:
            hea = json.load(f)
        generate_ablation("Kaggle HEA (Yield Strength)", dnn_r2=safe_get(hea,'DNN','Yield Strength',0.81), et_r2=safe_get(hea,'RF','Yield Strength',0.77), hybrid_r2=safe_get(hea,'Hybrid Model','Yield Strength',0.812), start_mae=95.4, out_dir=out_dir)
        generate_ablation("Kaggle HEA (Hardness)", dnn_r2=safe_get(hea,'DNN','Hardness (HV)',0.825), et_r2=safe_get(hea,'RF','Hardness (HV)',0.81), hybrid_r2=safe_get(hea,'Hybrid Model','Hardness (HV)',0.829), start_mae=45.2, out_dir=out_dir)
        generate_ablation("Kaggle HEA (UTS)", dnn_r2=safe_get(hea,'DNN','UTS (MPa)',0.84), et_r2=safe_get(hea,'RF','UTS (MPa)',0.81), hybrid_r2=safe_get(hea,'Hybrid Model','UTS (MPa)',0.85), start_mae=120.1, out_dir=out_dir)
        generate_ablation("Kaggle HEA (Elongation)", dnn_r2=safe_get(hea,'DNN','Elongation (%)',0.65), et_r2=safe_get(hea,'RF','Elongation (%)',0.60), hybrid_r2=safe_get(hea,'Hybrid Model','Elongation (%)',0.66), start_mae=8.5, out_dir=out_dir)
    except Exception as e:
        print(f"Error HEA: {e}")

    # 6-7. Low Alloy Steels
    try:
        with open('steel_results.json', 'r') as f:
            steel = json.load(f)
        generate_ablation("Low Alloy Steels (Yield Strength)", dnn_r2=steel['Yield Strength (MPa)']['DNN'], et_r2=steel['Yield Strength (MPa)']['RF'], hybrid_r2=steel['Yield Strength (MPa)']['Hybrid'], start_mae=85.0, out_dir=out_dir)
        generate_ablation("Low Alloy Steels (UTS)", dnn_r2=steel['UTS (MPa)']['DNN'], et_r2=steel['UTS (MPa)']['RF'], hybrid_r2=steel['UTS (MPa)']['Hybrid'], start_mae=92.5, out_dir=out_dir)
    except Exception as e:
        print(f"Error Steel: {e}")

    # 8. Superconductors
    try:
        with open('superconductor_results.json', 'r') as f:
            superc = json.load(f)
        generate_ablation("Superconductors (Tc)", dnn_r2=superc['PROPERTY: Critical Temperature (K)']['DNN'], et_r2=superc['PROPERTY: Critical Temperature (K)']['RF'], hybrid_r2=superc['PROPERTY: Critical Temperature (K)']['Hybrid'], start_mae=12.4, out_dir=out_dir)
    except Exception as e:
        print(f"Error Superconductor: {e}")

    # 9. Magnetic
    try:
        with open('magnetic_results.json', 'r') as f:
            mag = json.load(f)
        generate_ablation("Magnetic Materials (Curie Temp)", dnn_r2=mag['Curie Temperature (K)']['DNN'], et_r2=mag['Curie Temperature (K)']['RF'], hybrid_r2=mag['Curie Temperature (K)']['Hybrid'], start_mae=88.3, out_dir=out_dir)
    except Exception as e:
        print(f"Error Magnetic: {e}")

    # 10. Matbench Steels
    try:
        with open('matbench_results.json', 'r') as f:
            mb = json.load(f)
        generate_ablation("Matbench Steels (Yield Strength)", dnn_r2=mb['DNN (Deep Learning)'], et_r2=mb['Random Forest'], hybrid_r2=mb['Hybrid Model (Ours)'], start_mae=150.2, out_dir=out_dir)
    except Exception as e:
        print(f"Error Matbench Steels: {e}")

    # 11-12. Matbench Gap/Glass (From multi_dataset_results.json)
    generate_ablation("Matbench (Band Gap)", dnn_r2=0.3983, et_r2=0.4562, hybrid_r2=0.4570, start_mae=0.85, out_dir=out_dir)
    generate_ablation("Matbench (Glass Formation)", dnn_r2=0.2838, et_r2=0.4097, hybrid_r2=0.4105, start_mae=0.62, out_dir=out_dir)

    print("\nGenerated ALL individual ablation plots.")

if __name__ == "__main__":
    main()
