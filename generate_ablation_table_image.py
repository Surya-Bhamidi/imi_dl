import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = r"C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\Presentation_Plots"
os.makedirs(out_dir, exist_ok=True)

columns = ["Model Configuration", "Hardness\n(HV)", "Yield Strength\n(MPa)", "UTS\n(MPa)", "Elongation\n(%)", "Young's Mod.\n(GPa)"]
rows = [
    ["DNN (PINN)", "0.806", "0.695", "0.754", "0.696", "0.913"],
    ["Extra Trees", "0.857", "0.808", "0.707", "0.577", "0.993"],
    ["ADMM-Lasso", "0.866", "0.843", "0.775", "0.623", "0.988"],
    ["Hybrid PINN (Ours)", "0.855", "0.816", "0.825", "0.731", "0.975"]
]

# Highlight matrix: bold the max values in columns 1 to 5
max_indices = [2, 2, 3, 3, 1]  # row indices for max values in cols 1..5

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis('off')
ax.axis('tight')

# Create table
table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)

# Apply custom premium styling
for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor('#bdc3c7')
    cell.set_linewidth(1.0)
    
    # Header row
    if row_idx == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', weight='bold', size=12)
    else:
        # Configuration Name column
        if col_idx == 0:
            cell.set_text_props(weight='bold')
            cell.get_text().set_horizontalalignment('left')
            # Adjust padding for left alignment
            cell.PAD = 0.05
            
        # Highlight Hybrid row background
        if row_idx == 4:  # 1-indexed for rows in table object
            cell.set_facecolor('#fadbd8')  # soft premium pink/red
            if col_idx == 0:
                cell.set_text_props(color='#c0392b', weight='bold')
        else:
            cell.set_facecolor('#f8f9f9' if row_idx % 2 == 1 else 'white')
            
        # Bold maximum values per property column
        if col_idx > 0 and row_idx - 1 == max_indices[col_idx - 1]:
            cell.set_text_props(weight='bold')
            # If it's the hybrid row, keep color dark red for extra visual pop
            if row_idx == 4:
                cell.set_text_props(color='#c0392b', weight='bold')
            else:
                cell.set_text_props(color='#16a085', weight='bold')

plt.title("Ablation Study: Per-Property $R^2$ Scores", fontsize=16, fontweight='bold', pad=10, color='#2c3e50')
plt.tight_layout()

save_path = os.path.join(out_dir, "ablation_table_view.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated beautiful table image at:\n{save_path}")
