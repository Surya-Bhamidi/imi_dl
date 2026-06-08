import pandas as pd
from matminer.datasets import load_dataset
import os

print("Downloading Matbench Steels dataset...")
# Matminer directly downloads the dataframe
df = load_dataset('matbench_steels')

print("Formatting for pipeline...")
# Rename columns to exactly match what data_preprocessing.py expects
df = df.rename(columns={
    'composition': 'FORMULA',
    'yield strength': 'PROPERTY: YS (MPa)'
})

output_path = 'matbench_steels_dataset.csv'
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset successfully downloaded and saved to {output_path}")
print(f"Total samples: {len(df)}")
print("You can now run 'python train_hybrid_admm_dl.py' by temporarily pointing it to this new CSV in the code!")
