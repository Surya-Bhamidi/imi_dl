# pyrefly: ignore [missing-import]
from matbench.bench import MatbenchBenchmark
import pandas as pd

mb = MatbenchBenchmark(autoload=False, subset=["matbench_dielectric"])
for task in mb.tasks:
    task.load()
    df = task.df
    df.rename(columns={'composition': 'FORMULA', 'n': 'PROPERTY: Refractive Index'}, inplace=True)
    df.to_csv("matbench_dielectric_dataset.csv", index=False)
    print("Downloaded matbench_dielectric_dataset.csv")
