from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent

csv_file = (
    project / "data" / "processed"
    / "power_quality_training.csv"
)

data = pd.read_csv(csv_file)

print("EngiSense ML Dataset Verification")
print("--------------------------------")
print("Dataset shape:", data.shape)

print("\nSignal classes:")
print(data["signal_type"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())

print("\nFeature columns:")
print(data.columns.tolist())
