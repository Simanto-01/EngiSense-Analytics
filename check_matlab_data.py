from pathlib import Path
import pandas as pd


project_folder = Path(__file__).resolve().parent

csv_file = (
    project_folder
    / "data"
    / "raw"
    / "matlab"
    / "power_quality_signals.csv"
)


data = pd.read_csv(csv_file)


print("\n--- EngiSense MATLAB Data Verification ---")
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

