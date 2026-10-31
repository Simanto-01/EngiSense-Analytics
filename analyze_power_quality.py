from pathlib import Path
import numpy as np
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


signals = [
    "normal_voltage_v",
    "sag_voltage_v",
    "swell_voltage_v",
    "interruption_voltage_v",
    "transient_voltage_v",
]

print("\n--- EngiSense Power Quality RMS Analysis ---")


for signal in signals:
    values = data[signal].to_numpy(dtype=float)

    rms = np.sqrt(np.mean(values ** 2))

    print(f"{signal}: {rms:.2f} V")



print("\n--- EngiSense Peak and Crest Factor Analysis ---")

for signal in signals:
    values = data[signal].to_numpy(dtype=float)

    rms = np.sqrt(np.mean(values ** 2))
    peak_voltage = np.max(np.abs(values))
    crest_factor = peak_voltage / rms

    print(f"\nSignal: {signal}")
    print(f"Peak Voltage: {peak_voltage:.2f} V")
    print(f"Crest Factor: {crest_factor:.3f}")



print("\n--- EngiSense Statistical Features ---")

for signal in signals:
    values = data[signal].to_numpy(dtype=float)

    mean_voltage = np.mean(values)
    std_voltage = np.std(values)
    peak_to_peak = np.ptp(values)

    print(f"\nSignal: {signal}")
    print(f"Mean Voltage: {mean_voltage:.2f} V")
    print(f"Standard Deviation: {std_voltage:.2f} V")
    print(f"Peak-to-Peak Voltage: {peak_to_peak:.2f} V")


feature_rows = []

for signal in signals:
    values = data[signal].to_numpy(dtype=float)

    rms = np.sqrt(np.mean(values ** 2))
    peak = np.max(np.abs(values))

    feature_rows.append({
        "signal_type": signal,
        "rms_voltage_v": rms,
        "peak_voltage_v": peak,
        "crest_factor": peak / rms,
        "mean_voltage_v": np.mean(values),
        "std_voltage_v": np.std(values),
        "peak_to_peak_v": np.ptp(values),
    })


features = pd.DataFrame(feature_rows)


output_folder = project_folder / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "power_quality_features.csv"
features.to_csv(output_file, index=False)

print("\n--- EngiSense Feature Dataset ---")
print(features.round(3).to_string(index=False))
print("\nFeatures saved successfully!")
print("Saved to:", output_file)
