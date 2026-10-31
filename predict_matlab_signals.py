from pathlib import Path
import numpy as np
import pandas as pd
import joblib

project = Path(__file__).resolve().parent


data = pd.read_csv(
    project / "data/raw/matlab/power_quality_signals.csv"
)


model = joblib.load(
    project / "models/power_quality_rf.joblib"
)

time = data["time_s"].to_numpy(dtype=float)
fs = 1 / np.mean(np.diff(time))

signals = {
    "Normal": "normal_voltage_v",
    "Sag": "sag_voltage_v",
    "Swell": "swell_voltage_v",
    "Interruption": "interruption_voltage_v",
    "Transient": "transient_voltage_v"
}

results = []

for expected, column in signals.items():

    voltage = data[column].to_numpy(dtype=float)

    rms = np.sqrt(np.mean(voltage ** 2))
    peak = np.max(np.abs(voltage))

    fft_values = np.fft.rfft(voltage)
    frequencies = np.fft.rfftfreq(
        len(voltage), d=1 / fs
    )

    high_freq_mask = (
        (frequencies >= 700) &
        (frequencies <= 1500)
    )

    high_freq_energy = np.sum(
        (np.abs(fft_values[high_freq_mask])
         / len(voltage)) ** 2
    )

    features = pd.DataFrame([{
        "rms_voltage_v": rms,
        "peak_voltage_v": peak,
        "crest_factor": peak / rms,
        "mean_voltage_v": np.mean(voltage),
        "std_voltage_v": np.std(voltage),
        "peak_to_peak_v": np.ptp(voltage),
        "high_frequency_energy": high_freq_energy
    }])

    prediction = model.predict(features)[0]

    results.append({
        "expected": expected,
        "predicted": prediction,
        "correct": prediction == expected
    })

results_df = pd.DataFrame(results)

print("\n--- EngiSense MATLAB + ML Integration ---")
print(results_df.to_string(index=False))

accuracy = results_df["correct"].mean() * 100

print(f"\nMATLAB signal accuracy: {accuracy:.1f}%")


output_file = (
    project / "reports/matlab_ml_predictions.csv"
)

if output_file.exists():
    print("Results file already exists; not overwriting.")
else:
    results_df.to_csv(output_file, index=False)
    print("Results saved to:", output_file)
