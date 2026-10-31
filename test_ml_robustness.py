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
dt = np.mean(np.diff(time))

signals = {
    "Normal": "normal_voltage_v",
    "Sag": "sag_voltage_v",
    "Swell": "swell_voltage_v",
    "Interruption": "interruption_voltage_v",
    "Transient": "transient_voltage_v"
}

rng = np.random.default_rng(42)


noise_levels = [0, 1, 3, 5, 10, 15, 20]
repetitions = 20

results = []

for noise_pct in noise_levels:
    correct = 0
    total = 0

    for expected, column in signals.items():
        original = data[column].to_numpy(dtype=float)

        for _ in range(repetitions):
            noise_std = 325 * noise_pct / 100

            voltage = original + rng.normal(
                0, noise_std, len(original)
            )

            rms = np.sqrt(np.mean(voltage ** 2))
            peak = np.max(np.abs(voltage))

            fft_values = np.fft.rfft(voltage)
            frequencies = np.fft.rfftfreq(
                len(voltage), d=dt
            )

            mask = (
                (frequencies >= 700)
                & (frequencies <= 1500)
            )

            high_freq_energy = np.sum(
                (np.abs(fft_values[mask])
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

            predicted = model.predict(features)[0]

            correct += int(predicted == expected)
            total += 1

    accuracy = 100 * correct / total

    results.append({
        "noise_percent": noise_pct,
        "accuracy_percent": accuracy,
        "correct": correct,
        "total": total
    })

results_df = pd.DataFrame(results)

print("\n--- EngiSense ML Robustness Test ---")
print(results_df.to_string(index=False))

output_file = (
    project / "reports/ml_noise_robustness.csv"
)

if output_file.exists():
    print("\nReport already exists; not overwriting.")
else:
    results_df.to_csv(output_file, index=False)
    print("\nSaved to:", output_file)
