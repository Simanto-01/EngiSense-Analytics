from pathlib import Path
import numpy as np
import pandas as pd


project_folder = Path(__file__).resolve().parent
output_folder = project_folder / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

fs = 5000
duration = 0.1
t = np.arange(500) / fs
frequency = 50
amplitude = 325

records = []
signal_types = [
    "Normal", "Sag", "Swell",
    "Interruption", "Transient"
]

for signal_type in signal_types:
    for sample_id in range(200):


        amp = amplitude * rng.uniform(0.97, 1.03)
        phase = rng.uniform(0, 2 * np.pi)

        voltage = amp * np.sin(
            2 * np.pi * frequency * t + phase
        )

        start = rng.uniform(0.025, 0.045)
        end = start + rng.uniform(0.025, 0.045)
        mask = (t >= start) & (t < end)

        if signal_type == "Sag":
            voltage[mask] *= rng.uniform(0.3, 0.8)

        elif signal_type == "Swell":
            voltage[mask] *= rng.uniform(1.2, 1.7)

        elif signal_type == "Interruption":
            voltage[mask] *= rng.uniform(0, 0.05)

        elif signal_type == "Transient":
            relative_time = t - start
            transient_mask = t >= start

            voltage[transient_mask] += (
                rng.uniform(80, 200)
                * np.exp(
                    -rng.uniform(150, 300)
                    * relative_time[transient_mask]
                )
                * np.sin(
                    2 * np.pi
                    * rng.uniform(800, 1200)
                    * relative_time[transient_mask]
                )
            )


        voltage += rng.normal(0, 1, len(t))


        rms = np.sqrt(np.mean(voltage ** 2))
        peak = np.max(np.abs(voltage))


        fft_values = np.fft.rfft(voltage)
        frequencies = np.fft.rfftfreq(
            len(voltage), d=1 / fs
        )

        high_freq_mask = (
            (frequencies >= 700)
            & (frequencies <= 1500)
        )

        high_freq_energy = np.sum(
            (np.abs(fft_values[high_freq_mask])
             / len(voltage)) ** 2
        )

        records.append({
            "signal_type": signal_type,
            "rms_voltage_v": rms,
            "peak_voltage_v": peak,
            "crest_factor": peak / rms,
            "mean_voltage_v": np.mean(voltage),
            "std_voltage_v": np.std(voltage),
            "peak_to_peak_v": np.ptp(voltage),
            "high_frequency_energy": high_freq_energy
        })

dataset = pd.DataFrame(records)

output_file = (
    output_folder / "power_quality_training.csv"
)

dataset.to_csv(output_file, index=False)

print("\n--- EngiSense Training Dataset ---")
print("Total samples:", len(dataset))
print("\nSamples per class:")
print(dataset["signal_type"].value_counts())
print("\nSaved to:", output_file)
