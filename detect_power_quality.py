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

time = data["time_s"].to_numpy(dtype=float)
dt = np.mean(np.diff(time))
sampling_rate = 1 / dt


normal_rms = 325 / np.sqrt(2)

signals = [
    ("normal_voltage_v", "Normal"),
    ("sag_voltage_v", "Sag"),
    ("swell_voltage_v", "Swell"),
    ("interruption_voltage_v", "Interruption"),
    ("transient_voltage_v", "Transient"),
]


def detect_disturbance(values):


    window_size = round(sampling_rate / 50)

    windows = [
        values[i:i + window_size]
        for i in range(0, len(values), window_size)
        if len(values[i:i + window_size]) == window_size
    ]

    window_rms = np.array([
        np.sqrt(np.mean(window ** 2))
        for window in windows
    ])


    if np.any(window_rms < 0.1 * normal_rms):
        return "Interruption"

    if np.any(window_rms < 0.9 * normal_rms):
        return "Sag"

    if np.any(window_rms > 1.1 * normal_rms):
        return "Swell"


    N = len(values)

    frequencies = np.fft.rfftfreq(N, d=dt)
    spectrum = np.abs(np.fft.rfft(values)) / N

    high_frequency = (
        (frequencies >= 700) &
        (frequencies <= 1500)
    )

    high_frequency_energy = np.sum(
        spectrum[high_frequency] ** 2
    )

    if high_frequency_energy > 1:
        return "Transient"

    return "Normal"


print("\n--- EngiSense Automatic Disturbance Detection ---")

correct = 0

for column, expected in signals:

    values = data[column].to_numpy(dtype=float)

    predicted = detect_disturbance(values)

    match = predicted == expected

    if match:
        correct += 1

    print(
        f"{expected:15s} -> "
        f"Predicted: {predicted:15s} "
        f"{'PASS' if match else 'CHECK'}"
    )

accuracy = correct / len(signals) * 100

print(f"\nDetection accuracy on 5 test signals: {accuracy:.1f}%")
