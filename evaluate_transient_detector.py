from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft

project = Path(__file__).resolve().parent

data = pd.read_csv(
    project / "data/raw/matlab/power_quality_signals.csv"
)

time = data["time_s"].to_numpy(dtype=float)
fs = 1 / np.mean(np.diff(time))

normal = data["normal_voltage_v"].to_numpy(dtype=float)
transient = data["transient_voltage_v"].to_numpy(dtype=float)

rng = np.random.default_rng(123)


def transient_features(voltage):
    frequencies, times, Z = stft(
        voltage,
        fs=fs,
        window="hann",
        nperseg=100,
        noverlap=90
    )

    band = (
        (frequencies >= 700)
        & (frequencies <= 1500)
    )

    energy = np.sum(
        np.abs(Z[band, :]) ** 2,
        axis=0
    )

    max_energy = np.max(energy)
    mean_energy = np.mean(energy)

    concentration = max_energy / (
        mean_energy + 1e-12
    )

    return max_energy, concentration



energy_threshold = 100
concentration_threshold = 5


def detect_transient(voltage):
    max_energy, concentration = transient_features(
        voltage
    )

    detected = (
        max_energy > energy_threshold
        and concentration > concentration_threshold
    )

    return detected, max_energy, concentration


print("\n--- EngiSense Transient Detection ---")

for label, signal in [
    ("Clean Normal", normal),
    ("Genuine Transient", transient)
]:
    detected, energy, ratio = detect_transient(signal)

    print(
        f"{label:20s} | "
        f"Detected: {detected} | "
        f"Peak energy: {energy:.1f} | "
        f"Ratio: {ratio:.2f}"
    )

print("\n--- Noisy Normal Tests ---")

for noise_pct in [1, 3, 5, 10, 15, 20]:

    false_alarms = 0

    for _ in range(50):
        noisy = normal + rng.normal(
            0,
            325 * noise_pct / 100,
            len(normal)
        )

        detected, _, _ = detect_transient(noisy)

        false_alarms += int(detected)

    print(
        f"Noise {noise_pct:2d}% | "
        f"False alarms: {false_alarms}/50"
    )
