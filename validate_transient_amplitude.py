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


start_time = 0.04
transient_frequency = 1000
decay_rate = 200

relative_time = time - start_time
event_mask = time >= start_time


def create_transient(amplitude):
    signal = normal.copy()

    signal[event_mask] += (
        amplitude
        * np.exp(
            -decay_rate * relative_time[event_mask]
        )
        * np.sin(
            2 * np.pi
            * transient_frequency
            * relative_time[event_mask]
        )
    )

    return signal


def detect_transient(voltage):

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

    peak_energy = np.max(energy)

    concentration = peak_energy / (
        np.mean(energy) + 1e-12
    )

    detected = (
        peak_energy > 300
        and concentration > 5
    )

    return detected, peak_energy, concentration


print("\n--- EngiSense Transient Amplitude Test ---")

for amplitude in [20, 40, 60, 80, 100, 150, 200]:

    signal = create_transient(amplitude)

    detected, energy, ratio = detect_transient(signal)

    print(
        f"Amplitude: {amplitude:3d} V | "
        f"Detected: {str(detected):5s} | "
        f"Peak energy: {energy:8.1f} | "
        f"Concentration: {ratio:.2f}"
    )
