from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, sosfiltfilt

project = Path(__file__).resolve().parent

data = pd.read_csv(
    project / "data/raw/matlab/power_quality_signals.csv"
)

model = joblib.load(
    project / "models/power_quality_rf.joblib"
)

time = data["time_s"].to_numpy(dtype=float)
dt = np.mean(np.diff(time))
fs = 1 / dt

normal = data["normal_voltage_v"].to_numpy(dtype=float)

rng = np.random.default_rng(42)

noisy = normal + rng.normal(
    0, 325 * 0.05, len(normal)
)


sos = butter(
    4,
    1500,
    btype="lowpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, noisy)


def extract_features(voltage):
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

    return pd.DataFrame([{
        "rms_voltage_v": rms,
        "peak_voltage_v": peak,
        "crest_factor": peak / rms,
        "mean_voltage_v": np.mean(voltage),
        "std_voltage_v": np.std(voltage),
        "peak_to_peak_v": np.ptp(voltage),
        "high_frequency_energy": high_freq_energy
    }])


for name, signal in [
    ("Clean", normal),
    ("Noisy", noisy),
    ("Filtered", filtered)
]:
    features = extract_features(signal)
    prediction = model.predict(features)[0]

    print(
        f"{name:10s} -> "
        f"Prediction: {prediction:15s} "
        f"HF energy: "
        f"{features['high_frequency_energy'].iloc[0]:.3f}"
    )
