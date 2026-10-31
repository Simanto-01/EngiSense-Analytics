from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

project = Path(__file__).resolve().parent

data = pd.read_csv(
    project / "data/raw/matlab/power_quality_signals.csv"
)

time = data["time_s"].to_numpy(dtype=float)
fs = 1 / np.mean(np.diff(time))

normal = data["normal_voltage_v"].to_numpy(dtype=float)
transient = data["transient_voltage_v"].to_numpy(dtype=float)

rng = np.random.default_rng(42)
noisy_normal = normal + rng.normal(
    0, 325 * 0.05, len(normal)
)


def high_frequency_energy_over_time(voltage):
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

    return times, energy


fig, ax = plt.subplots(figsize=(11, 6))

for label, signal in [
    ("Clean Normal", normal),
    ("Noisy Normal (5%)", noisy_normal),
    ("Electrical Transient", transient)
]:
    times, energy = high_frequency_energy_over_time(signal)

    ax.plot(times, energy, label=label, linewidth=2)

ax.axvline(
    0.04,
    color="red",
    linestyle="--",
    label="Transient start (0.04 s)"
)

ax.set_title(
    "EngiSense - High-Frequency Energy Localization"
)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("High-Frequency STFT Energy")
ax.set_xlim(0, 0.1)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_file = (
    project / "reports/transient_energy_localization.png"
)

if output_file.exists():
    print("Plot already exists; not overwriting.")
else:
    fig.savefig(output_file, dpi=300)
    print("Saved to:", output_file)

plt.show()



print("\n--- Transient Energy Concentration ---")

for label, signal in [
    ("Clean Normal", normal),
    ("Noisy Normal (5%)", noisy_normal),
    ("Electrical Transient", transient)
]:

    times, energy = high_frequency_energy_over_time(signal)

    mean_energy = np.mean(energy)
    max_energy = np.max(energy)

    concentration = max_energy / (mean_energy + 1e-12)

    peak_time = times[np.argmax(energy)]

    print(f"\n{label}")
    print(f"Maximum energy: {max_energy:.3f}")
    print(f"Mean energy: {mean_energy:.3f}")
    print(f"Concentration ratio: {concentration:.3f}")
    print(f"Peak time: {peak_time:.4f} s")



print("\n--- Noise vs Transient Concentration ---")

rng = np.random.default_rng(123)

for noise_percent in [1, 3, 5, 10, 15, 20]:

    ratios = []

    for _ in range(50):

        noisy = normal + rng.normal(
            0,
            325 * noise_percent / 100,
            len(normal)
        )

        _, energy = high_frequency_energy_over_time(noisy)

        ratio = np.max(energy) / (
            np.mean(energy) + 1e-12
        )

        ratios.append(ratio)

    print(
        f"Noise {noise_percent:2d}%: "
        f"Mean ratio = {np.mean(ratios):.2f}, "
        f"Max ratio = {np.max(ratios):.2f}"
    )

_, transient_energy = high_frequency_energy_over_time(
    transient
)

transient_ratio = (
    np.max(transient_energy)
    / (np.mean(transient_energy) + 1e-12)
)

print(f"\nGenuine transient ratio: {transient_ratio:.2f}")
