import sys
from pathlib import Path

import pandas as pd



project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from engisense.statistics import calculate_statistics
from engisense.visualization import plot_signal
from engisense.validation import validate_adc, validate_timestamps, validate_voltage




file_path = project_root / "data" / "raw" / "wokwi" / "esp32_sensor_data.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of samples:", len(df))
print("Columns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())




voltage = df["voltage_v"]

print("\n--- Voltage Statistics (Pandas) ---")

print("Minimum voltage:", round(voltage.min(), 3), "V")
print("Maximum voltage:", round(voltage.max(), 3), "V")
print("Mean voltage:", round(voltage.mean(), 3), "V")
print("Standard deviation:", round(voltage.std(), 3), "V")




results = calculate_statistics(voltage)

print("\n--- EngiSense Statistics Module ---")

print("Minimum voltage:", round(results["minimum"], 3), "V")
print("Maximum voltage:", round(results["maximum"], 3), "V")
print("Mean voltage:", round(results["mean"], 3), "V")
print("Standard deviation:", round(results["standard_deviation"], 3), "V")




print("\n--- EngiSense Data Validation ---")

adc_valid = validate_adc(df["adc_raw"])

print("ADC readings valid:", adc_valid)
print("Total ADC samples checked:", len(df["adc_raw"]))


timestamp_results = validate_timestamps(df["time_ms"])

print("\n--- Timestamp Validation ---")
print("Timestamps valid:", timestamp_results["timestamps_valid"])
print("Mean interval:", timestamp_results["mean_interval_ms"], "ms")
print("Minimum interval:", timestamp_results["min_interval_ms"], "ms")
print("Maximum interval:", timestamp_results["max_interval_ms"], "ms")
# Check voltage conversion
voltage_results = validate_voltage(df["adc_raw"], df["voltage_v"])

print("\n--- Voltage Validation ---")
print("Voltage valid:", voltage_results["voltage_valid"])
print("Maximum error:", round(voltage_results["maximum_error_v"], 6), "V")




time_s = df["time_ms"] / 1000
recording_time = time_s.iloc[-1] - time_s.iloc[0]

print("\n--- Recording Information ---")
print("Total recording time:", round(recording_time, 2), "seconds")
sampling_frequency = (len(df) - 1) / recording_time

print("Sampling frequency:", round(sampling_frequency, 3), "Hz")
print("\n--- EngiSense Analysis Summary ---")
print("Total samples:", len(df))
print("Recording duration:", round(recording_time, 2), "seconds")
print("Sampling frequency:", round(sampling_frequency, 3), "Hz")
print("Mean voltage:", round(voltage.mean(), 3), "V")
print("ADC validation:", adc_valid)
all_valid = (
    adc_valid
    and timestamp_results["timestamps_valid"]
    and voltage_results["voltage_valid"]
)

print("Overall validation:", "PASS" if all_valid else "FAIL")
if all_valid:
    print("EngiSense: Dataset validation successful!")
    report_path = project_root / "reports" / "wokwi_analysis_report.txt"

    with open(report_path, "w") as file:
        file.write("EngiSense Analytics Report\n")
        file.write("-------------------------\n")
        file.write(f"Total samples: {len(df)}\n")
        file.write(f"Recording duration: {recording_time:.2f} seconds\n")
        file.write(f"Sampling frequency: {sampling_frequency:.3f} Hz\n")
        file.write(f"Mean voltage: {voltage.mean():.3f} V\n")
        file.write(f"Overall validation: {'PASS' if all_valid else 'FAIL'}\n")

    print("Analysis report saved successfully!")
    print("Report location:", report_path.resolve())
else:
    print("EngiSense: Dataset validation failed!")

    report_path = project_root / "reports" / "wokwi_analysis_report.txt"





plot_signal(
    time_s,
    voltage,
    "ESP32 Voltage Measurement - Wokwi Simulation",
    "Voltage (V)"
)
