from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent


input_file = (
    project / "data/raw/wokwi/esp32_sensor_data.csv"
)

data = pd.read_csv(input_file)


integrated = data.copy()


integrated["time_s"] = integrated["time_ms"] / 1000


integrated["source"] = "ESP32"


output_folder = project / "data/unified/esp32"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "esp32_integrated.csv"

print("\n--- EngiSense ESP32 Integration ---")
print("Samples:", len(integrated))
print("Columns:", integrated.columns.tolist())

print("\nSensor summary:")
print(
    integrated[
        ["temperature_c", "humidity_pct", "voltage_v"]
    ].describe().round(2)
)

if output_file.exists():
    print("\nIntegrated file already exists; not overwriting.")
else:
    integrated.to_csv(output_file, index=False)
    print("\nSaved to:", output_file)

print("\nESP32 integration completed!")
