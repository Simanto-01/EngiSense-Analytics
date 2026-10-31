import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib

print("EngiSense-Analytics Environment Check")
print("-" * 35)
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", matplotlib.__version__)


voltage = np.array([10.0, -10.0, 10.0, -10.0])
rms_voltage = np.sqrt(np.mean(voltage ** 2))

assert np.isclose(rms_voltage, 10.0)

print("-" * 35)
print(f"RMS Voltage: {rms_voltage:.2f} V")
print("Environment ready!")
