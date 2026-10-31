import pandas as pd




def validate_adc(adc_values):


    for value in adc_values:
        if value < 0 or value > 4095:
            return False

    return True




def validate_timestamps(timestamps):


    timestamps = pd.Series(timestamps)
    intervals = timestamps.diff().dropna()

    timestamps_valid = (intervals > 0).all()

    return {
        "timestamps_valid": bool(timestamps_valid),
        "mean_interval_ms": intervals.mean(),
        "min_interval_ms": intervals.min(),
        "max_interval_ms": intervals.max()
    }




def validate_voltage(adc_values, voltage_values):


    adc = pd.Series(adc_values).reset_index(drop=True)
    voltage = pd.Series(voltage_values).reset_index(drop=True)

    if len(adc) != len(voltage) or len(adc) == 0:
        raise ValueError("ADC and voltage data must have equal, nonzero lengths.")

    expected_voltage = adc * 3.3 / 4095

    voltage_error = abs(expected_voltage - voltage)

    voltage_valid = (voltage_error <= 0.001).all()

    return {
        "voltage_valid": bool(voltage_valid),
        "maximum_error_v": voltage_error.max()
    }




if __name__ == "__main__":


    valid_data = [0, 108, 2017, 4095]
    invalid_data = [0, 108, 5000, 4095]

    print("--- ADC Validation ---")
    print("Valid dataset:", validate_adc(valid_data))
    print("Invalid dataset:", validate_adc(invalid_data))


    test_timestamps = pd.Series([500, 2508, 4516, 6524])

    timestamp_results = validate_timestamps(test_timestamps)

    print("\n--- Timestamp Validation ---")
    print(timestamp_results)


    test_adc = pd.Series([0, 2048, 4095])
    test_voltage = pd.Series([0.000, 1.650, 3.300])

    voltage_results = validate_voltage(test_adc, test_voltage)

    print("\n--- Voltage Validation ---")
    print(voltage_results)
