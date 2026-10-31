import pandas as pd


def calculate_statistics(data):


    signal = pd.Series(data)

    results = {
        "minimum": signal.min(),
        "maximum": signal.max(),
        "mean": signal.mean(),
        "standard_deviation": signal.std()
    }

    return results
if __name__ == "__main__":
    test_data = [1, 2, 3, 4, 5]

    results = calculate_statistics(test_data)

    print("Test results:", results)
