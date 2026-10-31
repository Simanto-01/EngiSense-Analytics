import matplotlib.pyplot as plt
from pathlib import Path


def plot_signal(time, values, title, ylabel):


    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        values,
        color="blue",
        linewidth=1.5
    )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.tight_layout()
    project_root = Path(__file__).resolve().parents[2]
    reports_folder = project_root / "reports"
    reports_folder.mkdir(exist_ok=True)

    plt.savefig(reports_folder / "wokwi_voltage_graph.png")
    plt.show()
if __name__ == "__main__":
        test_time = [0, 1, 2, 3, 4]
        test_voltage = [0, 1, 2, 1, 0]

        plot_signal(
            test_time,
            test_voltage,
            "Test Voltage Signal",
            "Voltage (V)"
        )