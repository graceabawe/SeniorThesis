# Copyright (c) Acconeer AB, 2022
# All rights reserved

# How to run: python3 readh5.py test3.h5 --plot --global_start_time "14:06:00"

import argparse
import numpy as np
from datetime import datetime
from acconeer.exptool import a121

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--global_start_time", type=str, required=True, help="Global start time in HH:MM:SS format")
    args = parser.parse_args()
    filename = args.filename

    # Load the radar data
    record = a121.load_record(filename)

    # Parse the global start time
    global_start_time = datetime.strptime(args.global_start_time, "%H:%M:%S")
    radar_start_time = datetime.strptime(record.timestamp.split(" ")[1], "%H:%M:%S.%f")

    # Calculate the offset in seconds
    time_offset = (radar_start_time - global_start_time).total_seconds()

    print_record(record, time_offset)

    if args.plot:
        plot_record(record, time_offset)

def print_record(record: a121.Record, time_offset: float) -> None:
    estimated_update_rate = 1 / np.diff(record.stacked_results.tick_time).mean()

    print("ET version: ", record.lib_version)
    print("RSS version:", record.server_info.rss_version)
    print("HW name:    ", record.server_info.hardware_name)
    print("Data shape: ", record.frames.shape)
    print("Est. rate:  ", f"{estimated_update_rate:.3f} Hz")
    print("Timestamp:  ", record.timestamp)
    print(f"Time offset: {time_offset:.2f} seconds")
    print()
    print(record.session_config)
    print()
    print(record.metadata)
    print()

    first_result = next(record.results)
    print(first_result)

def plot_record(record: a121.Record, time_offset: float) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    z = abs(record.frames.mean(axis=1)).T
    x = record.stacked_results.tick_time + time_offset
    x -= x[0]
    y = np.arange(z.shape[0])

    ax.pcolormesh(x, y, z)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance point")
    ax.set_title("Mean sweep amplitude")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
