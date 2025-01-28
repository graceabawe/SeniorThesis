import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

# Use:
# yes timestamp: python script_name.py --mic_file mic_data.csv --radar_file radar_data.csv --output_file combined_data.csv --timestamp 14:30:30
# same but remove --timestamp arg


def pad_and_label_data(mic_file, radar_file, output_file, provided_time=None):
    # Load data
    mic_data = pd.read_csv(mic_file)
    radar_data = pd.read_csv(radar_file)

    # Standardize timestamp columns
    mic_data['timestamp'] = pd.to_datetime(mic_data['Time']).dt.time
    radar_data['timestamp'] = pd.to_datetime(radar_data['Timestamp'].str.split('.').str[0], format="%Y-%m-%d %H:%M:%S").dt.time

    # Pad microphone data to match radar data length
    repetitions = len(radar_data) // len(mic_data)  # Full repetitions
    mic_repeated = pd.DataFrame(
        np.repeat(mic_data.values, repetitions, axis=0),
        columns=mic_data.columns
    )

    # Add any extra rows to match exactly
    remainder = len(radar_data) - len(mic_repeated)
    if remainder > 0:
        extra_rows = mic_data.iloc[:remainder]
        mic_repeated = pd.concat([mic_repeated, extra_rows], ignore_index=True)

    mic_repeated["timestamp"] = radar_data["timestamp"].values  # Sync timestamps

    # Merge data
    combined_data = pd.concat([radar_data, mic_repeated.drop(columns=["timestamp"])], axis=1)

    # Add label column
    combined_data["label"] = 0

    # Label based on provided time
    if provided_time is not None:
        provided_time_obj = datetime.strptime(provided_time, "%H:%M:%S").time()
        cutoff_time = (datetime.combine(datetime.min, provided_time_obj) - timedelta(seconds=4)).time()
        combined_data.loc[combined_data["timestamp"] >= cutoff_time, "label"] = 1

    # Get rid of extra timestamp columns
    combined_data = combined_data.drop(columns=["Time", "timestamp"], errors='ignore')

    # Save to output file
    combined_data.to_csv(output_file, index=False)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Pad microphone data, merge with radar data, and label interactions.")
    parser.add_argument("--mic_file", required=True, help="Path to the microphone CSV file.")
    parser.add_argument("--radar_file", required=True, help="Path to the radar CSV file.")
    parser.add_argument("--output_file", required=True, help="Path to save the combined CSV file.")
    parser.add_argument("--timestamp", help="Interaction timestamp in HH:MM:SS format.")

    args = parser.parse_args()

    pad_and_label_data(
        mic_file=args.mic_file,
        radar_file=args.radar_file,
        output_file=args.output_file,
        provided_time=args.timestamp
    )

if __name__ == "__main__":
    main()
