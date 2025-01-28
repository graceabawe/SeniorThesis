import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import datetime
import signal
import sys

# Change this to control the folder and file naming
save_num = 4  # Change this value to trial number

# Get the world timestamp at the time the script is first run
timestamp = datetime.datetime.now()

# Create the base output directory if it doesn't exist
output_dir = f"trial{save_num}"
os.makedirs(output_dir, exist_ok=True)

# Create a folder specific to the save_num for this session
save_folder = os.path.join(output_dir, f"videos_{save_num}")
os.makedirs(save_folder, exist_ok=True)

# Video file paths
rgb_video_path = os.path.join(save_folder, f"rgb_{save_num}.avi")
depth_video_path = os.path.join(save_folder, f"depth_{save_num}.avi")
timestamp_file_path = os.path.join(save_folder, f"timestamp_{save_num}.txt")

# Create VideoWriter objects for saving the streams
rgb_writer = cv2.VideoWriter(rgb_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640, 480))
depth_writer = cv2.VideoWriter(depth_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640, 480), isColor=False)

# Write the timestamp to a .txt file
with open(timestamp_file_path, "w") as timestamp_file:
    timestamp_file.write(f"World timestamp (start time): {timestamp}\n")

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream RGB and Depth
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB feed
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth feed

# Start the pipeline
pipeline.start(config)

# Function to handle exit gracefully
def exit_gracefully(signum, frame):
    print("Terminating the recording...")
    pipeline.stop()
    rgb_writer.release()
    depth_writer.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, exit_gracefully)

# Main recording loop
try:
    print(f"Recording started in folder: {save_folder}. \n Press Ctrl+C to stop and save.")
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Get RGB and Depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap to the depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Write the frames to video
        rgb_writer.write(color_image)
        depth_writer.write(depth_image)  # Save the raw depth image (16-bit)

        # Display the frames (optional, can be commented out)
        cv2.imshow("RGB Feed", color_image)
        cv2.imshow("Depth Feed", depth_colormap)

        # Break the loop if the user presses 'q' (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
    exit_gracefully(None, None)
