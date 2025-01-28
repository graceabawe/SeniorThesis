import pyaudio
import wave
import threading
import time
import csv
import numpy as np
import math
import datetime
import usb.core
import usb.util
import os
from tuning import Tuning
import board
from digitalio import DigitalInOut, Direction
from playsound import playsound
import random

# Specify the directory where the output files will be saved
OUTPUT_DIR = "trial10"  # CHANGE this to your desired directory
#7 trial timestamp too early but was still true interaction

# Ensure the directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define parameters for audio recording
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 4  # Set to 4 channels for ReSpeaker (can be 1 for mono, 2 for stereo)
RATE = 16000  # Sample rate (16kHz is typical for voice)
CHUNK = 1024  # Size of each audio chunk
BASE_FILENAME = "output"  # Base name for the output files

# Setup for USB device (ReSpeaker)
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if dev is None:
    print("Device not found. Please ensure the microphone is connected.")
else:
    Mic_tuning = Tuning(dev)

# Initialize PyAudio for audio recording
p = pyaudio.PyAudio()

# Open the microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize a list to hold audio frames
frames = []

# Flag to control the recording
recording = True

# Full paths for the .wav and .csv files
wav_filename = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}.wav")
csv_filename = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}.csv")

# Open CSV file to save microphone data
with open(csv_filename, mode='w', newline='') as csvfile:
    fieldnames = ['Time', 'Direction', 'Voice Detected', 'Decibels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row

    # Define functions for microphone recording and logging
    def record_audio():
        """This function runs in a separate thread to record audio continuously."""
        global recording, frames
        print("Recording started...")
        while recording:
            try:
                # Read audio data from the microphone
                data = stream.read(CHUNK)
                frames.append(data)
            except IOError:
                # Handle any potential audio stream errors
                print("Error reading from audio stream.")
                break

    def stop_recording():
        """This function listens for user input in the main thread to stop recording."""
        global recording
        while recording:
            user_input = input("Press 'q' to stop recording: ")
            if user_input.lower() == 'q':  # If 'q' is entered, stop recording
                recording = False
                print("Stopping...")

    def get_decibel_level():
        """Calculate the decibel level from the audio stream."""
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate RMS value
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Convert to decibels (dB)
        decibels = 20 * math.log10(rms) if rms > 0 else -np.inf
        return decibels

    def log_data():
        """Logs microphone data (direction, voice detection, decibel level) to the CSV file."""
        while recording:
            # Get the current timestamp for the data
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get direction, voice detection, and decibel level
            direction = Mic_tuning.direction
            voice_detected = Mic_tuning.is_voice()
            decibels = get_decibel_level()
            
            # Write data to the CSV file
            writer.writerow({'Time': current_time, 'Direction': direction, 'Voice Detected': voice_detected, 'Decibels': decibels})
            
            # Sleep for a short period before checking again
            time.sleep(0.7)

    # Define functions for the capacitive touch sensor
    pad0_pin = board.D22
    pad1_pin = board.D21
    pad2_pin = board.D17
    pad3_pin = board.D24
    pad4_pin = board.D23

    pad0 = DigitalInOut(pad0_pin)
    pad1 = DigitalInOut(pad1_pin)
    pad2 = DigitalInOut(pad2_pin)
    pad3 = DigitalInOut(pad3_pin)
    pad4 = DigitalInOut(pad4_pin)

    pad0.direction = Direction.INPUT
    pad1.direction = Direction.INPUT
    pad2.direction = Direction.INPUT
    pad3.direction = Direction.INPUT
    pad4.direction = Direction.INPUT

    pad0_already_pressed = True
    pad1_already_pressed = True
    pad2_already_pressed = True
    pad3_already_pressed = True
    pad4_already_pressed = True

    def playSound():
        sounds = ["ommie_sounds/coo1.wav", "ommie_sounds/giggle1.wav", "ommie_sounds/huh1.wav",
                  "ommie_sounds/sigh1.wav", "ommie_sounds/yawn1.wav", "ommie_sounds/giggle2.wav", 
                  "ommie_sounds/huh2.wav", "ommie_sounds/sigh3.wav"]
        random_sound = random.choice(sounds)
        playsound(random_sound)

    firstTouch = False
    touchedTime = None

    def recordTime():
        global firstTouch, touchedTime
        if not firstTouch:
            firstTouch = True
            touchedTime = datetime.datetime.now()
            
            # Save the timestamp to timestampTouched.txt in the specified directory
            timestamp_file_path = os.path.join(OUTPUT_DIR, "timestampTouched.txt")
            
            # Append the timestamp to the file
            with open(timestamp_file_path, "a") as timestamp_file:
                timestamp_file.write(f"{touchedTime}\n")

            print(f"Timestamp saved to {timestamp_file_path}: {touchedTime}")  # Optional: To see it in the terminal

    def touch_sensor():
        global pad0_already_pressed, pad1_already_pressed, pad2_already_pressed, pad3_already_pressed, pad4_already_pressed
        while recording:
            if pad0.value and not pad0_already_pressed:
                print("Pad 0 pressed")
                playSound()
                recordTime()
            pad0_already_pressed = pad0.value

            if pad1.value and not pad1_already_pressed:
                print("Pad 1 pressed")
                playSound()
                recordTime()
            pad1_already_pressed = pad1.value

            if pad2.value and not pad2_already_pressed:
                print("Pad 2 pressed")
                playSound()
                recordTime()
            pad2_already_pressed = pad2.value

            if pad3.value and not pad3_already_pressed:
                print("Pad 3 pressed")
                playSound()
                recordTime()
            pad3_already_pressed = pad3.value

            if pad4.value and not pad4_already_pressed:
                print("Pad 4 pressed")
                playSound()
                recordTime()
            pad4_already_pressed = pad4.value

            time.sleep(0.1)

    # Start threads for microphone and touch sensor
    audio_thread = threading.Thread(target=record_audio)
    logging_thread = threading.Thread(target=log_data)
    touch_thread = threading.Thread(target=touch_sensor)
    input_thread = threading.Thread(target=stop_recording)

    audio_thread.start()
    logging_thread.start()
    touch_thread.start()
    input_thread.start()

    # Wait for threads to finish
    audio_thread.join()
    logging_thread.join()
    touch_thread.join()
    input_thread.join()

    # Stop the audio stream and close the interface
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio as a .wav file
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {wav_filename}")
    print(f"Data saved in {csv_filename}")
