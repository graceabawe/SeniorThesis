# SPDX-FileCopyrightText: 2019 Mikey Sklar for Adafruit Industries
#
# SPDX-License-Identifier: MIT

import time
import datetime
import board
from digitalio import DigitalInOut, Direction
from playsound import playsound
import random
import os

# Define the directory where the timestamp file will be saved
# Change this to the desired directory path
output_directory = "trial1"  # Update this path as needed

# Ensure the directory exists (create it if it doesn't)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Set the GPIO input pins
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

# only save the timestamp of the first time ommie is touched
# this is in case users want to interact with ommie more than once within a certain interval
# for a new interaction, run this script again
firstTouch = False
touchedTime = None

def recordTime():
    global firstTouch, touchedTime
    if not firstTouch:
        firstTouch = True
        touchedTime = datetime.datetime.now()
        
        # Save the timestamp to timestampTouched.txt in the specified directory
        timestamp_file_path = os.path.join(output_directory, "timestampTouched.txt")
        
        # Append the timestamp to the file
        with open(timestamp_file_path, "a") as timestamp_file:
            timestamp_file.write(f"{touchedTime}\n")

        print(f"Timestamp saved to {timestamp_file_path}: {touchedTime}")  # Optional: To see it in the terminal

while True:
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
